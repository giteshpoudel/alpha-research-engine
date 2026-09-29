"""Paper-trading runner: forward-simulate tuned mean reversion on live 1h bars.

Usage:
    python -m src.paper.runner --replay --from 2025-01-01   # rebuild history
    python -m src.paper.runner --once                       # top-up + step

Each symbol is an independent $1.00 sleeve. Fills are assumed at the signal
bar's close (matching the VectorBT backtests) with a 0.1%/side fee. Only closed
bars are used. Parameters come from the versioned ``tuned_params`` table: each
bar uses the version valid at that bar (defaults before the first version).
A sleeve is halted (no new entries, open long force-closed) while its realized
trailing-30d return is negative. Writes are idempotent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtesting import fees
from src.backtesting.data import WINDOWS, load_ohlcv
from src.backtesting.fees import fee_for_symbol
from src.backtesting.runner import DEFAULT_FEE
from src.backtesting.strategies import registry
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe
from src.meta_learning.assignments import assigned_strategy
from src.meta_learning.params import get_tuned_param_history, get_tuned_params
from src.paper import store
from src.paper.executor import SleeveState, step

STRATEGY = "assigned"  # resolve each symbol's strategy from its assignment
INTERVAL = "1h"        # default; actual interval comes from the strategy name
LOOKBACK_BARS = 200    # indicator warmup only; never used as trading history
TRAILING_DAYS = 30
_INTERVAL_DELTA = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1),
                   "1d": timedelta(days=1)}


def _interval_for(client, strategy: str, symbol: str) -> str:
    return registry.interval_of(_strategy_for(client, strategy, symbol))


def _step_delta(interval: str) -> timedelta:
    return _INTERVAL_DELTA.get(interval, timedelta(hours=1))


def _fee_for(requested: float | None, client, symbol: str) -> float:
    if requested is not None:
        return requested
    return fee_for_symbol(client, symbol) + fees.slippage_bps() / 1e4


def _strategy_for(client, requested: str, symbol: str) -> str:
    return assigned_strategy(client, symbol) if requested == "assigned" else requested


def _base_strategy(strategy: str) -> str:
    """Strip TEST_ and @interval so the name dispatches to a real strategy."""
    return registry.base_name(strategy)


def resolve_params(client, strategy: str, symbol: str,
                   as_of: datetime | None = None) -> dict:
    """Latest tuned params valid at ``as_of``, falling back to defaults."""
    tuned = get_tuned_params(client, strategy, symbol, as_of=as_of)
    return dict(tuned) if tuned else registry.defaults(_base_strategy(strategy))


def _signal_series(client, strategy: str, symbol: str,
                   prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Entries/exits where each bar uses its own valid param version.

    Signals are computed once per param version over the full price series
    (so rolling warmup is correct), then selected per bar by valid_from. The
    signal function is whatever strategy this symbol is assigned.
    """
    base = _base_strategy(strategy)
    signal_fn = registry.SIGNAL_FUNCS[base][0]
    history = get_tuned_param_history(client, strategy, symbol)
    default_e, default_x = signal_fn(prices, **registry.defaults(base))
    computed = {vf: signal_fn(prices, **params) for vf, params in history}
    bounds = [vf for vf, _ in history]
    entries = pd.Series(False, index=prices.index, dtype=bool)
    exits = pd.Series(False, index=prices.index, dtype=bool)
    for ts in prices.index:
        selected = None
        for vf in bounds:
            if vf <= ts:
                selected = vf
            else:
                break
        if selected is None:
            entries.at[ts], exits.at[ts] = bool(default_e[ts]), bool(default_x[ts])
        else:
            e, x = computed[selected]
            entries.at[ts], exits.at[ts] = bool(e[ts]), bool(x[ts])
    return entries, exits


def _trailing_return(equity_series: list[tuple[datetime, float]], as_of: datetime,
                     days: int = TRAILING_DAYS) -> float | None:
    """Return since ``as_of - days`` from the last known equity, or None if too short."""
    if not equity_series:
        return None
    cutoff = as_of - timedelta(days=days)
    past = [eq for ts, eq in equity_series if ts <= cutoff]
    if not past or past[-1] <= 0:
        return None
    return equity_series[-1][1] / past[-1] - 1.0


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _run_range(client, strategy: str, symbol: str, fee: float,
               process_from: datetime, latest: datetime,
               end: datetime | None = None,
               initial_state: SleeveState | None = None,
               use_history: bool = False) -> tuple[SleeveState, int]:
    """Step every closed bar in [process_from, end] and persist the results."""
    state = initial_state if initial_state is not None else SleeveState()
    if latest is None:
        return state, 0
    upper = latest if end is None else min(latest, end)
    if process_from > upper:
        return state, 0

    interval = registry.interval_of(strategy)
    load_start = process_from - LOOKBACK_BARS * _step_delta(interval)
    try:
        prices = load_ohlcv(client, symbol, interval=interval,
                            start=load_start, end=upper + _step_delta(interval))["close"]
    except ValueError:
        return state, 0
    entries, exits = _signal_series(client, strategy, symbol, prices)
    lag = fees.execution_lag()
    if lag:
        entries = entries.shift(lag, fill_value=False)
        exits = exits.shift(lag, fill_value=False)

    equity_series: list[tuple[datetime, float]] = (
        store.load_recent_equity(client, strategy, symbol, process_from) if use_history else [])

    mask = prices.index >= pd.Timestamp(process_from)
    equity_rows, trades = [], []
    current_day = None
    enabled = True
    for ts, close in prices[mask].items():
        if ts.date() != current_day:
            current_day = ts.date()
            trailing = _trailing_return(equity_series, ts.to_pydatetime())
            enabled = True if trailing is None else trailing >= 0.0
        state, trade, eq = step(state, ts.to_pydatetime(), float(close),
                                bool(entries.loc[ts]), bool(exits.loc[ts]), fee,
                                enabled=enabled)
        equity_rows.append(eq)
        equity_series.append((ts.to_pydatetime(), eq.equity))
        if trade is not None:
            trades.append(trade)

    if equity_rows:
        store.insert_equity(client, strategy, symbol, equity_rows)
        store.insert_trades(client, strategy, symbol, trades)
        store.upsert_position(client, strategy, symbol, state, equity_rows[-1].ts)
        last_ts = equity_rows[-1].ts
        trailing = _trailing_return(equity_series, last_ts)
        store.upsert_control(client, strategy, symbol, enabled,
                             trailing if trailing is not None else 0.0, last_ts)
    return state, len(equity_rows)


def replay(client, strategy: str, symbol: str, start: datetime,
           end: datetime | None = None, fee: float = DEFAULT_FEE) -> int:
    """Deterministically rebuild one sleeve from ``start`` (starts flat at $1)."""
    latest = store.latest_closed_bar_ts(client, symbol, registry.interval_of(strategy))
    return _run_range(client, strategy, symbol, fee, start, latest, end=end,
                      initial_state=SleeveState(), use_history=False)[1]


def step_forward(client, strategy: str, symbol: str, fee: float = DEFAULT_FEE) -> int:
    """Process closed bars after the last recorded bar for one sleeve."""
    interval = registry.interval_of(strategy)
    latest = store.latest_closed_bar_ts(client, symbol, interval)
    if latest is None:
        return 0
    last = store.last_paper_ts(client, strategy, symbol)
    if last is None:
        process_from, initial = WINDOWS["OOS"][0], SleeveState()
    else:
        process_from, initial = (last + _step_delta(interval),
                                 store.load_state(client, strategy, symbol))
    if process_from > latest:
        return 0
    return _run_range(client, strategy, symbol, fee, process_from, latest,
                      initial_state=initial, use_history=True)[1]


def _topup(client, symbols: tuple[str, ...], intervals: tuple[str, ...] = ("1h",)) -> None:
    """Incrementally refresh the latest OHLCV bars before stepping."""
    from src.ingestion.binance_us import BINANCE_US_SYMBOLS, backfill_ohlcv
    from src.ingestion.coinbase import backfill_coinbase, backfill_matic

    start = datetime.now(timezone.utc) - timedelta(days=3)
    binance = tuple(s for s in symbols if s != "MATIC" and s in BINANCE_US_SYMBOLS)
    if binance:
        try:
            backfill_ohlcv(client, symbols=binance, intervals=intervals, start=start)
        except Exception as exc:
            print(f"paper: price top-up failed (binance_us): {exc}")
    coinbase_only = tuple(s for s in symbols if s != "MATIC" and s not in BINANCE_US_SYMBOLS)
    if coinbase_only:
        try:
            backfill_coinbase(client, symbols=coinbase_only, intervals=intervals, start=start)
        except Exception as exc:
            print(f"paper: price top-up failed (coinbase): {exc}")
    if "MATIC" in symbols:
        try:
            backfill_matic(client, intervals=intervals, start=start)
        except Exception as exc:
            print(f"paper: price top-up failed (coinbase/matic): {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper-trading forward simulation")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--fee", type=float, default=None,
                        help="override fee; default is per-interval (see fees.py)")
    parser.add_argument("--once", action="store_true", help="incremental top-up + forward step")
    parser.add_argument("--replay", action="store_true", help="rebuild history from --from")
    parser.add_argument("--from", dest="start", default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--no-topup", action="store_true", help="skip the price top-up")
    args = parser.parse_args(argv)
    if not args.once and not args.replay:
        args.once = True

    client = get_clickhouse_client()
    symbols = universe(client) if args.symbols == "all" else tuple(args.symbols.split(","))

    if args.replay:
        start = _parse_date(args.start)
        end = _parse_date(args.end) if args.end else None
        for symbol in symbols:
            try:
                strategy = _strategy_for(client, args.strategy, symbol)
                n = replay(client, strategy, symbol, start, end,
                           fee=_fee_for(args.fee, client, symbol))
                print(f"paper replay {symbol} [{strategy}]: {n} bars")
            except Exception as exc:
                print(f"paper replay {symbol}: FAILED: {exc}")
        return

    if not args.no_topup:
        intervals = tuple(sorted({_interval_for(client, args.strategy, s)
                                  for s in symbols} | {"1h"}))
        _topup(client, symbols, intervals=intervals)
    for symbol in symbols:
        try:
            strategy = _strategy_for(client, args.strategy, symbol)
            n = step_forward(client, strategy, symbol, fee=_fee_for(args.fee, client, symbol))
            print(f"paper step {symbol} [{strategy}]: {n} bars")
        except Exception as exc:
            print(f"paper step {symbol}: FAILED: {exc}")


if __name__ == "__main__":
    main()
