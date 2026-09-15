"""Paper-trading runner: forward-simulate tuned mean reversion on live 1h bars.

Usage:
    python -m src.paper.runner --replay --from 2025-01-01   # rebuild history
    python -m src.paper.runner --once                       # top-up + step

Each symbol is an independent $1.00 sleeve. Fills are assumed at the signal
bar's close (matching the VectorBT backtests) with a 0.1%/side fee. Only closed
bars are used. Writes are idempotent, so replay/step can be re-run freely.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtesting.data import WINDOWS, load_ohlcv
from src.backtesting.runner import DEFAULT_FEE
from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES
from src.meta_learning.params import get_tuned_params
from src.paper import store
from src.paper.executor import SleeveState, step

STRATEGY = "mean_reversion"
INTERVAL = "1h"
LOOKBACK_BARS = 200  # indicator warmup only; never used as trading history


def resolve_params(client, strategy: str, symbol: str) -> dict:
    """Latest tuned params, falling back to the strategy defaults."""
    tuned = get_tuned_params(client, strategy, symbol)
    return dict(tuned) if tuned else dict(mean_reversion.MR_DEFAULTS)


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _run_range(client, strategy: str, symbol: str, fee: float,
               process_from: datetime, latest: datetime,
               end: datetime | None = None,
               initial_state: SleeveState | None = None) -> tuple[SleeveState, int]:
    """Step every closed bar in [process_from, end] and persist the results."""
    state = initial_state if initial_state is not None else SleeveState()
    if latest is None:
        return state, 0
    upper = latest if end is None else min(latest, end)
    if process_from > upper:
        return state, 0

    params = resolve_params(client, strategy, symbol)
    load_start = process_from - timedelta(hours=LOOKBACK_BARS)
    try:
        prices = load_ohlcv(client, symbol, interval=INTERVAL,
                            start=load_start, end=upper + timedelta(hours=1))["close"]
    except ValueError:
        return state, 0
    entries, exits = mean_reversion.signals(prices, **params)

    mask = prices.index >= pd.Timestamp(process_from)
    equity_rows, trades = [], []
    for ts, close in prices[mask].items():
        state, trade, eq = step(state, ts.to_pydatetime(), float(close),
                                bool(entries.loc[ts]), bool(exits.loc[ts]), fee)
        equity_rows.append(eq)
        if trade is not None:
            trades.append(trade)

    if equity_rows:
        store.insert_equity(client, strategy, symbol, equity_rows)
        store.insert_trades(client, strategy, symbol, trades)
        store.upsert_position(client, strategy, symbol, state, equity_rows[-1].ts)
    return state, len(equity_rows)


def replay(client, strategy: str, symbol: str, start: datetime,
           end: datetime | None = None, fee: float = DEFAULT_FEE) -> int:
    """Deterministically rebuild one sleeve from ``start`` (starts flat at $1)."""
    latest = store.latest_closed_bar_ts(client, symbol, INTERVAL)
    return _run_range(client, strategy, symbol, fee, start, latest, end=end,
                      initial_state=SleeveState())[1]


def step_forward(client, strategy: str, symbol: str, fee: float = DEFAULT_FEE) -> int:
    """Process closed bars after the last recorded bar for one sleeve."""
    latest = store.latest_closed_bar_ts(client, symbol, INTERVAL)
    if latest is None:
        return 0
    last = store.last_paper_ts(client, strategy, symbol)
    if last is None:
        process_from, initial = WINDOWS["OOS"][0], SleeveState()
    else:
        process_from, initial = last + timedelta(hours=1), store.load_state(client, strategy, symbol)
    if process_from > latest:
        return 0
    return _run_range(client, strategy, symbol, fee, process_from, latest,
                      initial_state=initial)[1]


def _topup(client, symbols: tuple[str, ...]) -> None:
    """Incrementally refresh the latest 1h OHLCV bars before stepping."""
    from src.ingestion.binance_us import backfill_ohlcv
    from src.ingestion.coinbase import backfill_matic

    start = datetime.now(timezone.utc) - timedelta(days=3)
    binance = tuple(s for s in symbols if s != "MATIC")
    if binance:
        try:
            backfill_ohlcv(client, symbols=binance, intervals=("1h",), start=start)
        except Exception as exc:
            print(f"paper: price top-up failed (binance_us): {exc}")
    if "MATIC" in symbols:
        try:
            backfill_matic(client, intervals=("1h",), start=start)
        except Exception as exc:
            print(f"paper: price top-up failed (coinbase): {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper-trading forward simulation")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--once", action="store_true", help="incremental top-up + forward step")
    parser.add_argument("--replay", action="store_true", help="rebuild history from --from")
    parser.add_argument("--from", dest="start", default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--no-topup", action="store_true", help="skip the price top-up")
    args = parser.parse_args(argv)
    if not args.once and not args.replay:
        args.once = True

    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    client = get_clickhouse_client()

    if args.replay:
        start = _parse_date(args.start)
        end = _parse_date(args.end) if args.end else None
        for symbol in symbols:
            try:
                n = replay(client, args.strategy, symbol, start, end, fee=args.fee)
                print(f"paper replay {symbol}: {n} bars")
            except Exception as exc:
                print(f"paper replay {symbol}: FAILED: {exc}")
        return

    if not args.no_topup:
        _topup(client, symbols)
    for symbol in symbols:
        try:
            n = step_forward(client, args.strategy, symbol, fee=args.fee)
            print(f"paper step {symbol}: {n} bars")
        except Exception as exc:
            print(f"paper step {symbol}: FAILED: {exc}")


if __name__ == "__main__":
    main()
