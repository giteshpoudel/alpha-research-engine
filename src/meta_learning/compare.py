"""OOS comparison simulation: static vs tuned vs causal allocator.

No optimization anywhere in this module — it consumes tuned_params produced
by tuner.py and evaluates on the OOS window only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtesting.data import WINDOWS, load_funding, load_ohlcv, slice_window
from src.backtesting.engine import (
    BacktestResult,
    _metrics,
    run_funding_backtest,
    run_signal_backtest,
)
from src.backtesting.runner import store_result
from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES
from src.meta_learning.allocator import allocator_windows, pick_strategy

DEFAULT_FEE = 0.001
_REBALANCE_DAYS = 7
_TRAILING_DAYS = 30


def _tuned_params(ch_client, strategy: str, symbol: str) -> dict:
    rows = ch_client.query(
        f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} LIMIT 1",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    if not rows:
        raise ValueError(f"no tuned params for {strategy} {symbol} — run tuner first")
    return json.loads(rows[0][0])


def _execute_variant(ch_client, strategy: str, symbol: str, params: dict, fee: float):
    """OOS backtest with explicit params. Returns (start_ts, end_ts, result)."""
    base_strategy = strategy.removeprefix("TEST_")
    start, _ = WINDOWS["OOS"]
    if base_strategy == "funding_arb":
        funding = slice_window(load_funding(ch_client, symbol, start=start), "OOS")
        result = run_funding_backtest(funding, threshold=params["threshold"], fee=fee)
    else:
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], "OOS")
        entries, exits = mean_reversion.signals(prices, **params)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
    equity = result.equity_curve
    return equity.index[0].to_pydatetime(), equity.index[-1].to_pydatetime(), result


def run_variant(ch_client, strategy: str, symbol: str, variant: str,
                params: dict, fee: float = DEFAULT_FEE) -> str:
    """Store one OOS variant run. Returns run_id."""
    start_ts, end_ts, result = _execute_variant(ch_client, strategy, symbol, params, fee)
    params_json = json.dumps({"variant": variant, **params})
    return store_result(ch_client, strategy, symbol, "1h", params_json, "OOS",
                        start_ts, end_ts, result)


def _segment_return(result: BacktestResult) -> float:
    return result.total_return


def run_allocator(ch_client, symbol: str, fee: float = DEFAULT_FEE,
                  strategy: str = "allocator") -> str:
    """Causal composite: at each rebalance, run the strategy with the best
    trailing-30-day return (computed strictly from data before the rebalance
    point). Stored under `strategy` with variant 'allocator'. When `strategy`
    is TEST_-prefixed, tuned params are read from the TEST_-prefixed strategy
    rows so tests stay isolated from production data."""
    test_run = strategy.startswith("TEST_")
    prefix = "TEST_" if test_run else ""
    oos_start, _ = WINDOWS["OOS"]
    prices = slice_window(load_ohlcv(ch_client, symbol, start=oos_start)["close"], "OOS")
    funding = slice_window(load_funding(ch_client, symbol, start=oos_start), "OOS")
    mr_params = _tuned_params(ch_client, f"{prefix}mean_reversion", symbol)
    fa_params = _tuned_params(ch_client, f"{prefix}funding_arb", symbol)

    oos_end = prices.index[-1].to_pydatetime()
    equity_points: dict[pd.Timestamp, float] = {pd.Timestamp(oos_start): 1.0}
    composite = 1.0
    segment_pnls: list[float] = []
    for win_start, win_end in allocator_windows(oos_start, oos_end, _REBALANCE_DAYS):
        trail_start = win_start - timedelta(days=_TRAILING_DAYS)
        trailing = {}
        if trail_start >= oos_start:
            trail_mr = prices.loc[trail_start:win_start]
            trail_fa = funding.loc[trail_start:win_start]
            trailing["mean_reversion"] = (
                _segment_return(run_signal_backtest(
                    trail_mr, *mean_reversion.signals(trail_mr, **mr_params), fee=fee))
                if len(trail_mr) > 48 else None
            )
            trailing["funding_arb"] = (
                _segment_return(run_funding_backtest(trail_fa, threshold=fa_params["threshold"], fee=fee))
                if len(trail_fa) > 0 else None
            )
        else:
            trailing = {"mean_reversion": None, "funding_arb": None}
        picked = pick_strategy(trailing)

        seg_prices = prices.loc[win_start:win_end]
        if picked == "funding_arb":
            seg_data = funding.loc[win_start:win_end]
            seg_result = run_funding_backtest(seg_data, threshold=fa_params["threshold"], fee=fee)
        else:
            seg_result = run_signal_backtest(
                seg_prices, *mean_reversion.signals(seg_prices, **mr_params), fee=fee)
        composite *= 1.0 + seg_result.total_return
        segment_pnls.append(seg_result.total_return)
        equity_points[pd.Timestamp(win_end)] = composite

    equity = pd.Series(equity_points).sort_index()
    result = _metrics(equity, segment_pnls)
    bad = ([(str(ts), v) for ts, v in equity_points.items() if not math.isfinite(v)]
           + [(f"segment_{i}", v) for i, v in enumerate(segment_pnls) if not math.isfinite(v)])
    if bad:
        raise ValueError(f"allocator {symbol}: non-finite values, refusing to store: {bad}")
    start_ts = equity.index[0].to_pydatetime()
    end_ts = equity.index[-1].to_pydatetime()
    params_json = json.dumps({
        "variant": "allocator", "rebalance_days": _REBALANCE_DAYS,
        "trailing_days": _TRAILING_DAYS,
    })
    return store_result(ch_client, strategy, symbol, "1h", params_json, "OOS",
                        start_ts, end_ts, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OOS comparison: static vs tuned vs allocator")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = parser.parse_args(argv)

    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    ch_client = get_clickhouse_client()
    print(f"{'symbol':<8}{'variant':<12}{'total_return':>13}{'sharpe':>9}{'max_dd':>9}")
    for symbol in symbols:
        for strategy in ("mean_reversion", "funding_arb"):
            try:
                tuned = _tuned_params(ch_client, strategy, symbol)
                static = (dict(mean_reversion.MR_DEFAULTS) if strategy == "mean_reversion"
                          else {"threshold": 0.0001})
                for variant, params in (("static", static), ("tuned", tuned)):
                    start_ts, end_ts, result = _execute_variant(ch_client, strategy, symbol, params, args.fee)
                    params_json = json.dumps({"variant": variant, **params})
                    store_result(ch_client, strategy, symbol, "1h", params_json, "OOS",
                                 start_ts, end_ts, result)
                    print(f"{symbol:<8}{strategy+'/'+variant:<12}"
                          f"{result.total_return:>13.3f}{result.sharpe:>9.3f}{result.max_drawdown:>9.3f}")
            except Exception as exc:
                print(f"{symbol:<8}{strategy}: FAILED: {exc}")
        try:
            run_allocator(ch_client, symbol, fee=args.fee)
            print(f"{symbol:<8}allocator: done")
        except Exception as exc:
            print(f"{symbol:<8}allocator: FAILED: {exc}")


if __name__ == "__main__":
    main()
