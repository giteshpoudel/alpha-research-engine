"""Backtest runner: windows x symbols x strategies -> ClickHouse results.

Usage:
    python -m src.backtesting.runner --strategy mean_reversion --symbols BTC,ETH --window IS
    python -m src.backtesting.runner --strategy all --symbols all --window ALL

run_id is a deterministic hash of strategy+symbol+interval+params+window+
dates, so re-running collapses under ReplacingMergeTree (idempotent).
No optimization: strategies always run on their fixed default parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from src.backtesting.data import (
    WINDOWS,
    load_funding,
    load_ohlcv,
    load_sentiment,
    slice_window,
)
from src.backtesting.engine import (
    run_funding_backtest,
    run_signal_backtest,
)
from src.backtesting.strategies import mean_reversion, sentiment_momentum
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES

STRATEGIES = ("mean_reversion", "funding_arb", "sentiment_momentum")
DEFAULT_FEE = 0.001
FUNDING_EARLIEST = datetime(2023, 5, 12, tzinfo=timezone.utc)

_RUN_COLUMNS = (
    "run_id", "strategy", "symbol", "interval", "params_json", "window",
    "start_ts", "end_ts", "total_return", "sharpe", "sortino",
    "max_drawdown", "calmar", "win_rate", "num_trades", "created_at",
)


def compute_run_id(strategy: str, symbol: str, interval: str, params_json: str,
                   window: str, start_ts: datetime, end_ts: datetime) -> str:
    payload = "|".join([
        strategy, symbol, interval, params_json, window,
        start_ts.isoformat(), end_ts.isoformat(),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _execute(ch_client, strategy: str, symbol: str, window: str, fee: float):
    """Run one backtest; returns (params_json, start_ts, end_ts, result)."""
    base_strategy = strategy.removeprefix("TEST_")
    if base_strategy == "funding_arb":
        start, end = WINDOWS[window]
        start = max(start, FUNDING_EARLIEST)
        # Load from start unbounded, then slice: loader `end` is exclusive but
        # WINDOWS ends are inclusive candle timestamps (boundary convention).
        funding = slice_window(load_funding(ch_client, symbol, start=start), window)
        result = run_funding_backtest(funding, fee=fee)
        params = {"threshold": 0.0001}
    elif base_strategy == "sentiment_momentum":
        # Sentiment history now spans the OOS window (backfilled 2025-09+);
        # evaluate on whatever window is requested. Before the first metric the
        # signal forward-fills to NaN -> no trades, which is honest.
        start, _ = WINDOWS[window]
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], window)
        sentiment = load_sentiment(ch_client, symbol)["weighted_score"]
        entries, exits = sentiment_momentum.signals(prices, sentiment, **sentiment_momentum.SM_DEFAULTS)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
        params = dict(sentiment_momentum.SM_DEFAULTS)
    else:
        start, end = WINDOWS[window]
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], window)
        entries, exits = mean_reversion.signals(prices, **mean_reversion.MR_DEFAULTS)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
        params = dict(mean_reversion.MR_DEFAULTS)
    equity = result.equity_curve
    return json.dumps(params), equity.index[0].to_pydatetime(), equity.index[-1].to_pydatetime(), result, window


def store_result(ch_client, strategy: str, symbol: str, interval: str,
                 params_json: str, window: str, start_ts: datetime,
                 end_ts: datetime, result) -> str:
    """Compute run_id and insert one backtest_runs row plus equity rows. Returns run_id."""
    run_id = compute_run_id(strategy, symbol, interval, params_json, window, start_ts, end_ts)
    now = datetime.now(timezone.utc)
    db = database_name()
    ch_client.insert(
        f"{db}.backtest_runs",
        [[run_id, strategy, symbol, interval, params_json, window, start_ts, end_ts,
          result.total_return, result.sharpe, result.sortino, result.max_drawdown,
          result.calmar, result.win_rate, result.num_trades, now]],
        column_names=list(_RUN_COLUMNS),
    )
    equity_rows = [
        [run_id, ts.to_pydatetime(), float(value)]
        for ts, value in result.equity_curve.items()
    ]
    ch_client.insert(
        f"{db}.backtest_equity", equity_rows,
        column_names=["run_id", "ts", "equity"],
    )
    return run_id


def run_backtest(ch_client, strategy: str, symbol: str, window: str,
                 fee: float = DEFAULT_FEE) -> str:
    """Run one (strategy, symbol, window) backtest and store results. Returns run_id."""
    params_json, start_ts, end_ts, result, window = _execute(ch_client, strategy, symbol, window, fee)
    return store_result(ch_client, strategy, symbol, "1h", params_json, window,
                        start_ts, end_ts, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests and store results")
    parser.add_argument("--strategy", default="all",
                        choices=[*STRATEGIES, "all"])
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--window", default="ALL", choices=["IS", "OOS", "PRELIM", "ALL"])
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = parser.parse_args(argv)

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    windows = ("IS", "OOS") if args.window == "ALL" else (args.window,)

    ch_client = get_clickhouse_client()
    for strategy in strategies:
        for symbol in symbols:
            for window in windows:
                if strategy == "sentiment_momentum" and window == "IS":
                    continue  # no sentiment history in IS
                try:
                    run_id = run_backtest(ch_client, strategy, symbol, window, fee=args.fee)
                    print(f"{strategy} {symbol} {window}: done ({run_id})")
                except Exception as exc:
                    print(f"{strategy} {symbol} {window}: FAILED: {exc}")


if __name__ == "__main__":
    main()
