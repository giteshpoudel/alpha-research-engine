"""Backfill orchestrator: OHLCV (binance_us + coinbase MATIC) + funding (hyperliquid).

Usage:
    python -m src.ingestion.backfill                        # everything from 2022-01-01
    python -m src.ingestion.backfill --symbols BTC ETH      # restrict tickers
    python -m src.ingestion.backfill --intervals 1h         # restrict ohlcv intervals
    python -m src.ingestion.backfill --start 2024-01-01     # floor for fresh series

Every series resumes from its stored max_ts, so this is also the ongoing
top-up job: run it any time to bring all series up to the latest closed
candle. A failing source is logged and skipped; the others still run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.ingestion.binance_us import backfill_ohlcv
from src.ingestion.coinbase import backfill_matic
from src.ingestion.hyperliquid import backfill_funding
from src.ingestion.schemas import get_clickhouse_client


def run_backfill(ch_client, symbols: list[str] | tuple[str, ...] | None = None,
                 intervals: tuple[str, ...] = ("1h", "1d"),
                 start: datetime | None = None) -> dict[str, int]:
    start = start or datetime(2022, 1, 1, tzinfo=timezone.utc)
    binance_symbols = tuple(s for s in symbols if s != "MATIC") if symbols else None
    want_matic = symbols is None or "MATIC" in symbols
    funding_coins = tuple(symbols) if symbols else None

    jobs: list[tuple[str, object]] = []
    if symbols is None or binance_symbols:
        jobs.append(("binance_us", lambda: backfill_ohlcv(
            ch_client, symbols=binance_symbols, intervals=intervals, start=start)))
    if want_matic:
        jobs.append(("coinbase_matic", lambda: backfill_matic(
            ch_client, intervals=intervals, start=start)))
    jobs.append(("hyperliquid_funding", lambda: backfill_funding(
        ch_client, coins=funding_coins, start=start)))

    stats: dict[str, int] = {}
    for name, fn in jobs:
        try:
            stats[name] = fn()
            print(f"backfill: {name} done ({stats[name]})")
        except Exception as exc:
            stats[name] = -1
            print(f"backfill: {name} failed: {exc}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV + funding-rate history")
    parser.add_argument("--symbols", nargs="+", metavar="TICKER", default=None,
                        help="restrict to these tickers (default: all 12)")
    parser.add_argument("--intervals", nargs="+", choices=["1h", "1d"], default=["1h", "1d"],
                        help="ohlcv intervals (default: 1h 1d)")
    parser.add_argument("--start", default="2022-01-01", metavar="YYYY-MM-DD",
                        help="floor date for fresh series (default 2022-01-01)")
    args = parser.parse_args(argv)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    run_backfill(get_clickhouse_client(), symbols=args.symbols,
                 intervals=tuple(args.intervals), start=start)


if __name__ == "__main__":
    main()
