"""Pipeline orchestrator: reddit -> rss -> score -> aggregate.

Usage:
    python -m src.ingestion.pipeline --once          # single cycle
    python -m src.ingestion.pipeline --interval 300  # loop (default 300s)

A failing stage is logged and skipped; later stages still run.
"""

from __future__ import annotations

import argparse
import time

from src.ingestion.aggregator import run_aggregation
from src.ingestion.reddit import poll_subreddits
from src.ingestion.rss import poll_feeds
from src.ingestion.schemas import get_clickhouse_client, get_qdrant_client
from src.ingestion.scorer import score_pending_posts


def run_cycle(ch_client=None, qd_client=None) -> dict[str, int]:
    ch_client = ch_client or get_clickhouse_client()
    qd_client = qd_client or get_qdrant_client()
    stages = (
        ("reddit", lambda: poll_subreddits(ch_client)),
        ("rss", lambda: poll_feeds(ch_client)),
        ("score", lambda: score_pending_posts(ch_client, qd_client)),
        ("aggregate", lambda: run_aggregation(ch_client)),
    )
    stats: dict[str, int] = {}
    for name, fn in stages:
        try:
            stats[name] = fn()
            print(f"pipeline: {name} done ({stats[name]})")
        except Exception as exc:
            stats[name] = -1
            print(f"pipeline: stage {name} failed: {exc}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sentiment ingestion & aggregation pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="run a single cycle and exit")
    group.add_argument("--interval", type=int, default=300,
                       help="seconds between cycles (default 300)")
    args = parser.parse_args(argv)
    if args.once:
        run_cycle()
        return
    while True:
        run_cycle()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
