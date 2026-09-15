"""CLI: python -m src.research.daily_report [--date YYYY-MM-DD]"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from src.ingestion.schemas import get_clickhouse_client
from src.research.graph import run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the daily research report")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="report date (default: today UTC)")
    args = parser.parse_args(argv)
    report_date = (date.fromisoformat(args.date) if args.date
                   else datetime.now(timezone.utc).date())
    md = run_report(get_clickhouse_client(), report_date)
    print(f"report for {report_date}: {len(md)} chars written")


if __name__ == "__main__":
    main()
