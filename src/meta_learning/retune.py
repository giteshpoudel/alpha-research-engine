"""Rolling walk-forward re-tune: advance the IS window and publish new params.

At deploy time ``T``, tune on data strictly before ``T - embargo`` and publish
the result valid from ``T``. The embargo guarantees the live period after ``T``
is never tuned on (guardrail #1).

Usage:
    python -m src.meta_learning.retune [--embargo-days 7] [--months 36]
                                       [--symbols all] [--trials 60]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES
from src.meta_learning.params import get_tuned_params
from src.meta_learning.tuner import (
    TUNABLE_STRATEGIES,
    add_months,
    adopt_candidate,
    fit,
    publish,
    validate_params,
)

DEFAULT_EMBARGO_DAYS = 7
DEFAULT_MONTHS = 36


def retune_window(now: datetime, embargo_days: int = DEFAULT_EMBARGO_DAYS,
                  months: int = DEFAULT_MONTHS) -> tuple[datetime, datetime, datetime]:
    """Return (deploy, is_start, is_end) for a rolling re-tune at ``now``."""
    deploy = now.replace(minute=0, second=0, microsecond=0)
    is_end = deploy - timedelta(days=embargo_days)
    is_start = add_months(is_end, -months)
    return deploy, is_start, is_end


def run_retune(ch_client, symbols: tuple[str, ...] | None = None,
               embargo_days: int = DEFAULT_EMBARGO_DAYS, months: int = DEFAULT_MONTHS,
               n_trials: int = 60) -> dict:
    symbols = tuple(symbols) if symbols else tuple(TICKER_ALIASES)
    deploy, is_start, is_end = retune_window(datetime.now(timezone.utc), embargo_days, months)
    print(f"retune: deploy={deploy.isoformat()} train=[{is_start.isoformat()}, {is_end.isoformat()})")
    results: dict = {}
    for strategy in TUNABLE_STRATEGIES:
        for symbol in symbols:
            try:
                candidate, folds = fit(ch_client, strategy, symbol, n_trials=n_trials,
                                       is_start=is_start, is_end_exclusive=is_end)
                # Champion/challenger: compare candidate and incumbent on the
                # SAME validation folds (a candidate only wins if it is not
                # worse than the live params on a like-for-like basis).
                candidate_val = validate_params(ch_client, strategy, symbol, candidate["params"], folds)
                incumbent = get_tuned_params(ch_client, strategy, symbol, as_of=deploy)
                incumbent_val = (validate_params(ch_client, strategy, symbol, incumbent, folds)
                                 if incumbent else None)
                if adopt_candidate(candidate_val, incumbent_val):
                    publish(ch_client, strategy, symbol, candidate, deploy)
                    verdict = "ADOPTED"
                else:
                    verdict = "rejected"
                results[f"{strategy} {symbol}"] = {
                    "adopted": verdict == "ADOPTED", "candidate": candidate,
                    "candidate_val": candidate_val, "incumbent_val": incumbent_val,
                }
                print(f"{strategy} {symbol}: {verdict} "
                      f"(candidate {candidate_val:+.3f} vs incumbent "
                      f"{'none' if incumbent_val is None else f'{incumbent_val:+.3f}'})")
            except Exception as exc:
                print(f"{strategy} {symbol}: FAILED: {exc}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rolling walk-forward re-tune (IS with embargo)")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args(argv)
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    run_retune(get_clickhouse_client(), symbols=symbols, embargo_days=args.embargo_days,
               months=args.months, n_trials=args.trials)


if __name__ == "__main__":
    main()
