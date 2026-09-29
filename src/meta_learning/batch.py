"""Parallel batch tuning across strategy x symbol, with champion/challenger adoption.

This is the throughput engine: it tunes many (strategy, symbol) combos
concurrently across CPU cores (each worker owns a ClickHouse client), then the
parent adopts each candidate only if it beats the incumbent on the SAME
validation folds (returned by the worker).

Usage:
    python -m src.meta_learning.batch [--trials 30] [--workers 8]
                                      [--symbols all] [--strategies all]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

from src.agents.change_log import log as log_change
from src.backtesting.strategies.registry import STRATEGY_NAMES
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe
from src.meta_learning.tuner import (
    DEFAULT_ADOPT_MARGIN,
    adopt_candidate,
    publish,
)


def _tune_combo(args: tuple[str, str, int]) -> dict:
    """Worker: tune one combo and validate candidate vs incumbent on the same folds."""
    strategy, symbol, n_trials = args
    from src.ingestion.schemas import get_clickhouse_client
    from src.meta_learning.params import get_tuned_params
    from src.meta_learning.tuner import fit, validate_params

    ch = get_clickhouse_client()
    record, folds = fit(ch, strategy, symbol, n_trials=n_trials)
    candidate_val = validate_params(ch, strategy, symbol, record["params"], folds)
    incumbent = get_tuned_params(ch, strategy, symbol)
    incumbent_val = (validate_params(ch, strategy, symbol, incumbent, folds)
                     if incumbent else None)
    return {"strategy": strategy, "symbol": symbol, "params": record["params"],
            "folds": record["folds"], "candidate_val": candidate_val,
            "incumbent_val": incumbent_val}


def run_batch(ch, combos: list[tuple[str, str]] | None = None, n_trials: int = 30,
              workers: int | None = None, margin: float = DEFAULT_ADOPT_MARGIN,
              use_lock: bool = True, prune: bool = True) -> dict:
    from src.agents import governor
    from src.meta_learning import performance

    if combos is None:
        symbols = universe(ch)
        combos = [(s, sym) for s in STRATEGY_NAMES for sym in symbols]

    # Skip combos already known dead.
    try:
        dead = performance.dead_combos(ch)
        combos = [c for c in combos if c not in dead]
    except Exception:
        pass

    workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))

    def _run_pool() -> list[dict]:
        out: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_tune_combo, [(s, sym, n_trials) for s, sym in combos]):
                out.append(result)
        return out

    if use_lock:
        with governor.heavy("batch") as acquired:
            if not acquired:
                return {"combos": 0, "adopted": 0, "workers": workers,
                        "skipped": "compute lock busy", "results": []}
            results = _run_pool()
    else:
        results = _run_pool()

    adopted = 0
    for res in results:
        if res["candidate_val"] is None:
            continue
        if not adopt_candidate(res["candidate_val"], res["incumbent_val"], margin=margin):
            continue
        publish(ch, res["strategy"], res["symbol"],
                {"params": res["params"], "train_sharpe": 0.0,
                 "validation_sharpe": res["candidate_val"], "folds": res["folds"]},
                datetime.now(timezone.utc))
        log_change(ch, "params_adopted", subject=res["symbol"],
                   detail={"strategy": res["strategy"], "params": res["params"],
                           "candidate_val": res["candidate_val"],
                           "incumbent_val": res["incumbent_val"], "source": "batch"})
        adopted += 1

    if prune:
        try:
            performance.refresh(ch)
            performance.prune_generated(ch)
        except Exception:
            pass

    summary = [{"strategy": r["strategy"], "symbol": r["symbol"],
                "candidate_val": r["candidate_val"], "incumbent_val": r["incumbent_val"]}
               for r in results]
    return {"combos": len(results), "adopted": adopted, "workers": workers,
            "results": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parallel batch tuning (strategy x symbol)")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--strategies", default="all")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    strategies = (STRATEGY_NAMES if args.strategies == "all"
                  else tuple(args.strategies.split(",")))
    combos = [(s, sym) for s in strategies for sym in symbols]
    print(f"batch: {len(combos)} combos x {args.trials} trials "
          f"(workers={args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))})")
    result = run_batch(ch, combos=combos, n_trials=args.trials, workers=args.workers)
    print(f"batch: adopted {result['adopted']}/{result['combos']}")


if __name__ == "__main__":
    main()
