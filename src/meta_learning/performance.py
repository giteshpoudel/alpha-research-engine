"""Track per (strategy, symbol) performance and prune dead candidates.

The search space grows with every strategy; this keeps it productive by marking
combos whose validation is weak and whose paper result is negative as ``dead``,
so the batch tuner skips them, and by disabling generated strategies that never
work.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.change_log import log as log_change
from src.ingestion.schemas import database_name

_COLUMNS = ("strategy", "symbol", "validation_sharpe", "paper_return", "status",
            "updated_at")


def refresh(ch, min_validation: float = 0.0) -> int:
    """Recompute the performance table from tuned_params + paper_equity."""
    validation = {
        (s, sym): float(v) for s, sym, v in ch.query(
            f"SELECT strategy, symbol, argMax(validation_sharpe, valid_from) "
            f"FROM {database_name()}.tuned_params FINAL GROUP BY strategy, symbol"
        ).result_rows
    }
    paper = {
        (s, sym): float(eq) - 1.0 for s, sym, eq in ch.query(
            f"SELECT strategy, symbol, argMax(equity, ts) "
            f"FROM {database_name()}.paper_equity FINAL "
            "WHERE NOT startsWith(strategy, 'TEST_') GROUP BY strategy, symbol"
        ).result_rows
    }
    now = datetime.now(timezone.utc)
    rows = []
    for (strategy, symbol), value in validation.items():
        ret = paper.get((strategy, symbol))
        if ret is None:
            ret = paper.get(("mean_reversion", symbol))  # paper tracks assigned strategy
        dead = value < min_validation and (ret is None or ret < 0)
        rows.append([strategy, symbol, value, ret if ret is not None else 0.0,
                     "dead" if dead else "active", now])
    if not rows:
        return 0
    ch.insert(f"{database_name()}.strategy_performance", rows,
              column_names=list(_COLUMNS))
    return len(rows)


def dead_combos(ch) -> set[tuple[str, str]]:
    rows = ch.query(
        f"SELECT strategy, symbol FROM {database_name()}.strategy_performance FINAL "
        "WHERE status = 'dead'"
    ).result_rows
    return {(s, sym) for s, sym in rows}


def prune_generated(ch) -> list[str]:
    """Disable + unregister generated strategies that are entirely dead."""
    dead_strategies = {s for s, _ in dead_combos(ch)}
    if not dead_strategies:
        return []
    active_generated = {
        r[0] for r in ch.query(
            f"SELECT name FROM {database_name()}.generated_strategies FINAL "
            "WHERE status = 'active'"
        ).result_rows
    }
    pruned = sorted(dead_strategies & active_generated)
    if not pruned:
        return []
    from src.backtesting.strategies.registry import GENERATED_DIR, load_generated

    now = datetime.now(timezone.utc)
    for name in pruned:
        ch.insert(f"{database_name()}.generated_strategies",
                  [[name, "", "pruned (dead)", "disabled", 0.0, "", now, now]],
                  column_names=["name", "code", "description", "status",
                                "validation_sharpe", "model", "created_at", "updated_at"])
        path = GENERATED_DIR / f"{name}.py"
        if path.exists():
            path.unlink()
        log_change(ch, "strategy_pruned", subject=name, detail={"reason": "dead"})
    load_generated()
    return pruned
