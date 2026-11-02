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


# --- generated-strategy backlog control -------------------------------------
# Synthesis can outpace tuning. Without these caps the batch tuner tried every
# generated strategy at once (31k combos), never finished, and held the compute
# lock so nothing else could run.

ARCHIVE_DIR = "data/generated_archive"


def _active_generated(ch) -> list[tuple[str, datetime]]:
    return [(r[0], r[1]) for r in ch.query(
        f"SELECT name, created_at FROM {database_name()}.generated_strategies FINAL "
        "WHERE status = 'active' ORDER BY created_at DESC"
    ).result_rows]


def _tuned_validation(ch) -> dict[str, float]:
    """Best latest validation Sharpe per strategy (any symbol)."""
    return {s: float(v) for s, v in ch.query(
        f"SELECT strategy, max(v) FROM (SELECT strategy, symbol, "
        f"argMax(validation_sharpe, valid_from) AS v "
        f"FROM {database_name()}.tuned_params FINAL GROUP BY strategy, symbol) "
        "GROUP BY strategy"
    ).result_rows}


def untested_generated(ch) -> list[str]:
    """Active generated strategies with no tuned_params row yet, newest first."""
    tuned = _tuned_validation(ch)
    return [name for name, _ in _active_generated(ch) if name not in tuned]


def select_generated(ch, limit: int) -> list[str]:
    """Generated strategies worth tuning in one batch, at most ``limit``.

    Half the slots go to the best already-tuned ones (positive validation),
    the rest to the newest untested ones, so the backlog drains steadily.
    """
    if limit <= 0:
        return []
    return pick_generated([n for n, _ in _active_generated(ch)],
                          _tuned_validation(ch), limit)


def pick_generated(active: list[str], tuned: dict[str, float], limit: int) -> list[str]:
    """Pure selection: best proven (validation > 0) first, then newest untested."""
    if limit <= 0:
        return []
    proven = sorted((n for n in active if tuned.get(n, 0.0) > 0),
                    key=lambda n: -tuned[n])[: max(1, limit // 2)]
    fresh = [n for n in active if n not in tuned]
    return (proven + fresh)[:limit]


def archive_untested(ch, keep: int = 0) -> list[str]:
    """Move never-tuned generated strategies out of the registry (reversible).

    Files go to ``data/generated_archive`` and the row is marked ``archived``;
    moving a file back and setting status to ``active`` restores it. The newest
    ``keep`` untested strategies stay active.
    """
    from pathlib import Path

    from src.backtesting.strategies.registry import GENERATED_DIR, load_generated

    archived = untested_generated(ch)[keep:]
    if not archived:
        return []
    archive = Path(ARCHIVE_DIR)
    archive.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    cols = ["name", "code", "description", "status", "validation_sharpe", "model",
            "created_at", "updated_at"]
    rows = ch.query(
        f"SELECT {', '.join(cols)} FROM {database_name()}.generated_strategies FINAL "
        "WHERE name IN {names:Array(String)}", parameters={"names": archived}
    ).result_rows
    ch.insert(f"{database_name()}.generated_strategies",
              [[*r[:3], "archived", *r[4:7], now] for r in rows], column_names=cols)
    for name in archived:
        path = GENERATED_DIR / f"{name}.py"
        if path.exists():
            path.rename(archive / path.name)
    log_change(ch, "strategies_archived", subject="generated",
               detail={"count": len(archived), "reason": "never tuned"})
    load_generated()
    return archived
