"""Daily profit goals and progress for the optimizer.

The optimizer starts from a small positive target and raises it once met, so the
goal ratchets up over iterations. Progress is measured as the equal-weight
paper-portfolio return over the trailing 24h.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.agents.change_log import log as log_change
from src.ingestion.schemas import database_name

DEFAULT_DAILY_GOAL_PCT = 0.10  # percent per day
_GOAL_COLUMNS = ("goal_date", "target_profit_pct", "achieved_profit_pct",
                 "status", "updated_at")


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _write(ch, day: date, target: float, achieved: float, status: str) -> None:
    ch.insert(
        f"{database_name()}.agent_goals",
        [[day, float(target), float(achieved), status, datetime.now(timezone.utc)]],
        column_names=list(_GOAL_COLUMNS),
    )


def get_goal(ch, day: date) -> dict | None:
    rows = ch.query(
        f"SELECT {', '.join(_GOAL_COLUMNS)} FROM {database_name()}.agent_goals FINAL "
        "WHERE goal_date = {d:Date}",
        parameters={"d": day},
    ).result_rows
    return dict(zip(_GOAL_COLUMNS, rows[0])) if rows else None


def ensure_goal(ch, day: date, target: float = DEFAULT_DAILY_GOAL_PCT) -> dict:
    row = get_goal(ch, day)
    if row is None:
        _write(ch, day, target, 0.0, "pending")
        row = get_goal(ch, day)
    return row


def update_goal(ch, day: date, achieved_pct: float, status: str | None = None) -> dict:
    row = ensure_goal(ch, day)
    status = status or ("met" if achieved_pct >= float(row["target_profit_pct"]) else "pending")
    _write(ch, day, float(row["target_profit_pct"]), achieved_pct, status)
    if status == "met" and row["status"] != "met":
        log_change(ch, "goal_met", subject=str(day),
                   detail={"target": float(row["target_profit_pct"]),
                           "achieved": achieved_pct})
    return get_goal(ch, day)


def raise_goal(ch, day: date, factor: float = 1.1) -> dict:
    row = ensure_goal(ch, day)
    new_target = float(row["target_profit_pct"]) * factor
    _write(ch, day, new_target, float(row["achieved_profit_pct"]), "raised")
    log_change(ch, "goal_raised", subject=str(day),
               detail={"from": float(row["target_profit_pct"]), "to": new_target})
    return get_goal(ch, day)


def daily_profit_pct(ch) -> float | None:
    """Equal-weight paper-portfolio return over the trailing 24h, in percent."""
    rows = ch.query(
        f"SELECT ts, avg(equity) AS eq FROM {database_name()}.paper_equity FINAL "
        "WHERE NOT startsWith(strategy, 'TEST_') GROUP BY ts ORDER BY ts"
    ).result_rows
    if len(rows) < 2:
        return None
    last_ts, last_eq = rows[-1]
    target = _utc(last_ts) - timedelta(hours=24)
    past = [eq for ts, eq in rows if _utc(ts) <= target]
    if not past or past[-1] <= 0:
        return None
    return (float(last_eq) / float(past[-1]) - 1.0) * 100.0
