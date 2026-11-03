"""Daily profit goals and progress for the optimizer.

Targets follow docs/2026-10-01-goal-recalibration.md: the real goal is ~10%/yr net,
so the daily target starts at its daily equivalent (~0.026%/day) and can never ratchet
above 0.10%/day (~44%/yr). The old cap of 10%/day was unreachable and made the
optimizer chase noise.

Progress is the average trailing-24h return of each active paper series
(strategy, symbol), each measured against its own equity 24h earlier.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.agents.change_log import log as log_change
from src.ingestion.schemas import database_name

ANNUAL_TARGET_PCT = 10.0       # evidence-based target (recalibration note, section 4)
DEFAULT_DAILY_GOAL_PCT = round(((1 + ANNUAL_TARGET_PCT / 100) ** (1 / 365) - 1) * 100, 4)
MAX_DAILY_GOAL_PCT = 0.10      # ~44%/yr; raise_goal can't drift past this
_ACTIVE_WITHIN = timedelta(hours=9)  # stale series are skipped; carry marks every 8h
_LOOKBACK_DAYS = 7
_LOOKBACK_FRACTION = 0.5
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


def _rolling_target(ch, day: date) -> float:
    """New day's target from recent achieved returns (so it can't overshoot on one spike)."""
    rows = ch.query(
        f"SELECT achieved_profit_pct FROM {database_name()}.agent_goals FINAL "
        "WHERE goal_date < {d:Date} ORDER BY goal_date DESC LIMIT {n:UInt32}",
        parameters={"d": day, "n": _LOOKBACK_DAYS},
    ).result_rows
    positive = [float(r[0]) for r in rows if float(r[0]) > 0]
    if not positive:
        return DEFAULT_DAILY_GOAL_PCT
    target = _LOOKBACK_FRACTION * sum(positive) / len(positive)
    return round(min(MAX_DAILY_GOAL_PCT, max(DEFAULT_DAILY_GOAL_PCT, target)), 4)


def annualized_pct(daily_pct: float) -> float:
    return ((1 + daily_pct / 100) ** 365 - 1) * 100


def ensure_goal(ch, day: date, target: float | None = None) -> dict:
    row = get_goal(ch, day)
    if row is None:
        _write(ch, day, target if target is not None else _rolling_target(ch, day),
               0.0, "pending")
        row = get_goal(ch, day)
    elif target is None and float(row["target_profit_pct"]) > MAX_DAILY_GOAL_PCT:
        # rows written under the old 10%/day cap
        _write(ch, day, MAX_DAILY_GOAL_PCT, float(row["achieved_profit_pct"]), "pending")
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
    """Ratchet the target up, at most once per day, capped at MAX_DAILY_GOAL_PCT."""
    row = ensure_goal(ch, day)
    already = ch.query(
        f"SELECT count() FROM {database_name()}.change_log "
        "WHERE event = 'goal_raised' AND subject = {s:String}",
        parameters={"s": str(day)},
    ).result_rows[0][0]
    if already:
        return row
    new_target = min(MAX_DAILY_GOAL_PCT, float(row["target_profit_pct"]) * factor)
    _write(ch, day, new_target, float(row["achieved_profit_pct"]), "raised")
    log_change(ch, "goal_raised", subject=str(day),
               detail={"from": float(row["target_profit_pct"]), "to": new_target})
    return get_goal(ch, day)


def daily_profit_pct(ch) -> float | None:
    """Mean trailing-24h return across active paper series, in percent.

    Each (strategy, symbol) is compared with its own equity 24h earlier. Averaging raw
    equity per timestamp instead (the old method) mixed different sets of series at
    different timestamps and produced fake +27% days.
    """
    db = database_name()
    latest = ch.query(
        f"SELECT strategy, symbol, max(ts), argMax(equity, ts) FROM {db}.paper_equity FINAL "
        "WHERE NOT startsWith(strategy, 'TEST_') GROUP BY strategy, symbol"
    ).result_rows
    if not latest:
        return None
    now = max(_utc(r[2]) for r in latest)
    cutoff = now - timedelta(hours=24)
    past = {(s, y): float(eq) for s, y, eq in ch.query(
        f"SELECT strategy, symbol, argMax(equity, ts) FROM {db}.paper_equity FINAL "
        "WHERE NOT startsWith(strategy, 'TEST_') AND ts <= {c:DateTime64(3)} "
        "GROUP BY strategy, symbol", parameters={"c": cutoff.replace(tzinfo=None)}
    ).result_rows}
    rets = [float(eq) / past[(s, y)] - 1.0 for s, y, ts, eq in latest
            if _utc(ts) >= now - _ACTIVE_WITHIN and past.get((s, y), 0) > 0]
    if not rets:
        return None
    return sum(rets) / len(rets) * 100.0
