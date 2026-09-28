"""Persisted circuit-breaker state.

Failures and cooldown windows are stored in ClickHouse so a provider (or tool)
that is rate-limited is skipped across process restarts, not just within one
run.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name

_COLUMNS = ("breaker", "failures", "opened_until", "updated_at")


class BreakerStore:
    """Read/write breaker state keyed by an arbitrary breaker name."""

    def __init__(self, ch):
        self.ch = ch

    def get(self, name: str) -> tuple[int, datetime | None]:
        rows = self.ch.query(
            f"SELECT failures, opened_until FROM {database_name()}.circuit_breakers FINAL "
            "WHERE breaker = {b:String}",
            parameters={"b": name},
        ).result_rows
        if not rows:
            return 0, None
        failures, opened_until = rows[0]
        if opened_until is not None and opened_until.tzinfo is None:
            opened_until = opened_until.replace(tzinfo=timezone.utc)
        return int(failures), opened_until

    def save(self, name: str, failures: int, opened_until: datetime | None) -> None:
        self.ch.insert(
            f"{database_name()}.circuit_breakers",
            [[name, int(failures), opened_until, datetime.now(timezone.utc)]],
            column_names=list(_COLUMNS),
        )
