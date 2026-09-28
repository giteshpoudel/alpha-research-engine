"""Append-only audit trail of optimizer-driven changes.

Unlike the state tables (ReplacingMergeTree), this records every event so the
human can see *when* a strategy was switched, params adopted, goal raised, or a
request filed — over time.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from src.ingestion.schemas import database_name

_COLUMNS = ("log_id", "event", "subject", "detail", "run_id", "created_at")


def log(ch, event: str, subject: str = "", detail: dict | None = None,
        run_id: str = "") -> str:
    log_id = uuid.uuid4().hex[:16]
    ch.insert(
        f"{database_name()}.change_log",
        [[log_id, event, subject, json.dumps(detail or {}, default=str), run_id,
          datetime.now(timezone.utc)]],
        column_names=list(_COLUMNS),
    )
    return log_id


def recent(ch, limit: int = 50) -> list[dict]:
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.change_log "
        "ORDER BY created_at DESC LIMIT {n:UInt32}",
        parameters={"n": limit},
    ).result_rows
    return [dict(zip(_COLUMNS, row)) for row in rows]
