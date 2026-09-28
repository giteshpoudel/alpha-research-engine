"""Shared test fixtures.

``change_log`` is append-only (MergeTree), so tests that exercise the optimizer
paths would otherwise leave audit rows behind. Use ``change_log_guard`` to clean
up everything written during a test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def change_log_guard():
    ch = get_clickhouse_client()
    marker = datetime.now(timezone.utc)
    yield
    ch.command(
        f"ALTER TABLE {database_name()}.change_log "
        "DELETE WHERE created_at >= {t:DateTime64(3)}",
        parameters={"t": marker},
        settings={"mutations_sync": 1},
    )
