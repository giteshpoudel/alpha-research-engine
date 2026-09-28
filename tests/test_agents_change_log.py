import pytest

from src.agents import change_log
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.change_log DELETE WHERE subject = 'TESTLOG'",
              settings={"mutations_sync": 1})


def test_log_appends_and_recent(ch):
    change_log.log(ch, "test_event", subject="TESTLOG", detail={"x": 1}, run_id="r1")
    change_log.log(ch, "test_event", subject="TESTLOG", detail={"x": 2}, run_id="r1")
    rows = [r for r in change_log.recent(ch, 100) if r["subject"] == "TESTLOG"]
    assert len(rows) == 2          # append-only (no dedup)
    assert rows[0]["event"] == "test_event"
