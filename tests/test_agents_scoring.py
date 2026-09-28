from datetime import datetime, timedelta, timezone

import pytest

from src.agents import scoring
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.agent_run_scores "
              "DELETE WHERE run_id LIKE 'TESTSCORE%'", settings={"mutations_sync": 1})


def test_compute_score():
    assert scoring.compute_score(adopted=1, proposals=2, requests=1, status="done") == 3.3
    assert scoring.compute_score(adopted=0, proposals=0, requests=0, status="failed") == -1.0


def test_score_run_writes_row(ch):
    score = scoring.score_run(ch, "TESTSCORE1", adopted=1, proposals=1)
    assert score > 0
    rows = ch.query(
        f"SELECT score, flag, adopted FROM {database_name()}.agent_run_scores FINAL "
        "WHERE run_id = 'TESTSCORE1'"
    ).result_rows
    assert rows and rows[0][1] == "ok" and rows[0][2] == 1


def test_detect_regression(ch):
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(5):  # older, high scores
        rows.append([f"TESTSCORE_hi_{i}", "optimizer", 10.0, "ok", 5, 0, 0, 0, 0,
                     0.0, 1.0, "done", now - timedelta(minutes=20 - i)])
    for i in range(5):  # newer, collapsed scores
        rows.append([f"TESTSCORE_lo_{i}", "optimizer", 0.1, "ok", 0, 0, 0, 0, 0,
                     0.0, 1.0, "done", now - timedelta(minutes=5 - i)])
    ch.insert(f"{database_name()}.agent_run_scores", rows,
              column_names=["run_id", "kind", "score", "flag", "adopted", "proposals",
                            "requests", "tokens_in", "tokens_out", "cost_usd",
                            "duration_ms", "status", "created_at"])
    regression = scoring.detect_regression(ch, window=10, factor=0.5)
    assert regression is not None and regression["recent"] < regression["older"]
