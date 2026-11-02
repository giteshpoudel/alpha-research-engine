from datetime import datetime, timezone

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.meta_learning import performance


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    for table in ("tuned_params", "strategy_performance"):
        mode = "DELETE WHERE strategy LIKE 'TESTPERF%'" if table == "tuned_params" else \
            "DELETE WHERE strategy LIKE 'TESTPERF%'"
        c.command(f"ALTER TABLE {database_name()}.{table} {mode}",
                  settings={"mutations_sync": 1})


def test_refresh_marks_dead_combo(ch):
    now = datetime.now(timezone.utc)
    ch.insert(
        f"{database_name()}.tuned_params",
        [["TESTPERF_dead", "BTC", "{}", 0.0, -1.0, 8, now, now]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at", "valid_from"],
    )
    assert performance.refresh(ch, min_validation=0.0) >= 1
    assert ("TESTPERF_dead", "BTC") in performance.dead_combos(ch)


def test_pick_generated_mixes_proven_and_fresh():
    from src.meta_learning.performance import pick_generated

    active = ["new3", "new2", "good", "bad", "best", "new1"]  # newest first
    tuned = {"good": 0.4, "bad": -0.2, "best": 0.9}
    assert pick_generated(active, tuned, 4) == ["best", "good", "new3", "new2"]
    assert pick_generated(active, tuned, 1) == ["best"]
    assert pick_generated(active, tuned, 0) == []
    # never-tuned backlog drains newest-first; failed ones are not retried
    assert "bad" not in pick_generated(active, tuned, 10)
