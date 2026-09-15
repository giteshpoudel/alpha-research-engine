import json
from datetime import datetime, timezone

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.meta_learning.params import get_tuned_params


@pytest.fixture
def ch_client():
    c = get_clickhouse_client()
    yield c
    c.command(
        f"ALTER TABLE {database_name()}.tuned_params DELETE WHERE strategy LIKE 'TEST%'",
        settings={"mutations_sync": 1},
    )


def test_get_tuned_params_returns_none_when_missing(ch_client):
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "NOPE") is None


def test_get_tuned_params_parses_seeded_row(ch_client):
    params = {"window": 30, "z_entry": -2.5, "z_exit": 0.1}
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [["TEST_mean_reversion", "BTC", json.dumps(params), 1.0, 1.0, 8,
          datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "BTC") == params
