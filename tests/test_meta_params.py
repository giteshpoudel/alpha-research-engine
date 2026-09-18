import json
from datetime import datetime, timezone

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.meta_learning.params import get_tuned_param_history, get_tuned_params


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


def _seed(ch_client, symbol, params, valid_from):
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [["TEST_mean_reversion", symbol, json.dumps(params), 1.0, 1.0, 8,
          datetime.now(timezone.utc), valid_from]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at", "valid_from"],
    )


def test_get_tuned_params_parses_seeded_row(ch_client):
    params = {"window": 30, "z_entry": -2.5, "z_exit": 0.1}
    _seed(ch_client, "BTC", params, datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "BTC") == params


def test_get_tuned_params_as_of_and_history(ch_client):
    v1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    v2 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    p1 = {"window": 20, "z_entry": -2.0, "z_exit": 0.0}
    p2 = {"window": 40, "z_entry": -2.5, "z_exit": 0.1}
    _seed(ch_client, "ETH", p1, v1)
    _seed(ch_client, "ETH", p2, v2)
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "ETH",
                            as_of=datetime(2025, 6, 1, tzinfo=timezone.utc)) == p1
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "ETH",
                            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc)) == p2
    assert get_tuned_params(ch_client, "TEST_mean_reversion", "ETH") == p2  # latest
    assert [p for _, p in get_tuned_param_history(ch_client, "TEST_mean_reversion", "ETH")] == [p1, p2]
