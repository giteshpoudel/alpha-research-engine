import pytest

from src.research import collectors
from src.ingestion.schemas import get_clickhouse_client


@pytest.fixture(scope="module")
def ch_client():
    return get_clickhouse_client()


def test_spike_flags():
    assert collectors.spike_flags(2.5, 1.0) == ["mention_spike"]
    assert collectors.spike_flags(1.0, 3.0) == ["engagement_spike"]
    assert collectors.spike_flags(2.5, 3.0) == ["mention_spike", "engagement_spike"]
    assert collectors.spike_flags(1.0, 1.0) == []


def test_collect_risk_data(ch_client):
    data = collectors.collect_risk_data(ch_client)
    assert set(data) == {"top_correlated", "top_volatile", "signal_states", "tuned_drawdowns"}
    for item in data["top_correlated"]:
        assert -1.0 <= item["corr"] <= 1.0
        assert " vs " in item["pair"]
    assert len(data["top_correlated"]) <= 5
    assert len(data["top_volatile"]) <= 5
    assert all(v["ann_vol"] > 0 for v in data["top_volatile"])
    states = {s["state"] for s in data["signal_states"]}
    assert states <= {"in", "out"}
    assert len(data["signal_states"]) == 12
    for item in data["tuned_drawdowns"]:
        assert set(item) == {"symbol", "max_dd"}


def test_params_for_symbol_falls_back_and_uses_tuned(ch_client):
    import json
    from datetime import datetime, timezone

    from src.ingestion.schemas import database_name

    assert collectors._params_for_symbol(ch_client, "NOPE") == collectors.MR_DEFAULTS
    params = {"window": 12, "z_entry": -1.5, "z_exit": 0.25}
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [["TEST_mean_reversion", "BTC", json.dumps(params), 1.0, 1.0, 8,
          datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
    try:
        assert collectors._params_for_symbol(
            ch_client, "BTC", strategy="TEST_mean_reversion") == params
    finally:
        ch_client.command(
            f"ALTER TABLE {database_name()}.tuned_params DELETE WHERE strategy LIKE 'TEST%'",
            settings={"mutations_sync": 1},
        )


def test_z_state_honors_params():
    import pandas as pd

    close = pd.Series([10.0, 10.0, 10.0, 1.0])
    state, z = collectors._z_state(close, {"window": 2, "z_entry": -1.0, "z_exit": 0.0})
    assert state == "in"
    assert z <= -1.0


def test_collect_macro_data(ch_client):
    data = collectors.collect_macro_data(ch_client)
    assert set(data) == {"sentiment", "price_changes_24h", "funding_extremes", "top_posts"}
    assert len(data["funding_extremes"]) <= 3
    assert len(data["top_posts"]) <= 3
    for item in data["sentiment"]:
        assert isinstance(item["flags"], list)
    if data["top_posts"]:
        post = data["top_posts"][0]
        assert post["likes"] >= data["top_posts"][-1]["likes"]  # sorted desc
        assert len(post["text_excerpt"]) <= 200
