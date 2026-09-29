import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.meta_learning import batch


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.tuned_params "
              "DELETE WHERE strategy LIKE 'TEST%'", settings={"mutations_sync": 1})


def test_run_batch_single_combo(ch):
    result = batch.run_batch(ch, combos=[("TEST_mean_reversion", "BTC")],
                             n_trials=3, workers=1, use_lock=False, prune=False)
    assert result["combos"] == 1
    assert result["workers"] == 1
    assert result["results"][0]["strategy"] == "TEST_mean_reversion"
    assert result["results"][0]["symbol"] == "BTC"
