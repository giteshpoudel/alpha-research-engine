import json

import pytest

from src.meta_learning.compare import run_allocator, run_variant
from src.ingestion.schemas import database_name, get_clickhouse_client

MR_PARAMS = {"window": 24, "z_entry": -2.0, "z_exit": 0.0}
FA_PARAMS = {"threshold": 0.0003}


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    # backtest_equity has no strategy column; clean up by run_id (equity first,
    # while the subquery can still resolve TEST run_ids).
    client.command(
        f"ALTER TABLE {db}.backtest_equity DELETE WHERE run_id IN "
        f"(SELECT run_id FROM {db}.backtest_runs WHERE strategy LIKE 'TEST%')",
        settings={"mutations_sync": 1},
    )
    client.command(
        f"ALTER TABLE {db}.backtest_runs DELETE WHERE strategy LIKE 'TEST%'",
        settings={"mutations_sync": 1},
    )
    client.command(
        f"ALTER TABLE {db}.tuned_params DELETE WHERE strategy LIKE 'TEST%'",
        settings={"mutations_sync": 1},
    )


def _stored_run_ids(ch_client):
    rows = ch_client.query(
        f"SELECT run_id FROM {database_name()}.backtest_runs FINAL WHERE strategy LIKE 'TEST%'"
    ).result_rows
    return {r[0] for r in rows}


def test_run_variant_stores_variant_and_distinct_ids(ch_client):
    rid_tuned = run_variant(ch_client, "TEST_mean_reversion", "BTC", "tuned", MR_PARAMS)
    rid_static = run_variant(ch_client, "TEST_mean_reversion", "BTC", "static", MR_PARAMS)
    assert rid_tuned != rid_static  # variant inside params_json -> different run_ids
    rows = ch_client.query(
        f"SELECT params_json, window FROM {database_name()}.backtest_runs FINAL "
        "WHERE strategy = 'TEST_mean_reversion'"
    ).result_rows
    variants = {json.loads(r[0])["variant"] for r in rows}
    assert variants == {"tuned", "static"}
    assert all(r[1] == "OOS" for r in rows)


def test_run_allocator_causal_composite(ch_client):
    # Seed tuned params the allocator reads (TEST_-prefixed strategy names)
    from datetime import datetime, timezone
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [["TEST_mean_reversion", "BTC", json.dumps(MR_PARAMS), 1.0, 1.0, 8,
          datetime.now(timezone.utc)],
         ["TEST_funding_arb", "BTC", json.dumps(FA_PARAMS), 1.0, 1.0, 8,
          datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
    rid = run_allocator(ch_client, "BTC")
    rows = ch_client.query(
        f"SELECT strategy, params_json FROM {database_name()}.backtest_runs FINAL "
        "WHERE run_id = {r:String}", parameters={"r": rid},
    ).result_rows
    assert rows[0][0] == "allocator"
    assert json.loads(rows[0][1])["variant"] == "allocator"
    eq = ch_client.query(
        f"SELECT count() FROM {database_name()}.backtest_equity FINAL "
        "WHERE run_id = {r:String}", parameters={"r": rid},
    ).result_rows
    assert eq[0][0] > 0
