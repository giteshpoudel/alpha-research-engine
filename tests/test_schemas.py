"""Integration tests for Phase 1 DDL.

Requires the Docker stack running: `docker compose up -d`.
Run from the repo root with the venv active: `python -m pytest tests/ -v`.
"""

import pytest

from src.ingestion.schemas import (
    create_clickhouse_schema,
    database_name,
    get_clickhouse_client,
)

EXPECTED_TABLES = ("ohlcv", "funding_rates", "sentiment_posts")


@pytest.fixture(scope="module")
def ch_client():
    client = get_clickhouse_client()
    create_clickhouse_schema(client)
    return client


def test_tables_exist(ch_client):
    rows = ch_client.query(f"SHOW TABLES FROM {database_name()}").result_rows
    names = {r[0] for r in rows}
    assert set(EXPECTED_TABLES) <= names


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_tables_use_replacing_merge_tree(ch_client, table):
    rows = ch_client.query(
        "SELECT engine FROM system.tables WHERE database = {db:String} AND name = {t:String}",
        parameters={"db": database_name(), "t": table},
    ).result_rows
    assert rows and "ReplacingMergeTree" in rows[0][0]


def test_ohlcv_columns(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.ohlcv").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["exchange"] == "LowCardinality(String)"
    assert cols["symbol"] == "String"
    assert cols["interval"] == "LowCardinality(String)"
    assert cols["ts"] == "DateTime64(3)"
    assert cols["open"] == "Float64"
    assert cols["high"] == "Float64"
    assert cols["low"] == "Float64"
    assert cols["close"] == "Float64"
    assert cols["volume"] == "Float64"
    assert cols["quote_volume"] == "Float64"
    assert cols["num_trades"] == "UInt32"


def test_sentiment_posts_columns(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.sentiment_posts").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["post_id"] == "String"
    assert cols["source"] == "LowCardinality(String)"
    assert cols["text"] == "String"
    assert cols["tickers"] == "Array(String)"
    assert cols["lang"] == "LowCardinality(String)"
    assert cols["likes"] == "UInt32"
    assert cols["retweets"] == "UInt32"
    assert cols["replies"] == "UInt32"
    assert cols["url"] == "String"
    assert cols["published_at"] == "DateTime64(3)"
    assert cols["ingested_at"] == "DateTime64(3)"
    assert cols["sentiment_score"] == "Nullable(Float32)"


def test_funding_rates_columns(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.funding_rates").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["exchange"] == "LowCardinality(String)"
    assert cols["symbol"] == "String"
    assert cols["ts"] == "DateTime64(3)"
    assert cols["funding_rate"] == "Float64"
    assert cols["mark_price"] == "Float64"
    assert cols["next_funding_ts"] == "DateTime64(3)"
