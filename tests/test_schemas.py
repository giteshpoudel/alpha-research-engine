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
    assert cols["author"] == "String"
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


def test_sentiment_metrics_table(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.sentiment_metrics").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["bucket_size"] == "LowCardinality(String)"
    assert cols["bucket_start"] == "DateTime64(3)"
    assert cols["ticker"] == "String"
    assert cols["post_count"] == "UInt32"
    assert cols["mean_score"] == "Float32"
    assert cols["weighted_score"] == "Float32"
    assert cols["engagement_total"] == "UInt64"
    assert cols["velocity"] == "Float32"
    assert cols["engagement_ratio"] == "Float32"
    assert cols["computed_at"] == "DateTime64(3)"


def test_sentiment_metrics_engine(ch_client):
    rows = ch_client.query(
        "SELECT engine FROM system.tables WHERE database = {db:String} AND name = 'sentiment_metrics'",
        parameters={"db": database_name()},
    ).result_rows
    assert rows and "ReplacingMergeTree" in rows[0][0]


from types import SimpleNamespace
from unittest.mock import Mock
import uuid

from qdrant_client.models import Distance

from src.ingestion.schemas import (
    QDRANT_COLLECTION,
    QDRANT_VECTOR_SIZE,
    create_qdrant_schema,
    get_qdrant_client,
    post_id_to_uuid,
)


@pytest.fixture(scope="module")
def qd_client():
    client = get_qdrant_client()
    create_qdrant_schema(client)
    return client


def test_collection_exists(qd_client):
    assert qd_client.collection_exists(QDRANT_COLLECTION)


def test_collection_vector_config(qd_client):
    info = qd_client.get_collection(QDRANT_COLLECTION)
    assert info.config.params.vectors.size == QDRANT_VECTOR_SIZE
    assert info.config.params.vectors.distance == Distance.COSINE


def test_payload_indexes(qd_client):
    info = qd_client.get_collection(QDRANT_COLLECTION)
    assert {"post_id", "source", "tickers", "published_at"} <= set(info.payload_schema)


def test_mismatched_vector_size_raises():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=512, distance=Distance.COSINE)
            )
        )
    )
    with pytest.raises(RuntimeError, match="vector size"):
        create_qdrant_schema(client)


def test_mismatched_vector_distance_raises():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=QDRANT_VECTOR_SIZE, distance=Distance.EUCLID)
            )
        )
    )
    with pytest.raises(RuntimeError, match="distance"):
        create_qdrant_schema(client)


def test_post_id_to_uuid_is_deterministic():
    first = post_id_to_uuid("tweet:123")
    assert first == post_id_to_uuid("tweet:123")
    assert first != post_id_to_uuid("tweet:124")
    uuid.UUID(first)  # raises unless valid UUID string


from src.ingestion.schemas import main, wait_for_services


def test_wait_for_services_returns_when_up():
    wait_for_services(timeout=30.0)  # must not raise while the stack is up


def test_wait_for_services_times_out():
    with pytest.raises(RuntimeError, match="Timed out"):
        wait_for_services(timeout=0.0)


def test_main_is_idempotent():
    main()
    main()  # second run must not raise
