"""Idempotent DDL for ClickHouse and Qdrant (Phase 1 storage infrastructure).

Usage: python -m src.ingestion.schemas
"""

from __future__ import annotations

import os
import uuid

import clickhouse_connect
from clickhouse_connect.driver import Client as ClickHouseClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def database_name() -> str:
    return _env("CLICKHOUSE_DB", "alpha")


def get_clickhouse_client() -> ClickHouseClient:
    load_dotenv()
    return clickhouse_connect.get_client(
        host=_env("CLICKHOUSE_HOST", "localhost"),
        port=int(_env("CLICKHOUSE_PORT", "8123")),
        username=_env("CLICKHOUSE_USER", "alpha"),
        password=_env("CLICKHOUSE_PASSWORD", ""),
    )


# ReplacingMergeTree everywhere: re-running historical imports replaces rows
# instead of duplicating them (blueprint guardrail: idempotent pipelines).
_OHLCV_DDL = """
CREATE TABLE IF NOT EXISTS {db}.ohlcv
(
    exchange LowCardinality(String),
    symbol String,
    interval LowCardinality(String),
    ts DateTime64(3),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    quote_volume Float64,
    num_trades UInt32
)
ENGINE = ReplacingMergeTree
ORDER BY (exchange, symbol, interval, ts)
"""

_FUNDING_RATES_DDL = """
CREATE TABLE IF NOT EXISTS {db}.funding_rates
(
    exchange LowCardinality(String),
    symbol String,
    ts DateTime64(3),
    funding_rate Float64,
    mark_price Float64,
    next_funding_ts DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (exchange, symbol, ts)
"""

# published_at is the post's own publication time so backtests never see
# future social data. sentiment_score stays NULL until the Phase 2 worker fills it.
_SENTIMENT_POSTS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.sentiment_posts
(
    post_id String,
    source LowCardinality(String),
    author String,
    text String,
    tickers Array(String),
    lang LowCardinality(String),
    likes UInt32,
    retweets UInt32,
    replies UInt32,
    url String,
    published_at DateTime64(3),
    ingested_at DateTime64(3),
    sentiment_score Nullable(Float32)
)
ENGINE = ReplacingMergeTree
ORDER BY (source, post_id)
"""


def create_clickhouse_schema(client: ClickHouseClient) -> None:
    """Create the database and all Phase 1 tables. Safe to re-run."""
    db = database_name()
    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL):
        client.command(ddl.format(db=db))


QDRANT_COLLECTION = "social_posts"
QDRANT_VECTOR_SIZE = 768  # Ollama nomic-embed-text output dimension

# Fixed namespace so post_id_to_uuid is stable across runs and machines.
_POST_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "alpha-research-engine")

_PAYLOAD_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("post_id", PayloadSchemaType.KEYWORD),
    ("source", PayloadSchemaType.KEYWORD),
    ("tickers", PayloadSchemaType.KEYWORD),
    ("published_at", PayloadSchemaType.INTEGER),  # unix seconds; range-filterable
)


def post_id_to_uuid(post_id: str) -> str:
    """Map a source post ID to a deterministic Qdrant point ID.

    Deterministic IDs make upserts idempotent: re-ingesting the same post
    overwrites its point instead of creating a duplicate.
    """
    return str(uuid.uuid5(_POST_ID_NAMESPACE, post_id))


def get_qdrant_client() -> QdrantClient:
    load_dotenv()
    return QdrantClient(
        host=_env("QDRANT_HOST", "localhost"),
        port=int(_env("QDRANT_PORT", "6333")),
    )


def create_qdrant_schema(client: QdrantClient) -> None:
    """Create the social_posts collection and payload indexes. Safe to re-run.

    Raises RuntimeError if the collection exists with a different vector
    size — recreating it would silently drop every embedded post.
    """
    if client.collection_exists(QDRANT_COLLECTION):
        existing = client.get_collection(QDRANT_COLLECTION).config.params.vectors.size
        if existing != QDRANT_VECTOR_SIZE:
            raise RuntimeError(
                f"Collection '{QDRANT_COLLECTION}' exists with vector size {existing}, "
                f"expected {QDRANT_VECTOR_SIZE}. Refusing to recreate (would drop data)."
            )
    else:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
    # Re-applying an identical index is a no-op on the server, so this stays idempotent.
    for field_name, schema_type in _PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=schema_type,
        )
