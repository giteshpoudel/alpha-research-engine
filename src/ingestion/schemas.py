"""Idempotent DDL for ClickHouse and Qdrant (Phase 1 storage infrastructure).

Usage: python -m src.ingestion.schemas
"""

from __future__ import annotations

import os
import time
import uuid

import clickhouse_connect
from clickhouse_connect.driver import Client as ClickHouseClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def database_name() -> str:
    load_dotenv()
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
    sentiment_score Nullable(Float32),
    label Nullable(String)
)
ENGINE = ReplacingMergeTree
ORDER BY (source, post_id)
"""

# Bucket metrics are recomputed with replace semantics: re-running the
# aggregator for the same (bucket_size, bucket_start, ticker) replaces the row.
_SENTIMENT_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.sentiment_metrics
(
    bucket_size LowCardinality(String),
    bucket_start DateTime64(3),
    ticker String,
    post_count UInt32,
    mean_score Float32,
    weighted_score Float32,
    engagement_total UInt64,
    velocity Float32,
    engagement_ratio Float32,
    computed_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (bucket_size, bucket_start, ticker)
"""

# run_id is a deterministic hash of strategy+symbol+params+window+dates, so
# re-running a backtest collapses to one row (idempotent, no version column).
_BACKTEST_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.backtest_runs
(
    run_id String,
    strategy LowCardinality(String),
    symbol String,
    interval LowCardinality(String),
    params_json String,
    window LowCardinality(String),
    start_ts DateTime64(3),
    end_ts DateTime64(3),
    total_return Float32,
    sharpe Float32,
    sortino Float32,
    max_drawdown Float32,
    calmar Float32,
    win_rate Float32,
    num_trades UInt32,
    created_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol, window, run_id)
"""

_BACKTEST_EQUITY_DDL = """
CREATE TABLE IF NOT EXISTS {db}.backtest_equity
(
    run_id String,
    ts DateTime64(3),
    equity Float64
)
ENGINE = ReplacingMergeTree
ORDER BY (run_id, ts)
"""


def create_clickhouse_schema(client: ClickHouseClient) -> None:
    """Create the database and all pipeline tables. Safe to re-run."""
    db = database_name()
    client.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL, _SENTIMENT_METRICS_DDL,
                _BACKTEST_RUNS_DDL, _BACKTEST_EQUITY_DDL):
        client.command(ddl.format(db=db))
    # Idempotent column migration for databases created before Phase 2.1.
    client.command(
        f"ALTER TABLE {db}.sentiment_posts ADD COLUMN IF NOT EXISTS label Nullable(String)"
    )


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
    size or distance — recreating it would silently drop every embedded post.
    """
    if client.collection_exists(QDRANT_COLLECTION):
        vectors = client.get_collection(QDRANT_COLLECTION).config.params.vectors
        if vectors.size != QDRANT_VECTOR_SIZE:
            raise RuntimeError(
                f"Collection '{QDRANT_COLLECTION}' exists with vector size {vectors.size}, "
                f"expected {QDRANT_VECTOR_SIZE}. Refusing to recreate (would drop data)."
            )
        if vectors.distance != Distance.COSINE:
            raise RuntimeError(
                f"Collection '{QDRANT_COLLECTION}' exists with distance {vectors.distance}, "
                f"expected {Distance.COSINE}. Refusing to recreate (would drop data)."
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


def _clickhouse_ready() -> bool:
    try:
        get_clickhouse_client().command("SELECT 1")
        return True
    except Exception:
        return False


def _qdrant_ready() -> bool:
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:
        return False


def wait_for_services(timeout: float = 60.0) -> None:
    """Block until ClickHouse and Qdrant both answer, with exponential backoff."""
    deadline = time.monotonic() + timeout
    delay = 1.0
    while True:
        # Deadline first: timeout=0.0 must raise even when the stack is up.
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for ClickHouse/Qdrant. "
                "Is the stack running? Try: docker compose up -d"
            )
        if _clickhouse_ready() and _qdrant_ready():
            return
        time.sleep(delay)
        delay = min(delay * 2, 10.0)


def main() -> None:
    wait_for_services()
    create_clickhouse_schema(get_clickhouse_client())
    create_qdrant_schema(get_qdrant_client())
    print(f"ClickHouse: database '{database_name()}' ready (ohlcv, funding_rates, sentiment_posts, sentiment_metrics)")
    print(f"Qdrant: collection '{QDRANT_COLLECTION}' ready ({QDRANT_VECTOR_SIZE}-dim cosine)")


if __name__ == "__main__":
    main()
