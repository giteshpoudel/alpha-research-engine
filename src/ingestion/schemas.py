"""Idempotent DDL for ClickHouse and Qdrant (Phase 1 storage infrastructure).

Usage: python -m src.ingestion.schemas
"""

from __future__ import annotations

import os

import clickhouse_connect
from clickhouse_connect.driver import Client as ClickHouseClient
from dotenv import load_dotenv


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
