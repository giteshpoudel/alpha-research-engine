# Phase 1 Storage Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spin up ClickHouse + Qdrant via Docker Compose and provide an idempotent Python DDL module that creates the market time-series and sentiment schemas from the Phase 1 spec.

**Architecture:** Two containerized databases (ClickHouse for columnar time-series, Qdrant for 768-dim vectors) defined in a root `docker-compose.yml`. A single Python module `src/ingestion/schemas.py` applies all DDL idempotently (ClickHouse `ReplacingMergeTree` tables + Qdrant collection with payload indexes) and doubles as a CLI (`python -m src.ingestion.schemas`).

**Tech Stack:** Docker Compose, ClickHouse 25.8, Qdrant v1.19.0, Python 3.11+, `clickhouse-connect`, `qdrant-client`, `python-dotenv`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-storage-infra-design.md`

## Global Constraints

- Python 3.11+; no existing `requirements.txt`, venv, or test framework — this plan introduces them.
- ClickHouse image: `clickhouse/clickhouse-server:25.8`. Qdrant image: `qdrant/qdrant:v1.19.0`. Do not substitute `latest`.
- Qdrant vectors: **768 dimensions, cosine distance** (Ollama `nomic-embed-text`).
- All ClickHouse tables use `ReplacingMergeTree`; all DDL must be idempotent (`IF NOT EXISTS` / safe re-runs). Re-running any script must never duplicate or drop data.
- If the Qdrant collection exists with a mismatched vector size, **raise an error** — never silently recreate it.
- Env config comes from `.env` (loaded with `python-dotenv`); defaults match `.env.template`.
- Ollama and Redpanda are **not** part of the compose file.
- Repo layout: `src/` packages at repo root, tests in `tests/`, run everything from repo root.

---

### Task 1: Docker Compose + environment wiring

**Files:**
- Create: `docker-compose.yml`
- Modify: `.env.template`
- Modify: `.env` (local, gitignored — merge new keys, don't overwrite existing values)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: running `clickhouse` (ports 8123/9000) and `qdrant` (ports 6333/6334) services with a pre-created `alpha` database; env keys `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_NATIVE_PORT`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DB`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT`, `OLLAMA_HOST` that Task 2's Python clients read.

- [ ] **Step 1: Update `.env.template` and local `.env`**

Append to `.env.template` (keeping the existing `CLICKHOUSE_HOST`/`QDRANT_HOST` lines; do not duplicate them):

```
CLICKHOUSE_PORT="8123"
CLICKHOUSE_NATIVE_PORT="9000"
CLICKHOUSE_USER="alpha"
CLICKHOUSE_PASSWORD=""
CLICKHOUSE_DB="alpha"
QDRANT_PORT="6333"
QDRANT_GRPC_PORT="6334"
OLLAMA_HOST="localhost"
```

Copy the same lines into the local `.env` (it exists and is gitignored). The ClickHouse container entrypoint accepts a custom user with an empty password (verified against `docker/server/entrypoint.sh`: it writes an empty `<password>` element and keeps network access enabled), so empty `CLICKHOUSE_PASSWORD` is fine for local dev.

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:25.8
    container_name: alpha-clickhouse
    ports:
      - "${CLICKHOUSE_PORT:-8123}:8123"
      - "${CLICKHOUSE_NATIVE_PORT:-9000}:9000"
    environment:
      CLICKHOUSE_DB: ${CLICKHOUSE_DB:-alpha}
      CLICKHOUSE_USER: ${CLICKHOUSE_USER:-alpha}
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD:-}
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://127.0.0.1:8123/ping"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s

  qdrant:
    image: qdrant/qdrant:v1.19.0
    container_name: alpha-qdrant
    ports:
      - "${QDRANT_PORT:-6333}:6333"
      - "${QDRANT_GRPC_PORT:-6334}:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      # qdrant image ships no curl/wget; use bash's /dev/tcp probe
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 5s

volumes:
  clickhouse_data:
  qdrant_data:
```

- [ ] **Step 3: Bring the stack up and verify health**

Run: `docker compose up -d && docker compose ps`
Expected: both `alpha-clickhouse` and `alpha-qdrant` show `(healthy)` within ~30s.

Then verify endpoints directly:

```bash
curl -s http://localhost:8123/ping
# Expected: Ok.
curl -s http://localhost:6333/healthz
# Expected: healthz check passed
curl -s "http://localhost:8123/?query=SHOW DATABASES"
# Expected output includes: alpha
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.template
git commit -m "feat: add docker-compose for ClickHouse and Qdrant"
```

---

### Task 2: Python scaffolding + ClickHouse schema (TDD)

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py` (repo root — makes pytest put repo root on `sys.path`)
- Create: `src/__init__.py` (empty)
- Create: `src/ingestion/__init__.py` (empty)
- Create: `src/ingestion/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: running containers and env keys from Task 1.
- Produces (used by Tasks 3–4 and later phases):
  - `get_clickhouse_client() -> clickhouse_connect.driver.Client`
  - `database_name() -> str`
  - `create_clickhouse_schema(client) -> None` — idempotent

- [ ] **Step 1: Create venv, `requirements.txt`, and package scaffolding**

`requirements.txt`:

```
clickhouse-connect>=0.8
qdrant-client>=1.12
python-dotenv>=1.0
pytest>=8.0
```

Run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
touch src/__init__.py src/ingestion/__init__.py conftest.py
```

Expected: install completes; `.venv/` is already in `.gitignore`.

- [ ] **Step 2: Write the failing ClickHouse tests**

`tests/test_schemas.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.schemas'`.

- [ ] **Step 4: Implement `src/ingestion/schemas.py` (ClickHouse half)**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: all 7 tests PASS (3 parametrized engine checks + 4 single tests).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt conftest.py src/__init__.py src/ingestion/__init__.py src/ingestion/schemas.py tests/test_schemas.py
git commit -m "feat: add ClickHouse schema module (ohlcv, funding_rates, sentiment_posts)"
```

---

### Task 3: Qdrant collection schema (TDD)

**Files:**
- Modify: `src/ingestion/schemas.py` (append Qdrant section)
- Modify: `tests/test_schemas.py` (append Qdrant tests)

**Interfaces:**
- Consumes: `_env` from Task 2's `schemas.py`; running Qdrant from Task 1.
- Produces (used by Task 4 and later phases):
  - `QDRANT_COLLECTION: str` — `"social_posts"`
  - `QDRANT_VECTOR_SIZE: int` — `768`
  - `post_id_to_uuid(post_id: str) -> str` — deterministic UUID5 for idempotent upserts
  - `get_qdrant_client() -> qdrant_client.QdrantClient`
  - `create_qdrant_schema(client) -> None` — idempotent; raises `RuntimeError` if the collection exists with a wrong vector size

- [ ] **Step 1: Write the failing Qdrant tests**

Append to `tests/test_schemas.py`:

```python
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
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=512)))
    )
    with pytest.raises(RuntimeError, match="vector size"):
        create_qdrant_schema(client)


def test_post_id_to_uuid_is_deterministic():
    first = post_id_to_uuid("tweet:123")
    assert first == post_id_to_uuid("tweet:123")
    assert first != post_id_to_uuid("tweet:124")
    uuid.UUID(first)  # raises unless valid UUID string
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: the whole file fails at collection with `ImportError: cannot import name 'QDRANT_COLLECTION'` (the module-level import aborts collection before any test runs).

- [ ] **Step 3: Implement the Qdrant half of `src/ingestion/schemas.py`**

Update the imports at the top of `schemas.py`:

```python
import os
import uuid

import clickhouse_connect
from clickhouse_connect.driver import Client as ClickHouseClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
```

Append to `schemas.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/schemas.py tests/test_schemas.py
git commit -m "feat: add Qdrant social_posts collection schema (768-dim cosine)"
```

---

### Task 4: CLI entry point with service readiness wait (TDD)

**Files:**
- Modify: `src/ingestion/schemas.py` (append CLI section, add `import time`)
- Modify: `tests/test_schemas.py` (append CLI test)

**Interfaces:**
- Consumes: `get_clickhouse_client`, `create_clickhouse_schema`, `get_qdrant_client`, `create_qdrant_schema` from Tasks 2–3.
- Produces: `wait_for_services(timeout: float = 60.0) -> None` (raises `RuntimeError` on timeout) and `main() -> None`; runnable as `python -m src.ingestion.schemas`.

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_schemas.py`:

```python
from src.ingestion.schemas import main, wait_for_services


def test_wait_for_services_returns_when_up():
    wait_for_services(timeout=30.0)  # must not raise while the stack is up


def test_wait_for_services_times_out():
    with pytest.raises(RuntimeError, match="Timed out"):
        wait_for_services(timeout=0.0)


def test_main_is_idempotent():
    main()
    main()  # second run must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: the 3 new tests FAIL with `ImportError: cannot import name 'main'`.

Note on `test_wait_for_services_times_out`: with `timeout=0.0` the deadline is already in the past, so the function must check the deadline first and readiness second — the implementation below does exactly that, and this test passes immediately regardless of container state. (Checking readiness first would return without raising whenever the stack is up — and the stack must be up for the other tests.)

- [ ] **Step 3: Implement the CLI section of `src/ingestion/schemas.py`**

Add `import time` to the imports, then append:

```python
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
    print(f"ClickHouse: database '{database_name()}' ready (ohlcv, funding_rates, sentiment_posts)")
    print(f"Qdrant: collection '{QDRANT_COLLECTION}' ready ({QDRANT_VECTOR_SIZE}-dim cosine)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: all 15 tests PASS.

- [ ] **Step 5: Verify the CLI end-to-end from the shell**

Run (from repo root, venv active):

```bash
python -m src.ingestion.schemas
python -m src.ingestion.schemas   # run twice: must succeed both times
```

Expected output (both runs):

```
ClickHouse: database 'alpha' ready (ohlcv, funding_rates, sentiment_posts)
Qdrant: collection 'social_posts' ready (768-dim cosine)
```

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/schemas.py tests/test_schemas.py
git commit -m "feat: add schema CLI with service readiness wait"
```

---

## Self-Review Notes

- **Spec coverage:** compose file (Task 1), `.env.template` additions (Task 1), all three ClickHouse tables with exact columns/engines (Task 2), Qdrant collection + indexes + uuid5 point IDs + mismatch guard (Task 3), CLI with wait/backoff and idempotent re-runs (Task 4), pytest smoke tests (Tasks 2–4), `requirements.txt` (Task 2). All spec sections covered.
- **Type consistency:** `create_clickhouse_schema(client)` / `create_qdrant_schema(client)` / `database_name()` / `QDRANT_COLLECTION` / `QDRANT_VECTOR_SIZE` / `post_id_to_uuid(str) -> str` / `wait_for_services(timeout: float)` / `main()` are spelled identically in the interfaces blocks, tests, and implementations.
- **Deviations from spec (implementation details):** Qdrant healthcheck uses bash `/dev/tcp` instead of an HTTP `/healthz` probe because the image ships no curl/wget; `.env.template` keeps an empty `CLICKHOUSE_PASSWORD`, which the ClickHouse entrypoint explicitly supports for local dev.
