# StockTwits Poller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a StockTwits symbol-stream poller to the sentiment pipeline (social source replacing blocked Reddit access), persisting Bullish/Bearish user labels in a new `sentiment_posts.label` column for later scorer validation.

**Architecture:** Mirrors the existing `reddit.py`/`rss.py` pollers: `fetch_stream` → `map_message` → `insert_posts`, per-symbol failure isolation, structural dedup via deterministic `post_id`. A new pipeline stage slots between rss and score; scorer and aggregator are untouched and pick up StockTwits rows automatically.

**Tech Stack:** Python 3.11, `httpx`, `clickhouse-connect`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-stocktwits-poller-design.md`

## Global Constraints

- All stages idempotent; dedup structural (ReplacingMergeTree + deterministic `post_id` = `stocktwits:{message_id}`).
- `published_at` is always the message's own `created_at` timestamp (blueprint guardrail #2).
- All HTTP via `get_with_backoff` (User-Agent, timeout, retry 429/5xx/transport 1s→2s→4s cap 10s, max 4 attempts).
- Tickers come from the API's structured `symbols` array (strip `.X` suffix, keep only symbols in `TICKER_ALIASES`) — NOT from `extract_tickers` text regex.
- `label` is persisted for scorer validation only; it is never a scoring input.
- 12 symbols × 1 request per cycle (~144/hour at 300s interval) stays under StockTwits' ~200/hour unauthenticated ceiling.
- No new dependencies.
- Test hygiene: integration tests seed rows with `post_id` starting with `test:` and delete them in teardown; never truncate or touch real rows.
- The existing 49 tests must stay green. Reddit poller code stays unchanged.
- Run tests from the repo root with the venv active: `python -m pytest tests/ -v`.

---

### Task 1: `label` column on `sentiment_posts`

**Files:**
- Modify: `src/ingestion/schemas.py` (DDL template + `create_clickhouse_schema`)
- Modify: `src/ingestion/posts.py` (column tuple + insert row)
- Test: `tests/test_schemas.py` (append one test)

**Interfaces:**
- Consumes: existing `create_clickhouse_schema(client)`, `database_name()` in `schemas.py`; `insert_posts` in `posts.py`.
- Produces: `sentiment_posts.label Nullable(String)`; `posts.SENTIMENT_POST_COLUMNS` gains `"label"`; `insert_posts` accepts an optional `"label"` key in row dicts (default `None`). Task 2 fills it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas.py`:

```python
def test_sentiment_posts_has_label_column(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.sentiment_posts").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["label"] == "Nullable(String)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py -v -k label`
Expected: FAIL — `KeyError: 'label'`.

- [ ] **Step 3: Add the column to `src/ingestion/schemas.py`**

In `_SENTIMENT_POSTS_DDL`, add the column after `sentiment_score Nullable(Float32)`:

```python
    sentiment_score Nullable(Float32),
    label Nullable(String)
```

In `create_clickhouse_schema`, after the DDL loop, add the idempotent migration for existing databases (with its comment):

```python
    # Idempotent column migration for databases created before Phase 2.1.
    client.command(
        f"ALTER TABLE {db}.sentiment_posts ADD COLUMN IF NOT EXISTS label Nullable(String)"
    )
```

- [ ] **Step 4: Update `src/ingestion/posts.py`**

Add `"label"` to `SENTIMENT_POST_COLUMNS`:

```python
SENTIMENT_POST_COLUMNS = (
    "post_id", "source", "author", "text", "tickers", "lang",
    "likes", "retweets", "replies", "url",
    "published_at", "ingested_at", "sentiment_score", "label",
)
```

In `insert_posts`, append the label value to the data row (after `None` for sentiment_score):

```python
            row["published_at"] or now,  # documented fallback: dateless RSS items
            now,
            None,
            row.get("label"),
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all 50 tests PASS (49 existing + 1 new). Existing reddit/rss inserts (no `"label"` key) still work via the `row.get("label")` default.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/schemas.py src/ingestion/posts.py tests/test_schemas.py
git commit -m "feat: add label column to sentiment_posts for source-provided sentiment"
```

---

### Task 2: StockTwits poller + pipeline stage

**Files:**
- Create: `src/ingestion/stocktwits.py`
- Modify: `src/ingestion/pipeline.py` (import + one stage)
- Test: `tests/test_stocktwits.py` (new), `tests/test_pipeline.py` (update two tests)

**Interfaces:**
- Consumes: `http.get_with_backoff`, `posts.insert_posts`, `tickers.TICKER_ALIASES` from earlier phases.
- Produces:
  - `stocktwits.SYMBOLS: tuple[str, ...]` — `("BTC.X", "ETH.X", …)` (12, one per `TICKER_ALIASES` key)
  - `stocktwits.map_message(msg: dict) -> dict | None`
  - `stocktwits.fetch_stream(http_client, symbol: str) -> list[dict]`
  - `stocktwits.poll_symbols(ch_client, http_client=None, symbols=SYMBOLS) -> int`
  - Pipeline cycle order becomes: reddit → rss → stocktwits → score → aggregate.

- [ ] **Step 1: Write the failing tests**

`tests/test_stocktwits.py`:

```python
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.stocktwits import fetch_stream, map_message, poll_symbols

MESSAGE_FIXTURE = {
    "id": 663469073,
    "body": "$BTC looking strong, breakout incoming",
    "created_at": "2026-09-02T18:58:51Z",
    "user": {"id": 9565930, "username": "Agonas"},
    "symbols": [{"id": 11418, "symbol": "BTC.X"}],
    "entities": {"media": [], "sentiment": {"basic": "Bullish"}},
    "likes": {"total": 7},
}

UNLABELED_FIXTURE = {
    "id": 663469074,
    "body": "BTC sideways all day, eth too",
    "created_at": "2026-09-02T19:01:00Z",
    "user": {"id": 42, "username": "watcher"},
    "symbols": [{"id": 11418, "symbol": "BTC.X"}, {"id": 11419, "symbol": "ETH.X"}],
    "entities": {"media": [], "sentiment": None},
}

EMPTY_FIXTURE = {
    "id": 663469075,
    "body": "   ",
    "created_at": "2026-09-02T19:02:00Z",
    "user": {"id": 1, "username": "nobody"},
    "symbols": [],
    "entities": {"sentiment": None},
}

STREAM_FIXTURE = {"messages": [MESSAGE_FIXTURE, UNLABELED_FIXTURE, EMPTY_FIXTURE]}


def test_map_message_fields():
    row = map_message(MESSAGE_FIXTURE)
    assert row["post_id"] == "stocktwits:663469073"
    assert row["source"] == "stocktwits"
    assert row["author"] == "Agonas"
    assert row["text"] == "$BTC looking strong, breakout incoming"
    assert row["tickers"] == ["BTC"]
    assert row["likes"] == 7
    assert row["retweets"] == 0
    assert row["replies"] == 0
    assert row["url"] == "https://stocktwits.com/Agonas/message/663469073"
    assert row["published_at"] == datetime(2026, 9, 2, 18, 58, 51, tzinfo=timezone.utc)
    assert row["label"] == "Bullish"


def test_map_message_unlabeled_and_multi_ticker():
    row = map_message(UNLABELED_FIXTURE)
    assert row["label"] is None
    assert row["tickers"] == ["BTC", "ETH"]


def test_map_message_skips_empty():
    assert map_message(EMPTY_FIXTURE) is None


def test_fetch_stream_filters():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=STREAM_FIXTURE)))
    rows = fetch_stream(client, "BTC.X")
    assert [r["post_id"] for r in rows] == ["stocktwits:663469073", "stocktwits:663469074"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')"
    )


def test_poll_symbols_inserts_idempotently_and_stores_label(ch_client):
    http_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=STREAM_FIXTURE)))
    import src.ingestion.stocktwits as st_mod
    original = st_mod.map_message
    st_mod.map_message = lambda msg: (
        None if original(msg) is None
        else {**original(msg), "post_id": "test:" + original(msg)["post_id"]}
    )
    try:
        assert poll_symbols(ch_client, http_client=http_client, symbols=("BTC.X",)) == 2
        assert poll_symbols(ch_client, http_client=http_client, symbols=("BTC.X",)) == 2
        rows = ch_client.query(
            f"SELECT post_id, label FROM {database_name()}.sentiment_posts FINAL "
            "WHERE startsWith(post_id, 'test:stocktwits:') ORDER BY post_id"
        ).result_rows
        assert rows == [
            ("test:stocktwits:663469073", "Bullish"),
            ("test:stocktwits:663469074", None),
        ]
    finally:
        st_mod.map_message = original
```

Update the two affected tests in `tests/test_pipeline.py`. In the `staged` fixture, add the stocktwits patch between rss and score:

```python
    monkeypatch.setattr(pipeline, "poll_symbols", lambda ch: calls.append("stocktwits") or 4)
```

Update `test_run_cycle_order_and_stats`:

```python
def test_run_cycle_order_and_stats(staged):
    stats = pipeline.run_cycle(ch_client=object(), qd_client=object())
    assert staged == ["reddit", "rss", "stocktwits", "score", "aggregate"]
    assert stats == {"reddit": 3, "rss": 2, "stocktwits": 4, "score": 5, "aggregate": 10}
```

Update `test_run_cycle_stage_failure_is_isolated`'s final assertion:

```python
    assert staged == ["rss", "stocktwits", "score", "aggregate"]  # later stages still ran
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stocktwits.py tests/test_pipeline.py -v`
Expected: `test_stocktwits.py` FAILs at collection — `ModuleNotFoundError: No module named 'src.ingestion.stocktwits'`; `test_pipeline.py` FAILs — `AttributeError: ... module 'src.ingestion.pipeline' has no attribute 'poll_symbols'`.

- [ ] **Step 3: Implement `src/ingestion/stocktwits.py`**

```python
"""StockTwits poller: symbol streams -> sentiment_posts (source='stocktwits').

Unauthenticated public JSON API. Tickers come from the API's structured
symbols array (more accurate than text regex for this source). User-supplied
Bullish/Bearish labels are persisted in the `label` column for scorer
validation only — never used as a scoring input.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.tickers import TICKER_ALIASES

SYMBOLS: tuple[str, ...] = tuple(f"{t}.X" for t in TICKER_ALIASES)


def _map_symbol(symbol: str) -> str | None:
    ticker = symbol.removesuffix(".X")
    return ticker if ticker in TICKER_ALIASES else None


def map_message(msg: dict) -> dict | None:
    """Map a StockTwits stream message to a row dict, or None to skip."""
    text = (msg.get("body") or "").strip()
    if not text:
        return None
    author = (msg.get("user") or {}).get("username") or ""
    tickers = sorted({
        ticker
        for ticker in (_map_symbol((s or {}).get("symbol", "")) for s in msg.get("symbols") or [])
        if ticker
    })
    sentiment = (msg.get("entities") or {}).get("sentiment") or {}
    likes = (msg.get("likes") or {}).get("total") or 0
    return {
        "post_id": f"stocktwits:{msg['id']}",
        "source": "stocktwits",
        "author": author,
        "text": text,
        "tickers": tickers,
        "lang": "en",
        "likes": max(int(likes), 0),
        "retweets": 0,
        "replies": 0,
        "url": f"https://stocktwits.com/{author}/message/{msg['id']}",
        "published_at": datetime.fromisoformat((msg.get("created_at") or "").replace("Z", "+00:00")),
        "label": sentiment.get("basic"),
    }


def fetch_stream(http_client: httpx.Client, symbol: str) -> list[dict]:
    resp = get_with_backoff(
        f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
        client=http_client,
    )
    rows = []
    for msg in resp.json().get("messages", []):
        row = map_message(msg)
        if row is not None:
            rows.append(row)
    return rows


def poll_symbols(ch_client, http_client: httpx.Client | None = None,
                 symbols: tuple[str, ...] = SYMBOLS) -> int:
    """Fetch each symbol's stream and insert messages.

    A symbol that fails is skipped, never fatal.
    """
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for symbol in symbols:
            try:
                inserted += insert_posts(ch_client, fetch_stream(http_client, symbol))
            except Exception as exc:
                print(f"stocktwits: skipping {symbol}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
```

- [ ] **Step 4: Add the pipeline stage in `src/ingestion/pipeline.py`**

Add the import (with the other stage imports):

```python
from src.ingestion.stocktwits import poll_symbols
```

Add the stage between rss and score in `run_cycle`:

```python
    stages = (
        ("reddit", lambda: poll_subreddits(ch_client)),
        ("rss", lambda: poll_feeds(ch_client)),
        ("stocktwits", lambda: poll_symbols(ch_client)),
        ("score", lambda: score_pending_posts(ch_client, qd_client)),
        ("aggregate", lambda: run_aggregation(ch_client)),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all 55 tests PASS (50 + 5 new stocktwits tests; the 3 pipeline tests pass in their updated form).

- [ ] **Step 6: Live smoke run**

With the Docker stack and Ollama running:

Run: `python -m src.ingestion.pipeline --once`
Expected: five `pipeline: <stage> done (N)` lines including `stocktwits done (N)` with N > 0; exit code 0. Then verify labels landed:

Run: `curl -s --user "alpha:" "http://localhost:8123/?query=SELECT%20source%2C%20count()%2C%20countIf(label%20IS%20NOT%20NULL)%20FROM%20alpha.sentiment_posts%20FINAL%20GROUP%20BY%20source%20FORMAT%20PrettyCompact"`
Expected: a `stocktwits` row with count() > 0 and some labeled messages.

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/stocktwits.py src/ingestion/pipeline.py tests/test_stocktwits.py tests/test_pipeline.py
git commit -m "feat: add StockTwits poller with Bullish/Bearish label persistence"
```

---

## Self-Review Notes

- **Spec coverage:** label column + idempotent ALTER (Task 1), `posts.py` mapping (Task 1), `stocktwits.py` with SYMBOLS/map_message/fetch_stream/poll_symbols (Task 2), pipeline stage + order (Task 2), tests for all of the above, no new dependencies, rate-limit note in Global Constraints. All spec sections covered.
- **Type consistency:** `map_message(dict) -> dict | None`, `fetch_stream(httpx.Client, str) -> list[dict]`, `poll_symbols(ch_client, http_client=None, symbols=SYMBOLS) -> int`, `SYMBOLS: tuple[str, ...]` are spelled identically in interfaces, code, and tests. Row dict keys match `SENTIMENT_POST_COLUMNS` (Task 1 adds `label` before Task 2 produces it).
- **Ordering dependency:** Task 2's `insert_posts` call path requires Task 1's `label` column — tasks must run in order (they are sequential in the plan).
- **Test counts:** 49 (current) + 1 (Task 1) = 50; + 5 (Task 2) = 55. Pipeline tests modified in place, not added.
