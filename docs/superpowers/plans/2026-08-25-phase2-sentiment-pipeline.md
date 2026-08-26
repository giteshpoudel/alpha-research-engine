# Phase 2 Sentiment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 2 sentiment pipeline: Reddit + RSS ingestion into `sentiment_posts`, embedding-distance sentiment scoring via local Ollama, and 5m/1h/24h aggregation into a new `sentiment_metrics` table — all idempotent and orchestrated by one CLI.

**Architecture:** Modular polling stages (`reddit.py`, `rss.py`, `scorer.py`, `aggregator.py`) run in sequence by `pipeline.py`. Posts land in the Phase 1 `sentiment_posts` table; the scorer fills `sentiment_score` via batched ClickHouse mutations and upserts vectors into the Phase 1 Qdrant collection; the aggregator recomputes buckets with replace semantics.

**Tech Stack:** Python 3.11, `clickhouse-connect`, `qdrant-client`, `httpx` (new), `feedparser` (new), Ollama `nomic-embed-text` (768-dim), pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-phase2-sentiment-pipeline-design.md`

## Global Constraints

- All stages are **idempotent** — safe to re-run any time; dedup is structural (ReplacingMergeTree + deterministic `post_id` / Qdrant uuid5 point IDs).
- `published_at` is always the item's own timestamp, never ingestion time (RSS items missing a date fall back to ingest time — the one documented exception).
- All external HTTP (Reddit, RSS, Ollama): `httpx` with timeouts and exponential backoff on 429/5xx/transport errors (1s → 2s → 4s … capped at 10s, max 4 attempts). Non-retryable status (e.g. 403) raises immediately.
- No new API keys: Reddit uses unauthenticated JSON endpoints with a descriptive User-Agent; RSS needs nothing.
- Embed model is exactly `nomic-embed-text` via `POST {OLLAMA_HOST}:11434/api/embed` — 768-dim vectors, matching the existing Qdrant collection.
- Score formula: `clamp(cos(v, pos_anchor) − cos(v, neg_anchor), −1, +1)`; anchors are mean vectors of module-constant phrase lists.
- Only new dependencies: `httpx`, `feedparser`. No other new packages.
- Existing Phase 1 code (`schemas.py` DDL for the 3 original tables, `wait_for_services`, `main`) must keep working; the 16 existing tests must stay green.
- Test hygiene: integration tests seed rows with `post_id` starting with `test:` (and metrics with ticker `TEST`), and delete them in teardown. Tests never truncate tables or touch real rows. Tests run from the repo root with the venv active: `python -m pytest tests/ -v`.
- Work happens in the repo root on a feature branch/worktree chosen at execution time.

---

### Task 1: Dependencies + `sentiment_metrics` table

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ingestion/schemas.py` (add one DDL template + register it)
- Test: `tests/test_schemas.py` (append tests)

**Interfaces:**
- Consumes: existing `create_clickhouse_schema(client)`, `database_name()` from `src/ingestion/schemas.py`.
- Produces: table `sentiment_metrics` with columns `bucket_size LowCardinality(String), bucket_start DateTime64(3), ticker String, post_count UInt32, mean_score Float32, weighted_score Float32, engagement_total UInt64, velocity Float32, engagement_ratio Float32, computed_at DateTime64(3)`, engine `ReplacingMergeTree`, `ORDER BY (bucket_size, bucket_start, ticker)`. Task 6 inserts into it.

- [ ] **Step 1: Add dependencies and install**

Append to `requirements.txt`:

```
httpx>=0.27
feedparser>=6.0
```

Run: `source .venv/bin/activate && pip install -r requirements.txt`
Expected: both install cleanly.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_schemas.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py -v -k sentiment_metrics`
Expected: FAIL — `Code: 60 ... Table alpha.sentiment_metrics does not exist`.

- [ ] **Step 4: Add the DDL to `src/ingestion/schemas.py`**

Add after `_SENTIMENT_POSTS_DDL`:

```python
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
```

In `create_clickhouse_schema`, extend the DDL loop tuple to include it:

```python
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL, _SENTIMENT_METRICS_DDL):
        client.command(ddl.format(db=db))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: all 18 tests PASS (16 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/ingestion/schemas.py tests/test_schemas.py
git commit -m "feat: add sentiment_metrics table schema and pipeline dependencies"
```

---

### Task 2: Ticker extraction (`tickers.py`)

**Files:**
- Create: `src/ingestion/tickers.py`
- Test: `tests/test_tickers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TICKER_ALIASES: dict[str, tuple[str, ...]]` and `extract_tickers(text: str) -> list[str]` (sorted, deduplicated, uppercase symbols) — used by `reddit.py` and `rss.py`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tickers.py`:

```python
from src.ingestion.tickers import extract_tickers


def test_cashtags():
    assert extract_tickers("$BTC and $eth are pumping") == ["BTC", "ETH"]


def test_aliases_case_insensitive():
    assert extract_tickers("Bitcoin rallies while Ethereum dumps") == ["BTC", "ETH"]


def test_dedup_and_sorted():
    assert extract_tickers("bitcoin BTC $btc Bitcoin") == ["BTC"]


def test_no_false_positives_on_dollar_amounts():
    assert extract_tickers("Price target is $100 this year") == []


def test_alias_word_boundaries():
    # "ether" alone is not an alias; "solana" is, "solace" is not
    assert extract_tickers("The ether between us, what solace") == []
    assert extract_tickers("solana summer") == ["SOL"]


def test_no_ambiguous_aliases():
    # 'link', 'dot', 'ada' as plain words must NOT match (chainlink/polkadot/cardano only)
    assert extract_tickers("check this link about the dot com era, ada lovelace") == []
    assert extract_tickers("chainlink and polkadot and cardano") == ["ADA", "DOT", "LINK"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tickers.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.tickers'`.

- [ ] **Step 3: Implement `src/ingestion/tickers.py`**

```python
"""Crypto ticker/entity extraction from social text.

Two signals: cashtags ($BTC) and an alias map (bitcoin -> BTC). Aliases are
deliberately conservative — ambiguous English words ('link', 'dot', 'ada')
only match their unambiguous project names.
"""

from __future__ import annotations

import re

TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
    "SOL": ("sol", "solana"),
    "XRP": ("xrp", "ripple"),
    "DOGE": ("doge", "dogecoin"),
    "ADA": ("cardano",),
    "AVAX": ("avax", "avalanche"),
    "LINK": ("chainlink",),
    "DOT": ("polkadot",),
    "BNB": ("bnb", "binance coin"),
    "MATIC": ("matic", "polygon"),
    "LTC": ("ltc", "litecoin"),
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{2,10})\b")
_ALIAS_RES: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
    )
    for ticker, aliases in TICKER_ALIASES.items()
}


def extract_tickers(text: str) -> list[str]:
    """Return sorted, deduplicated ticker symbols mentioned in text."""
    found: set[str] = set()
    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        if symbol in TICKER_ALIASES:
            found.add(symbol)
    for ticker, pattern in _ALIAS_RES.items():
        if pattern.search(text):
            found.add(ticker)
    return sorted(found)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tickers.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tickers.py tests/test_tickers.py
git commit -m "feat: add crypto ticker extraction from social text"
```

---

### Task 3: Shared HTTP/insert helpers + Reddit poller

**Files:**
- Create: `src/ingestion/http.py`
- Create: `src/ingestion/posts.py`
- Create: `src/ingestion/reddit.py`
- Test: `tests/test_http.py`, `tests/test_reddit.py`

**Interfaces:**
- Consumes: `get_clickhouse_client`, `database_name` from `src/ingestion/schemas.py`; `extract_tickers` from Task 2.
- Produces:
  - `http.get_with_backoff(url: str, *, client: httpx.Client | None = None, headers: dict | None = None, params: dict | None = None, max_attempts: int = 4, timeout: float = 15.0) -> httpx.Response` (used by `rss.py` too)
  - `posts.SENTIMENT_POST_COLUMNS: tuple[str, ...]` and `posts.insert_posts(ch_client, rows: list[dict]) -> int` (used by `rss.py` too). Row dict keys: `post_id, source, author, text, tickers, lang, likes, retweets, replies, url, published_at` (all optional except `post_id`, `source`, `text`; `published_at` may be `None` → ingest time).
  - `reddit.SUBREDDITS: tuple[str, ...]`, `reddit.map_post(post: dict) -> dict | None`, `reddit.fetch_new_posts(http_client, subreddit: str, limit: int = 100) -> list[dict]`, `reddit.poll_subreddits(ch_client, http_client=None, subreddits=SUBREDDITS) -> int` (used by `pipeline.py`).

- [ ] **Step 1: Write the failing tests for `http.py`**

`tests/test_http.py`:

```python
import httpx
import pytest

from src.ingestion.http import get_with_backoff


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_success_first_try():
    client = _client(lambda req: httpx.Response(200, json={"ok": True}))
    resp = get_with_backoff("https://example.com/x", client=client)
    assert resp.status_code == 200


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    resp = get_with_backoff("https://example.com/x", client=_client(handler))
    assert resp.status_code == 200
    assert len(calls) == 3


def test_non_retryable_status_raises_immediately():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(403)

    with pytest.raises(RuntimeError, match="403"):
        get_with_backoff("https://example.com/x", client=_client(handler))
    assert len(calls) == 1


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="429"):
        get_with_backoff("https://example.com/x",
                         client=_client(lambda req: httpx.Response(429)))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_http.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.http'`.

- [ ] **Step 3: Implement `src/ingestion/http.py`**

```python
"""Shared HTTP helper: GET with timeouts and exponential backoff.

Retries 429, 5xx, and transport errors (1s, 2s, 4s ... capped at 10s,
max 4 attempts). Any other >=400 status raises immediately.
"""

from __future__ import annotations

import time

import httpx

USER_AGENT = "alpha-research-engine/0.1 (local quant research)"


def get_with_backoff(
    url: str,
    *,
    client: httpx.Client | None = None,
    headers: dict | None = None,
    params: dict | None = None,
    max_attempts: int = 4,
    timeout: float = 15.0,
) -> httpx.Response:
    merged_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    owns_client = client is None
    client = client or httpx.Client()
    delay = 1.0
    try:
        for attempt in range(max_attempts):
            try:
                resp = client.get(
                    url, headers=merged_headers, params=params,
                    timeout=timeout, follow_redirects=True,
                )
            except httpx.TransportError:
                resp = None
            if resp is not None and resp.status_code < 400:
                return resp
            status = resp.status_code if resp is not None else "connection error"
            retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == max_attempts - 1:
                raise RuntimeError(f"GET {url} failed with status {status}")
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
    finally:
        if owns_client:
            client.close()
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_http.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Write the failing tests for `posts.py` + `reddit.py`**

`tests/test_reddit.py`:

```python
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.posts import insert_posts
from src.ingestion.reddit import fetch_new_posts, map_post, poll_subreddits
from src.ingestion.schemas import database_name, get_clickhouse_client

LISTING_FIXTURE = {
    "data": {
        "children": [
            {"data": {
                "name": "t3_abc", "author": "trader1",
                "title": "Bitcoin breaking out", "selftext": "BTC looks bullish",
                "score": 42, "num_comments": 7, "created_utc": 1700000000.0,
                "permalink": "/r/CryptoCurrency/comments/abc/x/",
            }},
            {"data": {
                "name": "t3_bot", "author": "AutoModerator",
                "title": "Daily discussion", "selftext": "thread",
                "score": 1, "num_comments": 100, "created_utc": 1700000060.0,
                "permalink": "/r/CryptoCurrency/comments/bot/y/",
            }},
            {"data": {
                "name": "t3_empty", "author": "lurker",
                "title": "", "selftext": "",
                "score": 0, "num_comments": 0, "created_utc": 1700000120.0,
                "permalink": "/r/CryptoCurrency/comments/empty/z/",
            }},
        ]
    }
}


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_map_post_fields():
    row = map_post(LISTING_FIXTURE["data"]["children"][0]["data"])
    assert row["post_id"] == "reddit:t3_abc"
    assert row["source"] == "reddit"
    assert row["text"] == "Bitcoin breaking out\nBTC looks bullish"
    assert row["tickers"] == ["BTC"]
    assert row["likes"] == 42
    assert row["replies"] == 7
    assert row["url"] == "https://www.reddit.com/r/CryptoCurrency/comments/abc/x/"
    assert row["published_at"] == datetime.fromtimestamp(1700000000.0, tz=timezone.utc)


def test_map_post_skips_bots_and_empty():
    assert map_post(LISTING_FIXTURE["data"]["children"][1]["data"]) is None
    assert map_post(LISTING_FIXTURE["data"]["children"][2]["data"]) is None


def test_fetch_new_posts_filters():
    client = _mock_client(lambda req: httpx.Response(200, json=LISTING_FIXTURE))
    rows = fetch_new_posts(client, "CryptoCurrency")
    assert [r["post_id"] for r in rows] == ["reddit:t3_abc"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')"
    )


def test_poll_subreddits_inserts_and_is_idempotent(ch_client):
    fixture = {
        "data": {"children": [{"data": {
            "name": "t3_test1", "author": "tester",
            "title": "test: bitcoin news", "selftext": "",
            "score": 5, "num_comments": 2, "created_utc": 1700000000.0,
            "permalink": "/r/Bitcoin/comments/test1/t/",
        }}]}
    }
    # map_post keys post_id off the fullname; force the test: prefix
    def handler(req):
        return httpx.Response(200, json=fixture)

    http_client = _mock_client(handler)
    import src.ingestion.reddit as reddit_mod
    original = reddit_mod.map_post
    reddit_mod.map_post = lambda post: {**original(post), "post_id": "test:" + original(post)["post_id"]}
    try:
        assert poll_subreddits(ch_client, http_client=http_client, subreddits=("Bitcoin",)) == 1
        assert poll_subreddits(ch_client, http_client=http_client, subreddits=("Bitcoin",)) == 1
        rows = ch_client.query(
            f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
            "WHERE post_id = 'test:reddit:t3_test1'"
        ).result_rows
        assert rows[0][0] == 1
    finally:
        reddit_mod.map_post = original
```

- [ ] **Step 6: Run to verify failure**

Run: `python -m pytest tests/test_reddit.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.posts'`.

- [ ] **Step 7: Implement `src/ingestion/posts.py`**

```python
"""Shared insert path for sentiment_posts rows."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name

SENTIMENT_POST_COLUMNS = (
    "post_id", "source", "author", "text", "tickers", "lang",
    "likes", "retweets", "replies", "url",
    "published_at", "ingested_at", "sentiment_score",
)


def insert_posts(ch_client, rows: list[dict]) -> int:
    """Insert row dicts into sentiment_posts. Returns rows inserted.

    Dedup is structural: the table's ReplacingMergeTree collapses repeated
    (source, post_id) keys on merge, so re-inserting the same post is safe.
    """
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [
        [
            row["post_id"],
            row["source"],
            row.get("author", ""),
            row["text"],
            row.get("tickers", []),
            row.get("lang", ""),
            row.get("likes", 0),
            row.get("retweets", 0),
            row.get("replies", 0),
            row.get("url", ""),
            row["published_at"] or now,  # documented fallback: dateless RSS items
            now,
            None,
        ]
        for row in rows
    ]
    ch_client.insert(
        f"{database_name()}.sentiment_posts",
        data,
        column_names=list(SENTIMENT_POST_COLUMNS),
    )
    return len(rows)
```

- [ ] **Step 8: Implement `src/ingestion/reddit.py`**

```python
"""Reddit poller: /r/{sub}/new.json -> sentiment_posts (source='reddit').

Unauthenticated public JSON endpoints; one request per subreddit per cycle
keeps us well under rate limits. Noise filter: known bot authors and posts
with no text content.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.tickers import extract_tickers

SUBREDDITS = ("CryptoCurrency", "Bitcoin", "ethereum", "solana", "CryptoMarkets")

_BOT_AUTHOR_RE = re.compile(r"(^automoderator$|_bot$|^bot_|bot$)", re.IGNORECASE)


def map_post(post: dict) -> dict | None:
    """Map a Reddit listing child's `data` to a row dict, or None to skip."""
    author = post.get("author") or ""
    if _BOT_AUTHOR_RE.search(author):
        return None
    title = (post.get("title") or "").strip()
    body = (post.get("selftext") or "").strip()
    text = f"{title}\n{body}".strip()
    if not text:
        return None
    return {
        "post_id": f"reddit:{post['name']}",  # fullname, e.g. t3_abc123
        "source": "reddit",
        "author": author,
        "text": text,
        "tickers": extract_tickers(text),
        "lang": "en",
        "likes": max(int(post.get("score") or 0), 0),
        "retweets": 0,
        "replies": max(int(post.get("num_comments") or 0), 0),
        "url": "https://www.reddit.com" + (post.get("permalink") or ""),
        "published_at": datetime.fromtimestamp(float(post["created_utc"]), tz=timezone.utc),
    }


def fetch_new_posts(http_client: httpx.Client, subreddit: str, limit: int = 100) -> list[dict]:
    resp = get_with_backoff(
        f"https://www.reddit.com/r/{subreddit}/new.json",
        client=http_client,
        params={"limit": str(limit), "raw_json": "1"},
    )
    children = resp.json().get("data", {}).get("children", [])
    rows = []
    for child in children:
        row = map_post(child.get("data", {}))
        if row is not None:
            rows.append(row)
    return rows


def poll_subreddits(ch_client, http_client: httpx.Client | None = None,
                    subreddits: tuple[str, ...] = SUBREDDITS) -> int:
    """Fetch the newest posts from each subreddit and insert them.

    A subreddit that fails (after backoff) is skipped, never fatal.
    """
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for sub in subreddits:
            try:
                inserted += insert_posts(ch_client, fetch_new_posts(http_client, sub))
            except RuntimeError as exc:
                print(f"reddit: skipping r/{sub}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
```

- [ ] **Step 9: Run to verify pass**

Run: `python -m pytest tests/test_http.py tests/test_reddit.py -v`
Expected: 4 + 4 tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/ingestion/http.py src/ingestion/posts.py src/ingestion/reddit.py tests/test_http.py tests/test_reddit.py
git commit -m "feat: add reddit poller with shared http/insert helpers"
```

---

### Task 4: RSS news poller (`rss.py`)

**Files:**
- Create: `src/ingestion/rss.py`
- Test: `tests/test_rss.py`

**Interfaces:**
- Consumes: `http.get_with_backoff`, `posts.insert_posts`, `tickers.extract_tickers` from Tasks 2–3.
- Produces: `rss.FEEDS: tuple[tuple[str, str], ...]`, `rss.map_entry(feed_name: str, entry) -> dict | None`, `rss.fetch_feed(feed_name: str, url: str, http_client=None) -> list[dict]`, `rss.poll_feeds(ch_client, http_client=None, feeds=FEEDS) -> int` (used by `pipeline.py`).

- [ ] **Step 1: Write the failing tests**

`tests/test_rss.py`:

```python
from datetime import datetime, timezone

import feedparser
import httpx
import pytest

from src.ingestion.rss import fetch_feed, map_entry, poll_feeds
from src.ingestion.schemas import database_name, get_clickhouse_client

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>CoinTelegraph</title>
<item>
  <title>Ethereum ETF approved, ETH rallies</title>
  <link>https://example.com/eth-etf</link>
  <guid>https://example.com/eth-etf</guid>
  <pubDate>Tue, 14 Nov 2023 22:00:00 +0000</pubDate>
  <description>ETH jumps on &lt;b&gt;ETF&lt;/b&gt; approval</description>
</item>
<item>
  <title></title>
  <link>https://example.com/empty</link>
  <guid>https://example.com/empty</guid>
</item>
</channel></rss>
"""


def _entries():
    return feedparser.parse(FEED_XML).entries


def test_map_entry_fields():
    row = map_entry("CoinTelegraph", _entries()[0])
    assert row["post_id"] == "news:https://example.com/eth-etf"
    assert row["source"] == "news"
    assert row["author"] == "CoinTelegraph"
    assert row["text"] == "Ethereum ETF approved, ETH rallies\nETH jumps on ETF approval"
    assert row["tickers"] == ["ETH"]
    assert row["likes"] == 0 and row["retweets"] == 0 and row["replies"] == 0
    assert row["published_at"] == datetime(2023, 11, 14, 22, 0, tzinfo=timezone.utc)


def test_map_entry_skips_empty():
    assert map_entry("CoinTelegraph", _entries()[1]) is None


def test_fetch_feed_parses_and_filters():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=FEED_XML)))
    rows = fetch_feed("CoinTelegraph", "https://example.com/rss", http_client=client)
    assert [r["post_id"] for r in rows] == ["news:https://example.com/eth-etf"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')"
    )


def test_poll_feeds_inserts_and_is_idempotent(ch_client):
    http_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=FEED_XML)))
    import src.ingestion.rss as rss_mod
    original = rss_mod.map_entry
    rss_mod.map_entry = lambda name, entry: (
        None if original(name, entry) is None
        else {**original(name, entry), "post_id": "test:" + original(name, entry)["post_id"]}
    )
    try:
        assert poll_feeds(ch_client, http_client=http_client,
                          feeds=(("CoinTelegraph", "https://example.com/rss"),)) == 1
        assert poll_feeds(ch_client, http_client=http_client,
                          feeds=(("CoinTelegraph", "https://example.com/rss"),)) == 1
        rows = ch_client.query(
            f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
            "WHERE post_id = 'test:news:https://example.com/eth-etf'"
        ).result_rows
        assert rows[0][0] == 1
    finally:
        rss_mod.map_entry = original
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_rss.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.rss'`.

- [ ] **Step 3: Implement `src/ingestion/rss.py`**

```python
"""RSS news poller: crypto news feeds -> sentiment_posts (source='news').

Items missing a published date get published_at=None and insert_posts
falls back to ingest time (the one documented exception to the
'published_at is the item's own timestamp' rule).
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timezone

import feedparser
import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.tickers import extract_tickers

FEEDS = (
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt", "https://decrypt.co/feed"),
)

_TAG_RE = re.compile(r"<[^>]+>")


def map_entry(feed_name: str, entry) -> dict | None:
    """Map a feedparser entry to a row dict, or None to skip."""
    title = (entry.get("title") or "").strip()
    summary = " ".join(_TAG_RE.sub(" ", entry.get("summary") or "").split())
    text = f"{title}\n{summary}".strip()
    if not text:
        return None
    guid = entry.get("id") or entry.get("link")
    if not guid:
        return None
    parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
    published_at = (
        datetime.fromtimestamp(calendar.timegm(parsed_date), tz=timezone.utc)
        if parsed_date else None
    )
    return {
        "post_id": f"news:{guid}",
        "source": "news",
        "author": feed_name,
        "text": text,
        "tickers": extract_tickers(text),
        "lang": "en",
        "likes": 0,
        "retweets": 0,
        "replies": 0,
        "url": entry.get("link") or "",
        "published_at": published_at,
    }


def fetch_feed(feed_name: str, url: str, http_client: httpx.Client | None = None) -> list[dict]:
    resp = get_with_backoff(url, client=http_client)
    parsed = feedparser.parse(resp.content)
    rows = []
    for entry in parsed.entries:
        row = map_entry(feed_name, entry)
        if row is not None:
            rows.append(row)
    return rows


def poll_feeds(ch_client, http_client: httpx.Client | None = None,
               feeds: tuple[tuple[str, str], ...] = FEEDS) -> int:
    """Fetch each feed and insert new items. A failing feed is skipped, never fatal."""
    inserted = 0
    for name, url in feeds:
        try:
            inserted += insert_posts(ch_client, fetch_feed(name, url, http_client=http_client))
        except RuntimeError as exc:
            print(f"rss: skipping {name}: {exc}")
    return inserted
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_rss.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/rss.py tests/test_rss.py
git commit -m "feat: add RSS news poller"
```

---

### Task 5: Sentiment scorer (`scorer.py`)

**Files:**
- Create: `src/ingestion/scorer.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Consumes: `database_name` from `schemas.py`; `QDRANT_COLLECTION`, `post_id_to_uuid` from `schemas.py`; live Ollama at `OLLAMA_HOST` (env, default `localhost`) port 11434.
- Produces:
  - `POSITIVE_PHRASES: tuple[str, ...]`, `NEGATIVE_PHRASES: tuple[str, ...]`
  - `sentiment_score(vector, pos_anchor, neg_anchor) -> float` (pure math)
  - `fetch_unscored(ch_client, limit: int = 256) -> list[tuple]` — rows `(post_id, source, text, tickers, published_at)` with NULL score, oldest first
  - `score_pending_posts(ch_client, qd_client, http_client=None, limit: int = 256) -> int` (used by `pipeline.py`)

- [ ] **Step 1: Write the failing tests**

`tests/test_scorer.py`:

```python
import json
import math
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.schemas import (
    QDRANT_COLLECTION,
    database_name,
    get_clickhouse_client,
    get_qdrant_client,
    post_id_to_uuid,
)
from src.ingestion.scorer import (
    fetch_unscored,
    sentiment_score,
    score_pending_posts,
)


def test_sentiment_score_pure_math():
    pos = [1.0, 0.0]
    neg = [-1.0, 0.0]
    assert sentiment_score([1.0, 0.0], pos, neg) == pytest.approx(1.0)
    assert sentiment_score([-1.0, 0.0], pos, neg) == pytest.approx(-1.0)
    # orthogonal text scores 0
    assert sentiment_score([0.0, 1.0], pos, neg) == pytest.approx(0.0)
    # never exceeds [-1, 1] even when both cosines are extreme
    assert -1.0 <= sentiment_score([0.9, 0.1], pos, neg) <= 1.0


def _fake_ollama_client(vectors_by_text: dict[str, list[float]]):
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [vectors_by_text.get(t, [0.0] * 768) for t in texts]
        return httpx.Response(200, json={"embeddings": embeddings})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def clients():
    ch = get_clickhouse_client()
    qd = get_qdrant_client()
    yield ch, qd
    ch.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')"
    )
    qd.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=["00000000-0000-0000-0000-000000000000"],  # no-op placeholder
        wait=True,
    )


def _seed_unscored(ch, post_id, text, published_at):
    ch.insert(
        f"{database_name()}.sentiment_posts",
        [[post_id, "reddit", "tester", text, ["BTC"], "en",
          10, 0, 3, "https://example.com", published_at,
          datetime.now(timezone.utc), None]],
        column_names=["post_id", "source", "author", "text", "tickers", "lang",
                      "likes", "retweets", "replies", "url",
                      "published_at", "ingested_at", "sentiment_score"],
    )


def test_fetch_unscored_ignores_scored(clients):
    ch, _ = clients
    _seed_unscored(ch, "test:unscored-1", "seed text", datetime(2023, 11, 14, tzinfo=timezone.utc))
    rows = fetch_unscored(ch, limit=1000)
    ids = [r[0] for r in rows]
    assert "test:unscored-1" in ids


def test_score_pending_posts_end_to_end(clients):
    ch, qd = clients
    _seed_unscored(ch, "test:score-me", "bullish bitcoin moon", datetime(2023, 11, 14, tzinfo=timezone.utc))

    bull = [1.0] + [0.0] * 767
    bear = [-1.0] + [0.0] * 767
    # Anchor phrases and the post text all get the bullish vector -> score +1
    import src.ingestion.scorer as scorer_mod
    vectors = {t: bull for t in scorer_mod.POSITIVE_PHRASES}
    vectors.update({t: bear for t in scorer_mod.NEGATIVE_PHRASES})
    vectors["bullish bitcoin moon"] = bull
    http_client = _fake_ollama_client(vectors)

    scored = score_pending_posts(ch, qd, http_client=http_client, limit=1000)
    assert scored >= 1

    rows = ch.query(
        f"SELECT sentiment_score FROM {database_name()}.sentiment_posts FINAL "
        "WHERE post_id = 'test:score-me'"
    ).result_rows
    assert rows[0][0] == pytest.approx(1.0, abs=1e-5)

    points = qd.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[post_id_to_uuid("test:score-me")],
        with_payload=True,
    )
    assert len(points) == 1
    assert points[0].payload["post_id"] == "test:score-me"
    assert points[0].payload["tickers"] == ["BTC"]
    assert points[0].payload["score"] == pytest.approx(1.0, abs=1e-5)
    qd.delete(collection_name=QDRANT_COLLECTION,
              points_selector=[post_id_to_uuid("test:score-me")], wait=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_scorer.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.scorer'`.

- [ ] **Step 3: Implement `src/ingestion/scorer.py`**

```python
"""Embedding-distance sentiment scorer.

score = clamp(cos(v, pos_anchor) - cos(v, neg_anchor), -1, +1), where anchors
are mean embeddings of the phrase lists below. Deterministic: same text ->
same score. Also upserts each post's vector into Qdrant with its score as
payload, then applies scores to ClickHouse in one batched mutation.
"""

from __future__ import annotations

import math
import os
import time
from datetime import timezone

import httpx
from qdrant_client.models import PointStruct

from src.ingestion.schemas import QDRANT_COLLECTION, database_name, post_id_to_uuid

EMBED_MODEL = "nomic-embed-text"

POSITIVE_PHRASES = (
    "bitcoin is going to the moon, extremely bullish, buy now",
    "huge breakout, massive gains ahead, so bullish",
    "accumulating here, long term hold, undervalued gem",
    "adoption is growing, institutions are buying, bullish news",
    "great recovery, strong fundamentals, green candles",
    "to the moon, rocket emoji, all time high incoming",
    "buy the dip, this is the bottom, reversal incoming",
    "bull market confirmed, prices will keep rising",
)

NEGATIVE_PHRASES = (
    "bitcoin is crashing, extremely bearish, sell everything",
    "huge dump incoming, massive losses, so bearish",
    "this is a scam, rug pull, ponzi scheme, fraud",
    "dead cat bounce, going to zero, capitulation",
    "panic selling, blood in the streets, red candles everywhere",
    "bear market confirmed, prices will keep falling",
    "get out now, top is in, distribute before the crash",
    "fud, hacks, exchange collapse, funds are gone",
)


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "localhost")


def _embed(http_client: httpx.Client, texts: list[str]) -> list[list[float]]:
    resp = http_client.post(
        f"http://{_ollama_host()}:11434/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def sentiment_score(vector, pos_anchor, neg_anchor) -> float:
    raw = _cosine(vector, pos_anchor) - _cosine(vector, neg_anchor)
    return max(-1.0, min(1.0, raw))


def _to_unix(dt) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_unscored(ch_client, limit: int = 256) -> list[tuple]:
    """Oldest-first posts with NULL sentiment_score.

    Returns (post_id, source, text, tickers, published_at) tuples.
    """
    return ch_client.query(
        f"SELECT post_id, source, text, tickers, published_at "
        f"FROM {database_name()}.sentiment_posts "
        "WHERE sentiment_score IS NULL "
        "ORDER BY published_at ASC LIMIT {n:UInt32}",
        parameters={"n": limit},
    ).result_rows


def _wait_for_mutations(ch_client, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = ch_client.query(
            "SELECT count() FROM system.mutations "
            "WHERE database = {db:String} AND table = 'sentiment_posts' AND is_done = 0",
            parameters={"db": database_name()},
        ).result_rows
        if rows[0][0] == 0:
            return
        time.sleep(1.0)
    raise RuntimeError("Timed out waiting for ClickHouse mutation to apply")


def _apply_scores(ch_client, scored: list[tuple[str, str, float]]) -> None:
    """One batched mutation: (source, post_id) -> sentiment_score."""
    branches, conditions, params = [], [], {}
    for i, (source, post_id, score) in enumerate(scored):
        conditions.append(f"(source = {{s{i}:String}} AND post_id = {{p{i}:String}})")
        branches.append(f"source = {{s{i}:String}} AND post_id = {{p{i}:String}}, {{v{i}:Float32}}")
        params[f"s{i}"] = source
        params[f"p{i}"] = post_id
        params[f"v{i}"] = score
    case_expr = "multiIf(" + ", ".join(branches) + ", sentiment_score)"
    where = " OR ".join(conditions)
    ch_client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts "
        f"UPDATE sentiment_score = {case_expr} WHERE {where}",
        parameters=params,
    )
    _wait_for_mutations(ch_client)


def score_pending_posts(ch_client, qd_client, http_client: httpx.Client | None = None,
                        limit: int = 256) -> int:
    """Score up to `limit` unscored posts. Returns how many were scored."""
    rows = fetch_unscored(ch_client, limit)
    if not rows:
        return 0
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    try:
        pos_anchor = _mean_vector(_embed(http_client, list(POSITIVE_PHRASES)))
        neg_anchor = _mean_vector(_embed(http_client, list(NEGATIVE_PHRASES)))
        texts = [text[:4000] for (_, _, text, _, _) in rows]
        vectors = _embed(http_client, texts)
        points, scored = [], []
        for (post_id, source, _text, tickers, published_at), vector in zip(rows, vectors):
            score = sentiment_score(vector, pos_anchor, neg_anchor)
            points.append(PointStruct(
                id=post_id_to_uuid(post_id),
                vector=vector,
                payload={
                    "post_id": post_id,
                    "source": source,
                    "tickers": list(tickers),
                    "published_at": _to_unix(published_at),
                    "score": score,
                },
            ))
            scored.append((source, post_id, score))
        qd_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        _apply_scores(ch_client, scored)
        return len(scored)
    finally:
        if owns_client:
            http_client.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_scorer.py -v`
Expected: 3 tests PASS. (Note: `test_score_pending_posts_end_to_end` uses a mocked Ollama transport but live ClickHouse + Qdrant; the mutation wait may take a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/scorer.py tests/test_scorer.py
git commit -m "feat: add embedding-distance sentiment scorer"
```

---

### Task 6: Metrics aggregator (`aggregator.py`)

**Files:**
- Create: `src/ingestion/aggregator.py`
- Test: `tests/test_aggregator.py`

**Interfaces:**
- Consumes: `database_name` from `schemas.py`; the `sentiment_metrics` table from Task 1.
- Produces:
  - `BUCKET_SIZES: tuple[str, ...]` = `("5m", "1h", "24h")`
  - `align_start(ts: datetime, bucket_size: str) -> datetime`
  - `compute_metrics(ch_client, bucket_size: str, bucket_start: datetime) -> int` — rows written
  - `run_aggregation(ch_client, now: datetime | None = None) -> int` (used by `pipeline.py`)

Metric definitions (exact):
- `post_count` = posts in bucket mentioning the ticker (any score state)
- `mean_score` = avg of non-NULL scores (0.0 when no scored posts)
- `weighted_score` = `sum(score * (1+likes+retweets+replies)) / sum(1+likes+retweets+replies)` over scored posts (0.0 when no scored posts)
- `engagement_total` = `sum(likes+retweets+replies)` over all posts in bucket
- `velocity` = `post_count / (baseline_post_count / n_baseline_buckets)`; 0.0 when baseline is 0
- `engagement_ratio` = same on engagement
- Baselines: `5m` → trailing 24h (288 buckets), `1h` → trailing 24h (24 buckets), `24h` → trailing 7d (7 buckets) — same-size buckets immediately before `bucket_start`.

- [ ] **Step 1: Write the failing tests**

`tests/test_aggregator.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from src.ingestion.aggregator import align_start, compute_metrics, run_aggregation
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_align_start():
    ts = datetime(2024, 1, 15, 10, 7, 33, tzinfo=timezone.utc)
    assert align_start(ts, "5m") == datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc)
    assert align_start(ts, "1h") == datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    assert align_start(ts, "24h") == datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    client.command(f"ALTER TABLE {db}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')")
    client.command(f"ALTER TABLE {db}.sentiment_metrics DELETE WHERE ticker = 'TEST'")


def _seed_post(ch, post_id, tickers, score, likes, replies, published_at):
    ch.insert(
        f"{database_name()}.sentiment_posts",
        [[post_id, "reddit", "tester", "seed", tickers, "en",
          likes, 0, replies, "", published_at, datetime.now(timezone.utc), score]],
        column_names=["post_id", "source", "author", "text", "tickers", "lang",
                      "likes", "retweets", "replies", "url",
                      "published_at", "ingested_at", "sentiment_score"],
    )


def test_compute_metrics_exact_values(ch_client):
    bucket_start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    inside = bucket_start + timedelta(minutes=2)
    # two scored TEST posts in the bucket
    _seed_post(ch_client, "test:agg-1", ["TEST"], 0.5, 10, 0, inside)
    _seed_post(ch_client, "test:agg-2", ["TEST"], -0.5, 30, 10, inside)
    # one unscored TEST post (counts for volume, not scores)
    _seed_post(ch_client, "test:agg-3", ["TEST"], None, 0, 0, inside)
    # baseline: one TEST post in each of the 24 preceding 1h buckets -> mean = 1.0
    for h in range(1, 25):
        _seed_post(ch_client, f"test:agg-base-{h}", ["TEST"], 0.0, 1, 0,
                   bucket_start - timedelta(hours=h))

    assert compute_metrics(ch_client, "1h", bucket_start) >= 1

    rows = ch_client.query(
        f"SELECT post_count, mean_score, weighted_score, engagement_total, "
        f"velocity, engagement_ratio "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' AND bucket_start = {ts:DateTime64(3)}",
        parameters={"ts": bucket_start},
    ).result_rows
    assert len(rows) == 1
    post_count, mean_score, weighted_score, engagement, velocity, eng_ratio = rows[0]
    assert post_count == 3
    assert mean_score == pytest.approx(0.0, abs=1e-6)          # (0.5 + -0.5) / 2
    # weights: p1 = 1+10 = 11, p2 = 1+30+10 = 41 -> (0.5*11 - 0.5*41) / 52
    assert weighted_score == pytest.approx(-15.0 / 52.0, abs=1e-4)
    assert engagement == 50                                     # 10 + (30+10) + 0
    assert velocity == pytest.approx(3.0, abs=1e-4)             # 3 / (24/24)
    assert eng_ratio == pytest.approx(50.0 / 1.0, abs=1e-2)     # 50 / (24*1/24)


def test_run_aggregation_writes_recent_buckets(ch_client):
    bucket_start = align_start(datetime.now(timezone.utc), "1h") - timedelta(hours=1)
    _seed_post(ch_client, "test:agg-run", ["TEST"], 0.25, 2, 1,
               bucket_start + timedelta(minutes=5))
    written = run_aggregation(ch_client)
    assert written >= 1
    rows = ch_client.query(
        f"SELECT post_count FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' AND bucket_start = {ts:DateTime64(3)}",
        parameters={"ts": bucket_start},
    ).result_rows
    assert rows and rows[0][0] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_aggregator.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.aggregator'`.

- [ ] **Step 3: Implement `src/ingestion/aggregator.py`**

```python
"""Time-bucket sentiment metrics: 5m / 1h / 24h per ticker.

Recompute-and-replace: computing a bucket that already exists replaces it
(ReplacingMergeTree on (bucket_size, bucket_start, ticker)), so the
aggregator is idempotent. Baselines are same-size buckets, so velocity and
engagement_ratio are scale-consistent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.ingestion.schemas import database_name

BUCKET_SIZES = ("5m", "1h", "24h")
_BUCKET_SECONDS = {"5m": 300, "1h": 3600, "24h": 86400}
_BASELINE_BUCKETS = {"5m": 288, "1h": 24, "24h": 7}

_METRICS_COLUMNS = (
    "bucket_size", "bucket_start", "ticker", "post_count", "mean_score",
    "weighted_score", "engagement_total", "velocity", "engagement_ratio",
    "computed_at",
)


def align_start(ts: datetime, bucket_size: str) -> datetime:
    epoch = int(ts.timestamp())
    secs = _BUCKET_SECONDS[bucket_size]
    return datetime.fromtimestamp(epoch - (epoch % secs), tz=timezone.utc)


def _window_stats(ch_client, start: datetime, end: datetime) -> dict[str, tuple]:
    """Per-ticker (post_count, mean_score, weighted_score, engagement) for a window."""
    rows = ch_client.query(
        f"""
        SELECT
            arrayJoin(tickers) AS ticker,
            count() AS post_count,
            avgIf(sentiment_score, sentiment_score IS NOT NULL) AS mean_score,
            sumIf(sentiment_score * (1 + likes + retweets + replies),
                  sentiment_score IS NOT NULL) AS weighted_sum,
            sumIf(1 + likes + retweets + replies,
                  sentiment_score IS NOT NULL) AS weight_total,
            sum(likes + retweets + replies) AS engagement_total
        FROM {database_name()}.sentiment_posts
        WHERE published_at >= {{start:DateTime64(3)}} AND published_at < {{end:DateTime64(3)}}
          AND notEmpty(tickers)
        GROUP BY ticker
        """,
        parameters={"start": start, "end": end},
    ).result_rows
    stats = {}
    for ticker, post_count, mean_score, wsum, wtotal, engagement in rows:
        scored_mean = float(mean_score) if mean_score is not None else 0.0
        weighted = float(wsum) / float(wtotal) if wtotal else 0.0
        stats[ticker] = (int(post_count), scored_mean, weighted, int(engagement))
    return stats


def compute_metrics(ch_client, bucket_size: str, bucket_start: datetime) -> int:
    """Recompute one bucket for every ticker mentioned in it. Returns rows written."""
    secs = _BUCKET_SECONDS[bucket_size]
    n_baseline = _BASELINE_BUCKETS[bucket_size]
    bucket_end = bucket_start + timedelta(seconds=secs)
    baseline_start = bucket_start - timedelta(seconds=secs * n_baseline)

    current = _window_stats(ch_client, bucket_start, bucket_end)
    if not current:
        return 0
    baseline = _window_stats(ch_client, baseline_start, bucket_start)

    now = datetime.now(timezone.utc)
    data = []
    for ticker, (post_count, mean_score, weighted, engagement) in current.items():
        base_count, _, _, base_engagement = baseline.get(ticker, (0, 0.0, 0.0, 0))
        mean_count = base_count / n_baseline
        mean_engagement = base_engagement / n_baseline
        velocity = post_count / mean_count if mean_count > 0 else 0.0
        engagement_ratio = engagement / mean_engagement if mean_engagement > 0 else 0.0
        data.append([
            bucket_size, bucket_start, ticker, post_count, mean_score, weighted,
            engagement, velocity, engagement_ratio, now,
        ])
    ch_client.insert(
        f"{database_name()}.sentiment_metrics",
        data,
        column_names=list(_METRICS_COLUMNS),
    )
    return len(data)


def run_aggregation(ch_client, now: datetime | None = None) -> int:
    """Compute the just-closed bucket for each size. Returns total rows written.

    The just-closed 24h bucket only changes at day boundaries; recomputing it
    every cycle is harmless because recompute replaces the same key.
    """
    now = now or datetime.now(timezone.utc)
    total = 0
    for size in BUCKET_SIZES:
        bucket_start = align_start(now, size) - timedelta(seconds=_BUCKET_SECONDS[size])
        total += compute_metrics(ch_client, size, bucket_start)
    return total
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_aggregator.py -v`
Expected: 3 tests PASS. (Note: `avgIf` over `Nullable(Float32)` returns Float64; exact-value assertions use `pytest.approx`. If ClickHouse returns `NaN` instead of `None` for empty `avgIf`, treat NaN as 0.0 in `_window_stats` — check with `math.isnan`.)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (18 schema + 6 tickers + 4 http + 4 reddit + 4 rss + 3 scorer + 3 aggregator = 42).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/aggregator.py tests/test_aggregator.py
git commit -m "feat: add time-bucket sentiment metrics aggregator"
```

---

### Task 7: Pipeline orchestrator (`pipeline.py`)

**Files:**
- Create: `src/ingestion/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `reddit.poll_subreddits(ch_client)`, `rss.poll_feeds(ch_client)`, `scorer.score_pending_posts(ch_client, qd_client)`, `aggregator.run_aggregation(ch_client)`; `get_clickhouse_client`, `get_qdrant_client` from `schemas.py`.
- Produces:
  - `run_cycle(ch_client=None, qd_client=None) -> dict[str, int]` — runs reddit → rss → score → aggregate in order; returns `{stage: count}` with `-1` for a failed stage
  - `main(argv: list[str] | None = None) -> None` — CLI: `--once` or `--interval N` (default 300)
  - Runnable as `python -m src.ingestion.pipeline`

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
import pytest

import src.ingestion.pipeline as pipeline


@pytest.fixture
def staged(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "poll_subreddits", lambda ch: calls.append("reddit") or 3)
    monkeypatch.setattr(pipeline, "poll_feeds", lambda ch: calls.append("rss") or 2)
    monkeypatch.setattr(pipeline, "score_pending_posts",
                        lambda ch, qd: calls.append("score") or 5)
    monkeypatch.setattr(pipeline, "run_aggregation",
                        lambda ch: calls.append("aggregate") or 10)
    return calls


def test_run_cycle_order_and_stats(staged):
    stats = pipeline.run_cycle(ch_client=object(), qd_client=object())
    assert staged == ["reddit", "rss", "score", "aggregate"]
    assert stats == {"reddit": 3, "rss": 2, "score": 5, "aggregate": 10}


def test_run_cycle_stage_failure_is_isolated(monkeypatch, staged):
    def boom(ch):
        raise RuntimeError("reddit down")
    monkeypatch.setattr(pipeline, "poll_subreddits", boom)
    stats = pipeline.run_cycle(ch_client=object(), qd_client=object())
    assert stats["reddit"] == -1
    assert staged == ["rss", "score", "aggregate"]  # later stages still ran


def test_main_once_runs_single_cycle(monkeypatch):
    cycles = []
    monkeypatch.setattr(pipeline, "run_cycle", lambda: cycles.append(1))
    pipeline.main(["--once"])
    assert len(cycles) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.pipeline'`.

- [ ] **Step 3: Implement `src/ingestion/pipeline.py`**

```python
"""Pipeline orchestrator: reddit -> rss -> score -> aggregate.

Usage:
    python -m src.ingestion.pipeline --once          # single cycle
    python -m src.ingestion.pipeline --interval 300  # loop (default 300s)

A failing stage is logged and skipped; later stages still run.
"""

from __future__ import annotations

import argparse
import time

from src.ingestion.aggregator import run_aggregation
from src.ingestion.reddit import poll_subreddits
from src.ingestion.rss import poll_feeds
from src.ingestion.schemas import get_clickhouse_client, get_qdrant_client
from src.ingestion.scorer import score_pending_posts


def run_cycle(ch_client=None, qd_client=None) -> dict[str, int]:
    ch_client = ch_client or get_clickhouse_client()
    qd_client = qd_client or get_qdrant_client()
    stages = (
        ("reddit", lambda: poll_subreddits(ch_client)),
        ("rss", lambda: poll_feeds(ch_client)),
        ("score", lambda: score_pending_posts(ch_client, qd_client)),
        ("aggregate", lambda: run_aggregation(ch_client)),
    )
    stats: dict[str, int] = {}
    for name, fn in stages:
        try:
            stats[name] = fn()
            print(f"pipeline: {name} done ({stats[name]})")
        except Exception as exc:
            stats[name] = -1
            print(f"pipeline: stage {name} failed: {exc}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sentiment ingestion & aggregation pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="run a single cycle and exit")
    group.add_argument("--interval", type=int, default=300,
                       help="seconds between cycles (default 300)")
    args = parser.parse_args(argv)
    if args.once:
        run_cycle()
        return
    while True:
        run_cycle()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Full suite + live smoke run**

Run: `python -m pytest tests/ -v`
Expected: all 45 tests PASS.

Then, with Docker stack and Ollama running, one real cycle against live sources:

Run: `python -m src.ingestion.pipeline --once`
Expected: four `pipeline: <stage> done (N)` lines; exit code 0. (Reddit/RSS counts depend on live data; score/aggregate may be > 0 if posts were ingested.)

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: add sentiment pipeline orchestrator CLI"
```

---

## Self-Review Notes

- **Spec coverage:** `sentiment_metrics` table (Task 1), `tickers.py` (Task 2), `reddit.py` (Task 3), `rss.py` (Task 4), `scorer.py` with anchors/mutation/Qdrant upsert (Task 5), `aggregator.py` with scale-consistent baselines (Task 6), `pipeline.py` CLI with isolated stage failures (Task 7), error handling/backoff (Task 3 `http.py`, used everywhere), testing strategy (per-task tests + live smoke run). New deps limited to `httpx` + `feedparser` (Task 1). All spec sections covered.
- **Type consistency:** `extract_tickers(str) -> list[str]`, `insert_posts(ch_client, list[dict]) -> int`, `map_post(dict) -> dict | None`, `map_entry(str, entry) -> dict | None`, `poll_subreddits(ch_client, http_client=None, subreddits=...) -> int`, `poll_feeds(ch_client, http_client=None, feeds=...) -> int`, `fetch_unscored(ch_client, limit=256) -> list[tuple]`, `score_pending_posts(ch_client, qd_client, http_client=None, limit=256) -> int`, `compute_metrics(ch_client, str, datetime) -> int`, `run_aggregation(ch_client, now=None) -> int`, `run_cycle(ch_client=None, qd_client=None) -> dict[str, int]`, `main(argv=None) -> None` are spelled identically in interfaces, code, and tests.
- **Test counts:** Task 6 Step 5 totals 43 (18+6+4+5+4+3+3); Task 7 adds 3 → 46.
- **Known implementation risks called out in-line:** ClickHouse `avgIf` on all-NULL may return NaN rather than None (Task 6 Step 4 note); mutation parameter binding is exercised by the scorer integration test.
