# StockTwits Poller Design — Social Source for the Sentiment Pipeline

Date: 2026-08-25
Scope: Add a StockTwits poller to the Phase 2 sentiment pipeline as the social data source (Reddit's unauthenticated access is blocked with 403 in the user's environment; Reddit's builder program now requires manual review). Includes persisting StockTwits' user-supplied Bullish/Bearish labels for later scorer validation.

## Decisions

- **Source:** StockTwits public symbol-stream API — `GET https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json`, unauthenticated, verified working 2026-08-25 (200 OK, 30 messages/stream, ~1/3 of messages carry `entities.sentiment.basic` = `Bullish`/`Bearish`).
- **Labels persisted:** new `label Nullable(String)` column on `sentiment_posts`; only StockTwits rows fill it. Enables a future scorer-accuracy report.
- **Pattern:** mirrors the existing `reddit.py`/`rss.py` pollers (fetch → map → `insert_posts`, per-source failure isolation). No new dependencies.
- **Tickers from the API, not regex:** StockTwits messages carry a structured `symbols` array — more accurate than `extract_tickers` for this source.
- **Reddit stage stays in the pipeline:** it degrades gracefully (skips with a log line) and works unchanged if ever run from an environment Reddit allows.

## Rate limits

12 symbols (one per `TICKER_ALIASES` key, mapped to `{TICKER}.X`) × 1 request per cycle. At the default 300s pipeline interval that's ~144 requests/hour, under StockTwits' ~200/hour unauthenticated ceiling. All requests go through the existing `get_with_backoff` (User-Agent, timeout, exponential backoff on 429/5xx).

## Schema change (`src/ingestion/schemas.py`, `src/ingestion/posts.py`)

- `sentiment_posts` CREATE TABLE DDL gains `label Nullable(String)` (after `sentiment_score`) for fresh setups.
- `create_clickhouse_schema()` additionally runs, idempotently, for the existing database:
  `ALTER TABLE {db}.sentiment_posts ADD COLUMN IF NOT EXISTS label Nullable(String)`
- `posts.SENTIMENT_POST_COLUMNS` gains `"label"`; `insert_posts` maps `row.get("label")` (default `None`).

## New module: `src/ingestion/stocktwits.py`

- `SYMBOLS: tuple[str, ...]` — `tuple(f"{t}.X" for t in TICKER_ALIASES)` (12 symbols).
- `map_message(msg: dict) -> dict | None`:
  - `post_id = f"stocktwits:{msg['id']}"`, `source = "stocktwits"`
  - `author = msg["user"]["username"]`, `text = msg["body"].strip()` (empty → None)
  - `tickers` — from `msg["symbols"]`: each entry's `symbol` stripped of the `.X` suffix, kept only if in `TICKER_ALIASES`; sorted, deduplicated
  - `likes = msg.get("likes", {}).get("total", 0)` (absent → 0), `retweets = 0`, `replies = 0`
  - `url = f"https://stocktwits.com/{author}/message/{msg['id']}"`
  - `published_at` — parsed from `created_at` (ISO8601, `Z` suffix); the message's own timestamp (guardrail #2)
  - `label` — `(msg.get("entities") or {}).get("sentiment") or {}` then `.get("basic")` → `"Bullish"` / `"Bearish"` / `None` (the API returns `sentiment: null` for unlabeled messages, so the `or {}` guard is required)
- `fetch_stream(http_client, symbol: str) -> list[dict]` — GET via `get_with_backoff`, map all messages, drop Nones.
- `poll_symbols(ch_client, http_client=None, symbols=SYMBOLS) -> int` — per-symbol `try/except Exception` skip-and-log; returns rows inserted. Client-ownership pattern same as `reddit.poll_subreddits`.

## Pipeline change (`src/ingestion/pipeline.py`)

New stage `("stocktwits", lambda: poll_symbols(ch_client))` inserted between `rss` and `score`. Cycle order becomes: reddit → rss → stocktwits → score → aggregate.

## Downstream behavior (no changes needed)

StockTwits posts flow through the existing scorer (NULL `sentiment_score` selection) and aggregator (ticker bucket metrics) automatically. The scorer's embedding-distance score remains the system of record; `label` is never used as a score input — it exists for validation reporting only.

## Error handling

- All HTTP via `get_with_backoff` (existing policy: retry 429/5xx/transport, 1s→2s→4s cap 10s, max 4 attempts, non-retryable raises immediately).
- A failing symbol is skipped with a log line; other symbols and stages continue.
- Idempotent: re-polling the overlapping 30-message stream window re-inserts the same `post_id`s, collapsed by ReplacingMergeTree.

## Testing

- Schema: `label` column exists with type `Nullable(String)` after `create_clickhouse_schema` re-run (idempotent against the existing DB).
- Mapper unit tests: fixture message with `entities.sentiment.basic = "Bearish"`, `likes.total = 7`, `symbols = ["BTC.X"]` — assert every field, including `label == "Bearish"` and `tickers == ["BTC"]`; unlabeled message → `label is None`; empty body → None.
- Fetch: `httpx.MockTransport` fixture stream → mapped rows.
- Integration (live ClickHouse, `test:` prefix, teardown delete): `poll_symbols` inserts rows, re-run keeps FINAL count at 1 per post, and the stored `label` round-trips.
- Pipeline: order/stats test updated for the 5-stage cycle.
- New dependencies: none.
