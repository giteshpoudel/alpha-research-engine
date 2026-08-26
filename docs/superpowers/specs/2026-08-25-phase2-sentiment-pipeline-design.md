# Phase 2 Design — Sentiment Processing & Signal Generation

Date: 2026-08-25
Scope: Blueprint Phase 2 — sentiment scoring worker (−1.0…+1.0) and time-bucket aggregation (5m, 1h, 24h) with Sentiment Polarity Index, engagement spikes, and ticker mention velocity — plus the thin-slice live ingestion (Reddit + news RSS) needed to feed it. Historical backfill (2022–present) and X/Twitter API ingestion are out of scope.

## Decisions

- **Data sources:** Reddit public JSON endpoints (`/r/{sub}/new.json`, unauthenticated) + crypto news RSS feeds. The user's X API token is valid but the account has no credits (verified 402 `credits-depleted` on 2026-08-25); X ingestion swaps in later behind the same `sentiment_posts` schema.
- **Scoring:** embedding-distance with local Ollama `nomic-embed-text` (768-dim — already pulled and verified, matches the Phase 1 Qdrant collection). Score = `clamp(cos(v, pos_anchor) − cos(v, neg_anchor), −1, +1)` against mean-embedded bullish/bearish anchor phrase sets. Deterministic; embeddings double as the Qdrant vectors.
- **Pipeline shape:** modular polling stages orchestrated by one CLI (rejected: monolithic daemon — coupled failures, hard to test; rejected: ClickHouse materialized views — they fire at INSERT when `sentiment_score` is still NULL).
- **Score writes:** batched `ALTER TABLE … UPDATE` mutations fill `sentiment_score` post-insert (avoids ReplacingMergeTree no-version-column replace ambiguity).

## New ClickHouse table (added to `src/ingestion/schemas.py`)

### `sentiment_metrics`

| Column | Type |
| --- | --- |
| bucket_size | LowCardinality(String) — `5m` / `1h` / `24h` |
| bucket_start | DateTime64(3) |
| ticker | String |
| post_count | UInt32 |
| mean_score | Float32 |
| weighted_score | Float32 — engagement-weighted mean |
| engagement_total | UInt64 |
| velocity | Float32 — post_count ÷ trailing mean of same-size buckets for that ticker (baseline windows below) |
| engagement_ratio | Float32 — engagement_total ÷ same baseline on engagement |
| computed_at | DateTime64(3) |

Engine: `ReplacingMergeTree`, `ORDER BY (bucket_size, bucket_start, ticker)`. Recomputing a bucket replaces it — idempotent.

## New modules in `src/ingestion/`

### `tickers.py`
- `extract_tickers(text: str) -> list[str]` — cashtag regex (`$BTC`) + alias map (`bitcoin`→`BTC`, `ethereum`→`ETH`, …) covering ~15 major tickers as a module-level dict. Case-insensitive on aliases, uppercase output, deduplicated.

### `reddit.py`
- Polls `/r/{sub}/new.json?limit=100` for `SUBREDDITS = ("CryptoCurrency", "Bitcoin", "ethereum", "solana", "CryptoMarkets")`, unauthenticated, descriptive User-Agent.
- Maps posts to `sentiment_posts` rows: `post_id = "reddit:{fullname}"`, `source='reddit'`, `text = title + "\n" + selftext`, `author`, `likes=score`, `comments→replies`, `published_at = created_utc`, `url`. Tickers via `extract_tickers`.
- Noise filter: skip authors matching known bot-name patterns (e.g. `AutoModerator`, `*_bot*` heuristics) and posts with no text content after trimming.
- Dedup is structural: `post_id` + ReplacingMergeTree. Exponential backoff on HTTP 429/errors (blueprint guardrail #3).

### `rss.py`
- `feedparser` over `FEEDS = (CoinTelegraph, CoinDesk, Decrypt)` (module-level tuple of URLs).
- Items → `sentiment_posts`: `post_id = "news:{guid or link}"`, `source='news'`, `text = title + "\n" + summary`, `author = feed name`, engagement fields 0, `published_at = published_parsed` (fallback: `ingested_at`), `url = link`.

### `scorer.py`
- `score_pending_posts(limit: int = 256) -> int` — selects posts where `sentiment_score IS NULL` (oldest first), embeds texts via Ollama `POST /api/embed` (batched input).
- Anchors: `_POSITIVE_PHRASES` / `_NEGATIVE_PHRASES` module constants (bullish/bearish crypto language, ~8 phrases each); pole vectors = mean of their embeddings, computed once per `score_pending_posts` call.
- Per post: upsert vector into Qdrant `social_posts` (point ID `post_id_to_uuid(post_id)`, payload: `post_id`, `source`, `tickers`, `published_at` unix, `score`); collect `(source, post_id, score)` and apply one batched `ALTER TABLE sentiment_posts UPDATE sentiment_score = … WHERE (source, post_id) IN …` mutation; wait for the mutation to complete (poll `system.mutations`, bounded) before returning the count scored.

### `aggregator.py`
- `compute_metrics(bucket_size: str, bucket_start: datetime) -> None` — for the given aligned bucket and every ticker mentioned: `post_count`, `mean_score`, `weighted_score` (weight = `1 + likes + retweets + replies`; falls back to plain mean when all weights are 1), `engagement_total`, `velocity` and `engagement_ratio`. Baselines are scale-consistent — each bucket is compared against the trailing mean of **same-size** buckets for that ticker: trailing 24h for `5m` (288 buckets) and `1h` (24 buckets), trailing 7 days for `24h` (7 buckets). Ratio is 0 when the baseline is 0. `computed_at = now()`.
- `run_aggregation(now: datetime) -> None` — computes the just-closed bucket for each of `5m`, `1h`, `24h` (24h only at day boundaries). Bucket alignment: `floor(ts, bucket)`. Only posts with non-NULL `sentiment_score` count toward score metrics; all posts count toward volume metrics.

### `pipeline.py`
- CLI: `python -m src.ingestion.pipeline --once` runs one cycle (reddit → rss → score → aggregate) and exits; `--interval N` loops with N seconds between cycles (default 300). Each stage logs a one-line summary; a stage failure logs and continues to the next stage (isolated failure domains).

## Configuration

- Subreddits, feeds, ticker aliases: module-level constants (no config file this phase).
- `OLLAMA_HOST` (already in `.env`) for the embed endpoint; embed model name constant `nomic-embed-text`.

## Error handling

- All external HTTP (Reddit, RSS, Ollama): `httpx` with timeouts and exponential backoff on 429/5xx; a source that keeps failing is skipped for the cycle, never crashes the pipeline.
- Every stage is idempotent and safe to re-run (structural dedup for posts; replace-recompute for metrics; scorer only touches NULL-score posts).
- `published_at` is always the item's own timestamp — never ingestion time — so backtests can't see future data (guardrail #2). RSS items missing a date fall back to `ingested_at` and are noted in the module docstring.

## Testing

- Unit: ticker extraction (cashtags, aliases, dedup, false positives like `$1` amounts), scoring math (hand-built vectors), bucket alignment, velocity/ratio math with known baselines.
- Integration (live ClickHouse, seeded fixtures): Reddit/RSS mappers via `httpx.MockTransport` recorded responses — assert correct rows, assert re-run inserts no duplicates; scorer with mocked Ollama transport — assert scores land via mutation and Qdrant points exist; aggregator on seeded posts — assert exact metric values for a known bucket.
- End-to-end: `pipeline --once` with mocked external HTTP and a seeded backlog — full cycle completes, metrics rows appear.
- New dependencies: `httpx`, `feedparser`.

## Known limitations (documented, not solved this phase)

- Reddit listings and RSS feeds only expose *recent* items (~last 1000 posts / latest feed entries). Historical 2022–present depth for Phase 3 backtesting requires the dataset route, a separate work item.
- X/Twitter ingestion is deferred until the account has API credits; the stage will slot into the pipeline alongside `reddit.py`/`rss.py` writing the same `sentiment_posts` schema.
- Unauthenticated Reddit access is rate-limited; conservative pacing (one request per subreddit per cycle, 300s default interval) keeps us well under limits.
