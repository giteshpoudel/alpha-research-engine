# Storage Infrastructure Design — Phase 1 (ClickHouse + Qdrant)

Date: 2026-08-25
Scope: Phase 1 of `docs/PROJECT_BLUEPRINT.MD` — containerized storage (ClickHouse, Qdrant) and idempotent database schemas for market time-series (OHLCV, funding rates) and sentiment payloads. Ingestion scripts and the noise-reduction pipeline are out of scope for this spec.

## Decisions

- **Embedding model:** local Ollama `nomic-embed-text` → 768-dim vectors in Qdrant.
- **Schema management:** a Python DDL module applied idempotently (rejected: ClickHouse initdb SQL scripts split across two mechanisms; rejected: Alembic/migration tooling — overkill for 3 tables + 1 collection and a poor fit for ClickHouse).
- **Compose scope:** ClickHouse + Qdrant only. Ollama runs natively on the host (Metal GPU on macOS); Redpanda is optional in the blueprint and not needed in Phase 1.
- **Unified OHLCV table** keyed by `(exchange, symbol, interval)` instead of per-exchange tables — simpler, and trivial for ClickHouse at this scale.

## docker-compose.yml (repo root)

Two services, both with pinned image tags, named volumes, and healthchecks:

- `clickhouse`
  - Image: `clickhouse/clickhouse-server:25.8` (latest stable line; 26.x is newest, 25.8 chosen for stability)
  - Ports: `8123` (HTTP), `9000` (native TCP)
  - Volume: `clickhouse_data:/var/lib/clickhouse`
  - Credentials from `.env`: `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DB=alpha`
  - Healthcheck: `clickhouse-client` ping
- `qdrant`
  - Image: `qdrant/qdrant:v1.19.0` (latest stable patch, verified on Docker Hub 2026-08-25)
  - Ports: `6333` (REST), `6334` (gRPC)
  - Volume: `qdrant_data:/qdrant/storage`
  - Healthcheck: HTTP GET `/healthz` on 6333

## .env.template additions

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

## ClickHouse schemas (database `alpha`)

All tables use `ReplacingMergeTree` so historical re-imports replace existing rows instead of duplicating them (blueprint guardrail #4: idempotent pipelines).

### `ohlcv`

| Column | Type |
| --- | --- |
| exchange | LowCardinality(String) |
| symbol | String |
| interval | LowCardinality(String) |
| ts | DateTime64(3) |
| open | Float64 |
| high | Float64 |
| low | Float64 |
| close | Float64 |
| volume | Float64 |
| quote_volume | Float64 |
| num_trades | UInt32 |

Engine: `ReplacingMergeTree`, `ORDER BY (exchange, symbol, interval, ts)`.

### `funding_rates`

| Column | Type |
| --- | --- |
| exchange | LowCardinality(String) |
| symbol | String |
| ts | DateTime64(3) |
| funding_rate | Float64 |
| mark_price | Float64 |
| next_funding_ts | DateTime64(3) |

Engine: `ReplacingMergeTree`, `ORDER BY (exchange, symbol, ts)`.

### `sentiment_posts`

| Column | Type |
| --- | --- |
| post_id | String |
| source | LowCardinality(String) — `twitter` / `news` |
| author | String |
| text | String |
| tickers | Array(String) |
| lang | LowCardinality(String) |
| likes | UInt32 |
| retweets | UInt32 |
| replies | UInt32 |
| url | String |
| published_at | DateTime64(3) |
| ingested_at | DateTime64(3) |
| sentiment_score | Nullable(Float32) — filled by Phase 2 worker |

Engine: `ReplacingMergeTree`, `ORDER BY (source, post_id)`. `published_at` is the post's publication time so historical backtests never see future social data (guardrail #2).

## Qdrant collection `social_posts`

- Vectors: 768 dimensions, cosine distance (nomic-embed-text output).
- Point ID: `uuid5(post_id)` so re-upserts overwrite instead of duplicating.
- Payload indexes: `post_id` (keyword), `source` (keyword), `tickers` (keyword), `published_at` (integer/unix timestamp for range filters).

## src/ingestion/schemas.py

- `create_clickhouse_schema()` — creates database `alpha` and the three tables with `CREATE ... IF NOT EXISTS`.
- `create_qdrant_schema()` — creates the `social_posts` collection if absent and ensures payload indexes exist.
- CLI entry point: `python -m src.ingestion.schemas` — waits for both containers to be healthy (bounded retry with backoff), applies both schemas, prints a summary.
- New dependencies (`requirements.txt` — none exists yet): `clickhouse-connect`, `qdrant-client`, `python-dotenv`, plus `pytest` as a dev dependency.

## Error handling

- Connection failures during init: retry with exponential backoff, clear error message if containers are unreachable after the retry budget.
- All DDL is idempotent — re-running the module is always safe.
- If the Qdrant collection exists with a mismatched vector size, fail loudly rather than silently recreating (would drop embedded data).

## Testing

- `tests/test_schemas.py` (pytest): against running containers, assert the `alpha` database and all three tables exist with expected columns, and the `social_posts` collection exists with 768-dim cosine vectors and the expected payload indexes.
- Requires `docker compose up -d` first; documented in the test module docstring.
