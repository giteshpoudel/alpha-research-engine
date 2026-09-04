# Price-Data Backfill Design — OHLCV + Funding Rates, 2022–Present

Date: 2026-09-04
Scope: Backfill the Phase 1 `ohlcv` and `funding_rates` ClickHouse tables for the 12 pipeline tickers from 2022-01-01 (OHLCV) / each coin's Hyperliquid listing (funding), plus a resumable top-up mode. Historical social data and scheduled pipeline operation are out of scope.

## Decisions (verified against live APIs 2026-09-03/04)

- **Binance.com is geo-blocked** from the user's location (HTTP 451-style `restricted location` error). Binance.US works for spot klines but has no futures.
- **OHLCV sources:** Binance.US spot klines for 11 tickers (`{T}USDT`, exchange `binance_us`) + Coinbase exchange candles for MATIC (`MATIC-USD` ≤ 2024-09-04 migration, `POL-USD` after, stored as symbol `MATIC`, exchange `coinbase`).
- **Funding source:** Hyperliquid `POST /info {"type":"fundingHistory"}` for all 12 (exchange `hyperliquid`). History starts at each coin's HL listing (BTC ≈ 2023-06); earlier funding data is unavailable from reachable free sources — documented limitation.
- **Hyperliquid candles are NOT used:** its public candleSnapshot only serves ~6 months of history.
- **Intervals:** `1h` + `1d` only. Coarser aggregations happen in queries; 5m deferred until a strategy needs it.
- **`mark_price` = 0.0** — HL funding history carries `premium`, not mark price; documented as not-backfilled.
- **`next_funding_ts`** = next record's timestamp within each coin's series; latest record gets `ts + 8h`.
- **Closed candles only** — fetch windows end at the current in-progress bucket.

## Data volume & pacing

~41k hourly candles per major symbol → Binance.US (1000/request) ≈ 43 requests per symbol × 11; Coinbase (300/request) ≈ 140 requests for MATIC+POL; HL funding (500/request) ≈ 8 per coin × 12. Total ≈ 700 requests via existing `get_with_backoff` (User-Agent, timeout, exponential backoff on 429/5xx/transport — blueprint guardrail #3). Runtime: minutes, not hours.

## New modules in `src/ingestion/`

### `market_data.py`
- `OHLCV_COLUMNS`, `FUNDING_COLUMNS` tuples.
- `insert_ohlcv(ch_client, rows: list[dict]) -> int` and `insert_funding(ch_client, rows: list[dict]) -> int` — same role as `posts.insert_posts`. Row dict keys: ohlcv → `exchange, symbol, interval, ts, open, high, low, close, volume, quote_volume, num_trades`; funding → `exchange, symbol, ts, funding_rate, mark_price, next_funding_ts`.
- `max_ts(ch_client, table, exchange, symbol, interval=None) -> datetime | None` — resume cursor.

### `binance_us.py`
- `BINANCE_US_SYMBOLS: tuple[str, ...]` — the 11 non-MATIC tickers.
- `map_kline(ticker: str, interval: str, k: list) -> dict` — Binance kline array `[openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, …]` → row dict (`exchange='binance_us'`, `symbol=ticker`, `ts=openTime` ms→datetime UTC).
- `fetch_klines(http_client, ticker, interval, start_ms, end_ms) -> list[dict]` — `GET /api/v3/klines?symbol={T}USDT&interval=&startTime=&endTime=&limit=1000`, paginated by advancing `startTime` past the last candle's openTime. Falls back to `{T}USD` pair if the USDT pair errors.
- `backfill_ohlcv(ch_client, http_client=None, symbols=..., intervals=("1h","1d"), start=datetime(2022,1,1)) -> int`.

### `coinbase.py`
- `MATIC_MIGRATION = datetime(2024, 9, 4, tzinfo=utc)`.
- `map_candle(interval: str, c: list) -> dict` — Coinbase `[time, low, high, open, close, volume]` (note low/high order; `quote_volume=0.0`, `num_trades=0`) → row dict (`exchange='coinbase'`, `symbol='MATIC'`).
- `fetch_candles(http_client, product, interval, start, end) -> list[dict]` — `GET /products/{product}/candles?granularity=&start=&end=` (max 300/request, returned newest-first → reversed).
- `backfill_matic(ch_client, http_client=None, intervals=("1h","1d"), start=datetime(2022,1,1)) -> int` — splits each interval's range at `MATIC_MIGRATION`: `MATIC-USD` before, `POL-USD` after.

### `hyperliquid.py`
- `map_funding(coin: str, rec: dict, next_ts: datetime) -> dict` — `{coin, fundingRate, premium, time}` → row dict (`exchange='hyperliquid'`, `mark_price=0.0`, `next_funding_ts=next_ts`).
- `fetch_funding(http_client, coin, start_ms) -> list[dict]` — POST `https://api.hyperliquid.xyz/info` `{"type":"fundingHistory","coin":…,"startTime":…}`, paginated 500/call by advancing `startTime` past the last record; sets `next_funding_ts` from the following record (`ts + 8h` for the latest).
- `backfill_funding(ch_client, http_client=None, coins=TICKER_ALIASES keys, start=datetime(2023,1,1)) -> int` — skips coins with no history (empty response is normal for late-listed coins).

### `backfill.py`
- CLI: `python -m src.ingestion.backfill [--symbols BTC ETH …] [--intervals 1h 1d] [--start YYYY-MM-DD]`.
- Default run: Binance.US OHLCV (11 tickers), Coinbase MATIC, Hyperliquid funding (12 coins) — each source wrapped so one failure doesn't abort the others (same isolation philosophy as the pipeline).
- Resumable: each (exchange, symbol, interval) resumes from `max_ts` + one bucket; a fresh DB starts at `--start` (default 2022-01-01). Re-running any time tops up to the latest closed candle — the same script is the ongoing update job.

## Error handling

- All HTTP via `get_with_backoff`; a source that fails is logged and skipped, others continue.
- Idempotent everywhere: resume-from-max plus ReplacingMergeTree collapse on `(exchange, symbol, interval, ts)` / `(exchange, symbol, ts)`.
- Timestamps are exchange-provided open/funding times, UTC (guardrail #2 analogue: no lookahead).

## Testing

- Unit: `map_kline` (field order, ms→datetime), `map_candle` (Coinbase low/high swap, zeroed quote_volume/num_trades), MATIC migration split boundary (2024-09-04 in MATIC-USD vs POL-USD), `map_funding` (next_funding_ts derivation incl. last-record +8h).
- Fetchers: `httpx.MockTransport` with recorded fixtures; pagination asserted (multi-page mock, correct startTime advancement); binance `{T}USD` fallback on USDT-pair error.
- Integration (live ClickHouse, symbol `TEST`, teardown `ALTER TABLE … DELETE WHERE symbol = 'TEST'`): end-to-end backfill of a tiny window via mocked HTTP inserts expected rows; re-run inserts no duplicates (FINAL count stable); resume from `max_ts` fetches only newer candles.
- Verification (post-backfill, manual step): row counts per (exchange, symbol, interval); spot-check BTC 1h close at a known timestamp against public reference.
- New dependencies: none.

## Known limitations (documented)

- No funding data before each coin's Hyperliquid listing (BTC ≈ 2023-06, alts later). Strategies needing pre-2023 funding must treat it as missing, not zero.
- `mark_price` is 0.0 everywhere until a premium-index source is integrated.
- Binance.US spot prices (not perp) back OHLCV; funding-rate-arb analysis must account for spot/perp basis across venues.
- MATIC series spans the MATIC→POL token migration; price continuity across 2024-09-04 is approximate (1:1 rebrand, but liquidity/venue changed).
