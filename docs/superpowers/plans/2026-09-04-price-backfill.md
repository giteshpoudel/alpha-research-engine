# Price-Data Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill `ohlcv` (Binance.US spot for 11 tickers + Coinbase MATIC/POL) and `funding_rates` (Hyperliquid, 12 coins) from 2022-01-01 / HL listing dates, via a resumable CLI that doubles as the ongoing top-up job.

**Architecture:** `market_data.py` provides the shared insert path and `max_ts` resume cursor; three source modules (`binance_us.py`, `coinbase.py`, `hyperliquid.py`) each expose a mapper, a paginated fetcher, and a `backfill_*` function; `backfill.py` orchestrates with per-source failure isolation.

**Tech Stack:** Python 3.11, `httpx`, `clickhouse-connect`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-04-price-backfill-design.md`

## Global Constraints

- Idempotent everywhere: resume-from-`max_ts` plus ReplacingMergeTree collapse on `(exchange, symbol, interval, ts)` / `(exchange, symbol, ts)`. Re-running never duplicates.
- Closed candles only: every fetch window ends at the current in-progress bucket (`floor(now, interval)`).
- All HTTP via `get_with_backoff` (GET) or `post_with_backoff` (POST) from `src/ingestion/http.py` — no raw `httpx.get`/`post`.
- Timestamps are exchange-provided open/funding times, stored as UTC datetimes. `max_ts` must normalize ClickHouse's possibly-naive datetimes to tz-aware UTC (naive↔aware comparison is a TypeError).
- `mark_price = 0.0` everywhere (not backfilled — documented gap). `next_funding_ts` = next record's time in the series, `ts + 8h` for the latest.
- Coinbase candle layout is `[time, low, high, open, close, volume]` — low BEFORE high; `quote_volume = 0.0`, `num_trades = 0`.
- MATIC stored as symbol `MATIC` for the whole series: `MATIC-USD` ≤ 2024-09-04, `POL-USD` after (exchange `coinbase`).
- Intervals: `1h` and `1d` only.
- Test hygiene: integration tests use symbol `TEST` (and fake `TESTUSDT`/`TESTUSD` pairs via mocked HTTP), teardown `ALTER TABLE … DELETE WHERE symbol = 'TEST'` on the relevant table. Never touch real rows.
- The existing 55 tests must stay green. No new dependencies.
- Run tests from the repo root with the venv active: `python -m pytest tests/ -v`.

---

### Task 1: Shared market-data insert path (`market_data.py`)

**Files:**
- Create: `src/ingestion/market_data.py`
- Test: `tests/test_market_data.py`

**Interfaces:**
- Consumes: `database_name` from `src/ingestion/schemas.py`.
- Produces (used by Tasks 2–4):
  - `OHLCV_COLUMNS: tuple[str, ...]` — `("exchange", "symbol", "interval", "ts", "open", "high", "low", "close", "volume", "quote_volume", "num_trades")`
  - `FUNDING_COLUMNS: tuple[str, ...]` — `("exchange", "symbol", "ts", "funding_rate", "mark_price", "next_funding_ts")`
  - `insert_ohlcv(ch_client, rows: list[dict]) -> int`
  - `insert_funding(ch_client, rows: list[dict]) -> int`
  - `max_ts(ch_client, table: str, exchange: str, symbol: str, interval: str | None = None) -> datetime | None` — tz-aware UTC or None for empty series

- [ ] **Step 1: Write the failing tests**

`tests/test_market_data.py`:

```python
from datetime import datetime, timezone

import pytest

from src.ingestion.market_data import (
    insert_funding,
    insert_ohlcv,
    max_ts,
)
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    client.command(f"ALTER TABLE {db}.ohlcv DELETE WHERE symbol = 'TEST'")
    client.command(f"ALTER TABLE {db}.funding_rates DELETE WHERE symbol = 'TEST'")


def _ohlcv_row(ts):
    return {
        "exchange": "binance_us", "symbol": "TEST", "interval": "1h", "ts": ts,
        "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
        "volume": 12.5, "quote_volume": 1300.0, "num_trades": 42,
    }


def test_insert_ohlcv_and_max_ts(ch_client):
    t1 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2022, 1, 1, 1, 0, tzinfo=timezone.utc)
    assert insert_ohlcv(ch_client, [_ohlcv_row(t1), _ohlcv_row(t2)]) == 2
    assert insert_ohlcv(ch_client, []) == 0
    latest = max_ts(ch_client, "ohlcv", "binance_us", "TEST", "1h")
    assert latest == t2
    assert latest.tzinfo is not None  # must be tz-aware


def test_max_ts_empty_series_returns_none(ch_client):
    assert max_ts(ch_client, "ohlcv", "binance_us", "TEST", "1h") is None
    assert max_ts(ch_client, "funding_rates", "hyperliquid", "TEST") is None


def test_insert_funding_and_max_ts(ch_client):
    t1 = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2023, 6, 1, 8, 0, tzinfo=timezone.utc)
    rows = [
        {"exchange": "hyperliquid", "symbol": "TEST", "ts": t1,
         "funding_rate": 0.0001, "mark_price": 0.0, "next_funding_ts": t2},
        {"exchange": "hyperliquid", "symbol": "TEST", "ts": t2,
         "funding_rate": -0.0002, "mark_price": 0.0,
         "next_funding_ts": datetime(2023, 6, 1, 16, 0, tzinfo=timezone.utc)},
    ]
    assert insert_funding(ch_client, rows) == 2
    assert max_ts(ch_client, "funding_rates", "hyperliquid", "TEST") == t2


def test_insert_ohlcv_idempotent(ch_client):
    t1 = datetime(2022, 2, 1, 0, 0, tzinfo=timezone.utc)
    insert_ohlcv(ch_client, [_ohlcv_row(t1)])
    insert_ohlcv(ch_client, [_ohlcv_row(t1)])
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_market_data.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.market_data'`.

- [ ] **Step 3: Implement `src/ingestion/market_data.py`**

```python
"""Shared insert path and resume cursor for market-data tables."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name

OHLCV_COLUMNS = (
    "exchange", "symbol", "interval", "ts",
    "open", "high", "low", "close", "volume", "quote_volume", "num_trades",
)

FUNDING_COLUMNS = (
    "exchange", "symbol", "ts", "funding_rate", "mark_price", "next_funding_ts",
)


def insert_ohlcv(ch_client, rows: list[dict]) -> int:
    """Insert ohlcv row dicts. Dedup is structural (ReplacingMergeTree)."""
    if not rows:
        return 0
    data = [
        [r["exchange"], r["symbol"], r["interval"], r["ts"],
         r["open"], r["high"], r["low"], r["close"], r["volume"],
         r["quote_volume"], r["num_trades"]]
        for r in rows
    ]
    ch_client.insert(f"{database_name()}.ohlcv", data, column_names=list(OHLCV_COLUMNS))
    return len(rows)


def insert_funding(ch_client, rows: list[dict]) -> int:
    """Insert funding_rates row dicts. Dedup is structural (ReplacingMergeTree)."""
    if not rows:
        return 0
    data = [
        [r["exchange"], r["symbol"], r["ts"], r["funding_rate"],
         r["mark_price"], r["next_funding_ts"]]
        for r in rows
    ]
    ch_client.insert(f"{database_name()}.funding_rates", data, column_names=list(FUNDING_COLUMNS))
    return len(rows)


def max_ts(ch_client, table: str, exchange: str, symbol: str,
           interval: str | None = None) -> datetime | None:
    """Latest stored ts for a series (tz-aware UTC), or None for an empty series.

    maxOrNull returns NULL for empty sets (plain max returns the type default,
    epoch 0). ClickHouse may return naive datetimes; normalize to aware UTC
    because callers compare against tz-aware resume windows.
    """
    sql = (f"SELECT maxOrNull(ts) FROM {database_name()}.{table} "
           "WHERE exchange = {e:String} AND symbol = {s:String}")
    params = {"e": exchange, "s": symbol}
    if interval is not None:
        sql += " AND interval = {i:String}"
        params["i"] = interval
    rows = ch_client.query(sql, parameters=params).result_rows
    latest = rows[0][0] if rows else None
    if latest is None:
        return None
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_market_data.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/market_data.py tests/test_market_data.py
git commit -m "feat: add market-data insert path with resume cursor"
```

---

### Task 2: Binance.US klines (`binance_us.py`)

**Files:**
- Create: `src/ingestion/binance_us.py`
- Test: `tests/test_binance_us.py`

**Interfaces:**
- Consumes: `http.get_with_backoff`; `market_data.insert_ohlcv`, `market_data.max_ts`; `tickers.TICKER_ALIASES`.
- Produces:
  - `BINANCE_US_SYMBOLS: tuple[str, ...]` — 11 tickers (all of `TICKER_ALIASES` except MATIC)
  - `map_kline(ticker: str, interval: str, k: list) -> dict`
  - `fetch_klines(http_client, ticker, interval, start, end) -> list[dict]` — closed candles in `[start, end)`
  - `backfill_ohlcv(ch_client, http_client=None, symbols=None, intervals=("1h", "1d"), start=None, end=None) -> int` (used by Task 5; `symbols=None` → `BINANCE_US_SYMBOLS`, `start=None` → 2022-01-01 UTC, `end=None` → current in-progress bucket)

- [ ] **Step 1: Write the failing tests**

`tests/test_binance_us.py`:

```python
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.binance_us import backfill_ohlcv, fetch_klines, map_kline
from src.ingestion.schemas import database_name, get_clickhouse_client

# [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, ...]
KLINE = [1640995200000, "46192.43", "46715.27", "46192.43", "46670.23",
         "17.363974", 1640998799999, "806563.32", 543, "12.21", "567219.24", "0"]

T0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_map_kline_fields():
    row = map_kline("BTC", "1h", KLINE)
    assert row["exchange"] == "binance_us"
    assert row["symbol"] == "BTC"
    assert row["interval"] == "1h"
    assert row["ts"] == T0
    assert row["open"] == 46192.43
    assert row["high"] == 46715.27
    assert row["low"] == 46192.43
    assert row["close"] == 46670.23
    assert row["volume"] == 17.363974
    assert row["quote_volume"] == 806563.32
    assert row["num_trades"] == 543


def _kline_at(ms):
    k = list(KLINE)
    k[0] = ms
    return k


def test_fetch_klines_paginates():
    calls = []

    def handler(req):
        calls.append(dict(req.url.params))
        start = int(req.url.params["startTime"])
        if start == 1640995200000:
            # full page of 1000 -> forces a second request
            return httpx.Response(200, json=[_kline_at(start + i * 3600000) for i in range(1000)])
        return httpx.Response(200, json=[_kline_at(start)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    end = datetime(2022, 5, 9, tzinfo=timezone.utc)  # beyond the mocked pages
    rows = fetch_klines(client, "TEST", "1h", T0, end)
    assert len(calls) == 2
    assert int(calls[1]["startTime"]) == 1640995200000 + 1000 * 3600000  # advanced past last candle
    assert rows[0]["ts"] == T0
    assert all(r["symbol"] == "TEST" for r in rows)


def test_fetch_klines_falls_back_to_usd_pair():
    def handler(req):
        if req.url.params["symbol"] == "TESTUSDT":
            return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
        return httpx.Response(200, json=[KLINE])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_klines(client, "TEST", "1h", T0, datetime(2022, 1, 2, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert rows[0]["ts"] == T0


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TEST'")


def test_backfill_inserts_and_resumes(ch_client):
    candles = [_kline_at(1640995200000 + i * 3600000) for i in range(3)]
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=candles)))
    end = datetime(2022, 1, 1, 3, 0, tzinfo=timezone.utc)
    assert backfill_ohlcv(ch_client, http_client=client, symbols=("TEST",),
                          intervals=("1h",), start=T0, end=end) == 3
    # resume: max_ts is candle 2 -> window start would be candle 3 == end -> nothing to do
    assert backfill_ohlcv(ch_client, http_client=client, symbols=("TEST",),
                          intervals=("1h",), start=T0, end=end) == 0
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_binance_us.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.binance_us'`.

- [ ] **Step 3: Implement `src/ingestion/binance_us.py`**

```python
"""Binance.US spot klines backfill (exchange='binance_us').

Uses {T}USDT pairs with a {T}USD fallback. Binance.com is geo-blocked in
the user's location; Binance.US has no futures, so these are spot candles.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.market_data import insert_ohlcv, max_ts
from src.ingestion.tickers import TICKER_ALIASES

BINANCE_US_SYMBOLS: tuple[str, ...] = tuple(t for t in TICKER_ALIASES if t != "MATIC")

_INTERVAL_SECONDS = {"1h": 3600, "1d": 86400}
_KLINES_URL = "https://api.binance.us/api/v3/klines"
_PAGE = 1000


def map_kline(ticker: str, interval: str, k: list) -> dict:
    """Binance kline array -> ohlcv row dict."""
    return {
        "exchange": "binance_us",
        "symbol": ticker,
        "interval": interval,
        "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
        "quote_volume": float(k[7]),
        "num_trades": int(k[8]),
    }


def _fetch_pair(http_client: httpx.Client, pair: str, interval: str,
                start_ms: int, end_ms: int) -> list:
    out = []
    cursor = start_ms
    while cursor < end_ms:
        resp = get_with_backoff(_KLINES_URL, client=http_client, params={
            "symbol": pair, "interval": interval,
            "startTime": str(cursor), "endTime": str(end_ms - 1), "limit": str(_PAGE),
        })
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        cursor = batch[-1][0] + _INTERVAL_SECONDS[interval] * 1000
        if len(batch) < _PAGE:
            break
    return out


def fetch_klines(http_client: httpx.Client, ticker: str, interval: str,
                 start: datetime, end: datetime) -> list[dict]:
    """Closed candles for {ticker}USDT (fallback {ticker}USD) in [start, end)."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    try:
        raw = _fetch_pair(http_client, f"{ticker}USDT", interval, start_ms, end_ms)
    except RuntimeError:
        raw = _fetch_pair(http_client, f"{ticker}USD", interval, start_ms, end_ms)
    return [map_kline(ticker, interval, k) for k in raw]


def _closed_end(now: datetime, secs: int) -> datetime:
    epoch = int(now.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % secs), tz=timezone.utc)


def backfill_ohlcv(ch_client, http_client: httpx.Client | None = None,
                   symbols: tuple[str, ...] | None = None,
                   intervals: tuple[str, ...] = ("1h", "1d"),
                   start: datetime | None = None,
                   end: datetime | None = None) -> int:
    """Backfill (or top up) klines for each (symbol, interval). Resumable via max_ts.

    A failing (symbol, interval) is logged and skipped, never fatal.
    """
    symbols = symbols or BINANCE_US_SYMBOLS
    start = start or datetime(2022, 1, 1, tzinfo=timezone.utc)
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for ticker in symbols:
            for interval in intervals:
                secs = _INTERVAL_SECONDS[interval]
                resume = max_ts(ch_client, "ohlcv", "binance_us", ticker, interval)
                window_start = max(start, resume + timedelta(seconds=secs)) if resume else start
                window_end = end or _closed_end(datetime.now(timezone.utc), secs)
                if window_start >= window_end:
                    continue
                try:
                    rows = fetch_klines(http_client, ticker, interval, window_start, window_end)
                    inserted += insert_ohlcv(ch_client, rows)
                except Exception as exc:
                    print(f"binance_us: skipping {ticker} {interval}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_binance_us.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/binance_us.py tests/test_binance_us.py
git commit -m "feat: add Binance.US klines backfill"
```

---

### Task 3: Coinbase MATIC/POL candles (`coinbase.py`)

**Files:**
- Create: `src/ingestion/coinbase.py`
- Test: `tests/test_coinbase.py`

**Interfaces:**
- Consumes: `http.get_with_backoff`; `market_data.insert_ohlcv`, `market_data.max_ts`.
- Produces:
  - `MATIC_MIGRATION: datetime` — `datetime(2024, 9, 4, tzinfo=timezone.utc)`
  - `map_candle(interval: str, c: list, symbol: str = "MATIC") -> dict`
  - `fetch_candles(http_client, product, interval, start, end, symbol="MATIC") -> list[dict]`
  - `backfill_matic(ch_client, http_client=None, intervals=("1h", "1d"), start=None, end=None, symbol="MATIC") -> int` (used by Task 5; `symbol` exists for test isolation — production always uses the default)

- [ ] **Step 1: Write the failing tests**

`tests/test_coinbase.py`:

```python
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.coinbase import (
    MATIC_MIGRATION,
    backfill_matic,
    fetch_candles,
    map_candle,
)
from src.ingestion.schemas import database_name, get_clickhouse_client

# [time, low, high, open, close, volume] -- Coinbase puts low BEFORE high
CANDLE = [1640995200, 46192.43, 46715.27, 46205.0, 46670.23, 686.07385399]

T0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_map_candle_low_high_order():
    row = map_candle("1h", CANDLE)
    assert row["exchange"] == "coinbase"
    assert row["symbol"] == "MATIC"
    assert row["ts"] == T0
    assert row["low"] == 46192.43
    assert row["high"] == 46715.27
    assert row["open"] == 46205.0
    assert row["close"] == 46670.23
    assert row["volume"] == 686.07385399
    assert row["quote_volume"] == 0.0
    assert row["num_trades"] == 0


def test_fetch_candles_reverses_newest_first():
    batch = [[1640998800, 1, 2, 1.5, 1.8, 10], [1640995200, 1, 2, 1.2, 1.5, 20]]
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=batch)))
    rows = fetch_candles(client, "TEST-USD", "1h", T0,
                         datetime(2022, 1, 1, 2, 0, tzinfo=timezone.utc), symbol="TEST")
    assert [r["ts"] for r in rows] == [T0, datetime(2022, 1, 1, 1, 0, tzinfo=timezone.utc)]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TEST'")


def test_backfill_splits_at_migration(ch_client):
    requested = []

    def handler(req):
        product = req.url.path.split("/products/")[1].split("/")[0]
        requested.append(product)
        return httpx.Response(200, json=[list(CANDLE)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    start = MATIC_MIGRATION - timedelta_days(1)
    end = MATIC_MIGRATION + timedelta_days(1)
    inserted = backfill_matic(ch_client, http_client=client, intervals=("1d",),
                              start=start, end=end, symbol="TEST")
    assert requested == ["MATIC-USD", "POL-USD"]
    assert inserted == 2
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 2


def timedelta_days(n):
    from datetime import timedelta
    return timedelta(days=n)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coinbase.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.coinbase'`.

- [ ] **Step 3: Implement `src/ingestion/coinbase.py`**

```python
"""Coinbase spot candles for the MATIC series (exchange='coinbase').

MATIC-USD history runs through the token migration (2024-09-04), POL-USD
after; both are stored as symbol 'MATIC' for series continuity. Coinbase
candle layout is [time, low, high, open, close, volume] -- low BEFORE high,
no quote volume, no trade count.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.market_data import insert_ohlcv, max_ts

MATIC_MIGRATION = datetime(2024, 9, 4, tzinfo=timezone.utc)

_INTERVAL_SECONDS = {"1h": 3600, "1d": 86400}
_MAX_CANDLES = 300  # Coinbase per-request cap


def map_candle(interval: str, c: list, symbol: str = "MATIC") -> dict:
    """Coinbase candle array -> ohlcv row dict."""
    return {
        "exchange": "coinbase",
        "symbol": symbol,
        "interval": interval,
        "ts": datetime.fromtimestamp(c[0], tz=timezone.utc),
        "open": float(c[3]),
        "high": float(c[2]),
        "low": float(c[1]),
        "close": float(c[4]),
        "volume": float(c[5]),
        "quote_volume": 0.0,
        "num_trades": 0,
    }


def fetch_candles(http_client: httpx.Client, product: str, interval: str,
                  start: datetime, end: datetime, symbol: str = "MATIC") -> list[dict]:
    """Closed candles for a Coinbase product in [start, end), sorted oldest-first."""
    rows: list[dict] = []
    cursor = start
    step = timedelta(seconds=_INTERVAL_SECONDS[interval] * _MAX_CANDLES)
    while cursor < end:
        window_end = min(cursor + step, end)
        resp = get_with_backoff(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            client=http_client,
            params={
                "granularity": str(_INTERVAL_SECONDS[interval]),
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
            },
        )
        batch = resp.json()
        rows.extend(map_candle(interval, c, symbol) for c in batch)
        cursor = window_end
    rows.sort(key=lambda r: r["ts"])
    return rows


def backfill_matic(ch_client, http_client: httpx.Client | None = None,
                   intervals: tuple[str, ...] = ("1h", "1d"),
                   start: datetime | None = None,
                   end: datetime | None = None,
                   symbol: str = "MATIC") -> int:
    """Backfill the MATIC series, splitting each window at MATIC_MIGRATION.

    Resumable via max_ts. A failing segment is logged and skipped, never fatal.
    """
    start = start or datetime(2022, 1, 1, tzinfo=timezone.utc)
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for interval in intervals:
            secs = _INTERVAL_SECONDS[interval]
            resume = max_ts(ch_client, "ohlcv", "coinbase", symbol, interval)
            window_start = max(start, resume + timedelta(seconds=secs)) if resume else start
            if end is None:
                epoch = int(datetime.now(timezone.utc).timestamp())
                window_end = datetime.fromtimestamp(epoch - (epoch % secs), tz=timezone.utc)
            else:
                window_end = end
            if window_start >= window_end:
                continue
            segments = []
            if window_start < MATIC_MIGRATION:
                segments.append(("MATIC-USD", window_start, min(window_end, MATIC_MIGRATION)))
            if window_end > MATIC_MIGRATION:
                segments.append(("POL-USD", max(window_start, MATIC_MIGRATION), window_end))
            for product, seg_start, seg_end in segments:
                if seg_start >= seg_end:
                    continue
                try:
                    rows = fetch_candles(http_client, product, interval,
                                         seg_start, seg_end, symbol=symbol)
                    inserted += insert_ohlcv(ch_client, rows)
                except Exception as exc:
                    print(f"coinbase: skipping {product} {interval}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coinbase.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/coinbase.py tests/test_coinbase.py
git commit -m "feat: add Coinbase MATIC/POL candle backfill"
```

---

### Task 4: Hyperliquid funding history (`hyperliquid.py`)

**Files:**
- Create: `src/ingestion/hyperliquid.py`
- Test: `tests/test_hyperliquid.py`

**Interfaces:**
- Consumes: `http.post_with_backoff`; `market_data.insert_funding`, `market_data.max_ts`; `tickers.TICKER_ALIASES`.
- Produces:
  - `map_funding(coin: str, rec: dict, next_ts: datetime) -> dict`
  - `fetch_funding(http_client, coin, start) -> list[dict]` — all records from `start` to now
  - `backfill_funding(ch_client, http_client=None, coins=None, start=None) -> int` (used by Task 5; `coins=None` → all 12, `start=None` → 2023-01-01 UTC)

- [ ] **Step 1: Write the failing tests**

`tests/test_hyperliquid.py`:

```python
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.ingestion.hyperliquid import backfill_funding, fetch_funding, map_funding
from src.ingestion.schemas import database_name, get_clickhouse_client

T0 = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)


def _rec(hours_after_t0, rate="0.0001"):
    ms = int((T0 + timedelta(hours=hours_after_t0)).timestamp() * 1000)
    return {"coin": "TEST", "fundingRate": rate, "premium": "0.00009", "time": ms}


def test_map_funding_fields():
    next_ts = T0 + timedelta(hours=8)
    row = map_funding("TEST", _rec(0), next_ts)
    assert row["exchange"] == "hyperliquid"
    assert row["symbol"] == "TEST"
    assert row["ts"] == T0
    assert row["funding_rate"] == 0.0001
    assert row["mark_price"] == 0.0
    assert row["next_funding_ts"] == next_ts


def test_fetch_funding_pagination_and_next_ts_chain():
    calls = []
    page1 = [_rec(i * 8) for i in range(500)]
    page2 = [_rec(500 * 8), _rec(501 * 8)]

    def handler(req):
        import json as _json
        body = _json.loads(req.content)
        calls.append(body["startTime"])
        batch = page1 if len(calls) == 1 else page2
        start = body["startTime"]
        return httpx.Response(200, json=[r for r in batch if r["time"] >= start])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_funding(client, "TEST", T0)
    assert len(calls) == 2
    assert calls[1] == page1[-1]["time"] + 1
    assert len(rows) == 502
    # next_funding_ts chains to the following record...
    assert rows[0]["next_funding_ts"] == rows[1]["ts"]
    # ...and the last record gets ts + 8h
    assert rows[-1]["next_funding_ts"] == rows[-1]["ts"] + timedelta(hours=8)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.funding_rates DELETE WHERE symbol = 'TEST'")


def test_backfill_funding_inserts_and_resumes(ch_client):
    records = [_rec(0), _rec(8), _rec(16)]

    def handler(req):
        import json as _json
        start = _json.loads(req.content)["startTime"]
        return httpx.Response(200, json=[r for r in records if r["time"] >= start])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert backfill_funding(ch_client, http_client=client, coins=("TEST",), start=T0) == 3
    # resume: max_ts is the last record -> window start passes it -> 0 new
    assert backfill_funding(ch_client, http_client=client, coins=("TEST",), start=T0) == 0
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.funding_rates FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hyperliquid.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.hyperliquid'`.

- [ ] **Step 3: Implement `src/ingestion/hyperliquid.py`**

```python
"""Hyperliquid funding-rate history backfill (exchange='hyperliquid').

POST /info {"type":"fundingHistory"} returns up to 500 records per call,
each {coin, fundingRate, premium, time}. mark_price is not available from
this endpoint and is stored as 0.0 (documented gap). next_funding_ts is
derived from the next record in the series (ts + 8h for the latest).
History starts at each coin's listing; empty responses are normal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import post_with_backoff
from src.ingestion.market_data import insert_funding, max_ts
from src.ingestion.tickers import TICKER_ALIASES

_INFO_URL = "https://api.hyperliquid.xyz/info"
_PAGE = 500


def map_funding(coin: str, rec: dict, next_ts: datetime) -> dict:
    """Hyperliquid funding record -> funding_rates row dict."""
    return {
        "exchange": "hyperliquid",
        "symbol": coin,
        "ts": datetime.fromtimestamp(rec["time"] / 1000, tz=timezone.utc),
        "funding_rate": float(rec["fundingRate"]),
        "mark_price": 0.0,
        "next_funding_ts": next_ts,
    }


def fetch_funding(http_client: httpx.Client, coin: str, start: datetime) -> list[dict]:
    """All funding records for coin from `start` to now, oldest-first."""
    records: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    while True:
        resp = post_with_backoff(
            _INFO_URL,
            json={"type": "fundingHistory", "coin": coin, "startTime": cursor},
            client=http_client,
            timeout=30.0,
        )
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        if len(batch) < _PAGE:
            break
        cursor = batch[-1]["time"] + 1
    rows = []
    for i, rec in enumerate(records):
        if i + 1 < len(records):
            next_ts = datetime.fromtimestamp(records[i + 1]["time"] / 1000, tz=timezone.utc)
        else:
            next_ts = datetime.fromtimestamp(rec["time"] / 1000, tz=timezone.utc) + timedelta(hours=8)
        rows.append(map_funding(coin, rec, next_ts))
    return rows


def backfill_funding(ch_client, http_client: httpx.Client | None = None,
                     coins: tuple[str, ...] | None = None,
                     start: datetime | None = None) -> int:
    """Backfill (or top up) funding history per coin. Resumable via max_ts.

    A failing coin is logged and skipped, never fatal.
    """
    coins = coins or tuple(TICKER_ALIASES)
    start = start or datetime(2023, 1, 1, tzinfo=timezone.utc)
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for coin in coins:
            resume = max_ts(ch_client, "funding_rates", "hyperliquid", coin)
            window_start = max(start, resume + timedelta(milliseconds=1)) if resume else start
            try:
                rows = fetch_funding(http_client, coin, window_start)
                inserted += insert_funding(ch_client, rows)
            except Exception as exc:
                print(f"hyperliquid: skipping {coin}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hyperliquid.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/hyperliquid.py tests/test_hyperliquid.py
git commit -m "feat: add Hyperliquid funding-rate backfill"
```

---

### Task 5: Backfill orchestrator CLI (`backfill.py`)

**Files:**
- Create: `src/ingestion/backfill.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `binance_us.backfill_ohlcv(ch_client, http_client=None, symbols=None, intervals=("1h","1d"), start=None, end=None)`, `coinbase.backfill_matic(ch_client, http_client=None, intervals=("1h","1d"), start=None, end=None, symbol="MATIC")`, `hyperliquid.backfill_funding(ch_client, http_client=None, coins=None, start=None)`, `schemas.get_clickhouse_client`.
- Produces:
  - `run_backfill(ch_client, symbols=None, intervals=("1h", "1d"), start=None) -> dict[str, int]` — job names `"binance_us"`, `"coinbase_matic"`, `"hyperliquid_funding"`; `-1` on failure
  - `main(argv: list[str] | None = None) -> None` — CLI: `--symbols TICKER…`, `--intervals {1h,1d}…`, `--start YYYY-MM-DD`
  - Runnable as `python -m src.ingestion.backfill`

- [ ] **Step 1: Write the failing tests**

`tests/test_backfill.py`:

```python
from datetime import datetime, timezone

import pytest

import src.ingestion.backfill as backfill


@pytest.fixture
def jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "backfill_ohlcv",
                        lambda ch, symbols=None, intervals=("1h", "1d"), start=None:
                        calls.append(("binance_us", symbols)) or 100)
    monkeypatch.setattr(backfill, "backfill_matic",
                        lambda ch, intervals=("1h", "1d"), start=None:
                        calls.append(("coinbase_matic", None)) or 50)
    monkeypatch.setattr(backfill, "backfill_funding",
                        lambda ch, coins=None, start=None:
                        calls.append(("hyperliquid_funding", coins)) or 200)
    return calls


def test_run_backfill_all_sources(jobs):
    stats = backfill.run_backfill(object())
    assert [c[0] for c in jobs] == ["binance_us", "coinbase_matic", "hyperliquid_funding"]
    assert stats == {"binance_us": 100, "coinbase_matic": 50, "hyperliquid_funding": 200}
    assert jobs[0][1] is None  # symbols=None -> all 11 binance tickers
    assert jobs[2][1] is None  # coins=None -> all 12


def test_run_backfill_symbol_filter(jobs):
    backfill.run_backfill(object(), symbols=("BTC",))
    assert [c[0] for c in jobs] == ["binance_us", "hyperliquid_funding"]  # no MATIC -> no coinbase job
    assert jobs[0][1] == ("BTC",)
    assert jobs[1][1] == ("BTC",)


def test_run_backfill_source_failure_isolated(monkeypatch, jobs):
    monkeypatch.setattr(backfill, "backfill_matic",
                        lambda ch, intervals=("1h", "1d"), start=None: (_ for _ in ()).throw(RuntimeError("coinbase down")))
    stats = backfill.run_backfill(object())
    assert stats["coinbase_matic"] == -1
    assert stats["hyperliquid_funding"] == 200


def test_main_parses_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(backfill, "run_backfill",
                        lambda ch, symbols=None, intervals=("1h", "1d"), start=None:
                        seen.update(symbols=symbols, intervals=intervals, start=start))
    monkeypatch.setattr(backfill, "get_clickhouse_client", lambda: object())
    backfill.main(["--symbols", "BTC", "ETH", "--intervals", "1h", "--start", "2024-01-01"])
    assert seen["symbols"] == ["BTC", "ETH"]
    assert seen["intervals"] == ("1h",)
    assert seen["start"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backfill.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.ingestion.backfill'`.

- [ ] **Step 3: Implement `src/ingestion/backfill.py`**

```python
"""Backfill orchestrator: OHLCV (binance_us + coinbase MATIC) + funding (hyperliquid).

Usage:
    python -m src.ingestion.backfill                        # everything from 2022-01-01
    python -m src.ingestion.backfill --symbols BTC ETH      # restrict tickers
    python -m src.ingestion.backfill --intervals 1h         # restrict ohlcv intervals
    python -m src.ingestion.backfill --start 2024-01-01     # floor for fresh series

Every series resumes from its stored max_ts, so this is also the ongoing
top-up job: run it any time to bring all series up to the latest closed
candle. A failing source is logged and skipped; the others still run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.ingestion.binance_us import backfill_ohlcv
from src.ingestion.coinbase import backfill_matic
from src.ingestion.hyperliquid import backfill_funding
from src.ingestion.schemas import get_clickhouse_client


def run_backfill(ch_client, symbols: list[str] | tuple[str, ...] | None = None,
                 intervals: tuple[str, ...] = ("1h", "1d"),
                 start: datetime | None = None) -> dict[str, int]:
    start = start or datetime(2022, 1, 1, tzinfo=timezone.utc)
    binance_symbols = tuple(s for s in symbols if s != "MATIC") if symbols else None
    want_matic = symbols is None or "MATIC" in symbols
    funding_coins = tuple(symbols) if symbols else None

    jobs: list[tuple[str, object]] = []
    if symbols is None or binance_symbols:
        jobs.append(("binance_us", lambda: backfill_ohlcv(
            ch_client, symbols=binance_symbols, intervals=intervals, start=start)))
    if want_matic:
        jobs.append(("coinbase_matic", lambda: backfill_matic(
            ch_client, intervals=intervals, start=start)))
    jobs.append(("hyperliquid_funding", lambda: backfill_funding(
        ch_client, coins=funding_coins, start=start)))

    stats: dict[str, int] = {}
    for name, fn in jobs:
        try:
            stats[name] = fn()
            print(f"backfill: {name} done ({stats[name]})")
        except Exception as exc:
            stats[name] = -1
            print(f"backfill: {name} failed: {exc}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV + funding-rate history")
    parser.add_argument("--symbols", nargs="+", metavar="TICKER", default=None,
                        help="restrict to these tickers (default: all 12)")
    parser.add_argument("--intervals", nargs="+", choices=["1h", "1d"], default=["1h", "1d"],
                        help="ohlcv intervals (default: 1h 1d)")
    parser.add_argument("--start", default="2022-01-01", metavar="YYYY-MM-DD",
                        help="floor date for fresh series (default 2022-01-01)")
    args = parser.parse_args(argv)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    run_backfill(get_clickhouse_client(), symbols=args.symbols,
                 intervals=tuple(args.intervals), start=start)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all 73 tests PASS (55 existing + 4 market_data + 4 binance_us + 3 coinbase + 3 hyperliquid + 4 backfill).

- [ ] **Step 5: Live smoke run (small real window)**

With the Docker stack running:

Run: `python -m src.ingestion.backfill --symbols BTC --intervals 1h --start 2026-08-25`
Expected: `binance_us done (N)` with N ≈ 240 (10 days × 24), `hyperliquid_funding done (M)` with M ≈ 30 (10 days × 3/day), `coinbase_matic` absent (BTC filter), exit code 0.

Then verify:

Run: `curl -s --user "alpha:" "http://localhost:8123/?query=SELECT%20exchange%2C%20symbol%2C%20interval%2C%20count()%2C%20min(ts)%2C%20max(ts)%20FROM%20alpha.ohlcv%20FINAL%20WHERE%20symbol%3D%27BTC%27%20GROUP%20BY%20exchange%2C%20symbol%2C%20interval%20FORMAT%20PrettyCompact"`
Expected: one `binance_us BTC 1h` row with count ≈ 240 and max(ts) within the last day.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/backfill.py tests/test_backfill.py
git commit -m "feat: add backfill orchestrator CLI"
```

---

## Self-Review Notes

- **Spec coverage:** `market_data.py` incl. `max_ts` (Task 1), Binance.US 11 tickers with USD-pair fallback and closed-candle windowing (Task 2), Coinbase MATIC/POL with migration split and low/high swap (Task 3), Hyperliquid funding with pagination/next_funding_ts/mark_price=0.0 (Task 4), CLI with symbol/interval/start filters and per-source isolation (Task 5), resume/top-up semantics (Tasks 2–4 via `max_ts`), tests for all of the above, no new dependencies. All spec sections covered.
- **Type consistency:** `insert_ohlcv(ch_client, list[dict]) -> int`, `insert_funding(ch_client, list[dict]) -> int`, `max_ts(ch_client, str, str, str, str|None) -> datetime|None`, `map_kline(str, str, list) -> dict`, `fetch_klines(client, str, str, datetime, datetime) -> list[dict]`, `backfill_ohlcv(ch_client, http_client=None, symbols=None, intervals=("1h","1d"), start=None, end=None) -> int`, `map_candle(str, list, symbol="MATIC") -> dict`, `fetch_candles(client, str, str, datetime, datetime, symbol="MATIC") -> list[dict]`, `backfill_matic(ch_client, http_client=None, intervals=("1h","1d"), start=None, end=None, symbol="MATIC") -> int`, `map_funding(str, dict, datetime) -> dict`, `fetch_funding(client, str, datetime) -> list[dict]`, `backfill_funding(ch_client, http_client=None, coins=None, start=None) -> int`, `run_backfill(ch_client, symbols=None, intervals=("1h","1d"), start=None) -> dict[str,int]`, `main(argv=None) -> None` — spelled identically in interfaces, code, and tests.
- **Test count:** 55 + 4 + 4 + 3 + 3 + 4 = 73.
- **Ordering dependency:** Tasks 2–4 consume Task 1's `market_data.py`; Task 5 consumes Tasks 2–4. Sequential execution required.
- **Full historical run is NOT a plan step:** after merge, run `python -m src.ingestion.backfill` (all symbols, from 2022) as a one-time data operation (~700 requests, minutes).
