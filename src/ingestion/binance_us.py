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
