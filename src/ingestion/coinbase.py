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
