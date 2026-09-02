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
