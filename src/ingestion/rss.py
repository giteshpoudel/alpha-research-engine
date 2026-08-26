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
        except Exception as exc:
            print(f"rss: skipping {name}: {exc}")
    return inserted
