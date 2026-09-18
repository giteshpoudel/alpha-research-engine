"""GDELT DOC 2.0 historical news backfill -> sentiment_posts (source='gdelt').

GDELT indexes global news from 2017. Its DOC API is aggressively rate-limited
to roughly one request every 5 seconds, so this backfill walks daily windows
with a throttle. Article lists are capped at 250 records per request, so very
high-volume days may truncate (documented limitation).

Usage:
    python -m src.ingestion.gdelt [--start 2024-01-01] [--end YYYY-MM-DD] [--sleep 5]
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES, extract_tickers

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_LOOKBACK_DAYS = 365
_MAX_RECORDS = 250
_MIN_INTERVAL = 5.0


def build_query() -> str:
    """English-language query OR-ing every ticker alias."""
    terms = set()
    for ticker, aliases in TICKER_ALIASES.items():
        terms.add(ticker.lower())
        terms.update(aliases)
    quoted = [f'"{t}"' if " " in t else t for t in sorted(terms)]
    return "(" + " OR ".join(quoted) + ") sourcelang:english"


def parse_seendate(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def map_article(article: dict, source: str = "gdelt") -> dict | None:
    """Map a GDELT article to a sentiment_posts row, or None to skip."""
    url = (article.get("url") or "").strip()
    title = (article.get("title") or "").strip()
    published_at = parse_seendate(article.get("seendate") or "")
    if not url or not title or published_at is None:
        return None
    return {
        "post_id": f"{source}:{url}",
        "source": source,
        "author": (article.get("domain") or "").strip(),
        "text": title,
        "tickers": extract_tickers(title),
        "lang": (article.get("language") or "").strip().lower(),
        "likes": 0,
        "retweets": 0,
        "replies": 0,
        "url": url,
        "published_at": published_at,
    }


def fetch_articles(http_client: httpx.Client, start: datetime, end: datetime,
                   query: str) -> list[dict]:
    """Article list for [start, end) from the GDELT DOC API."""
    resp = get_with_backoff(GDELT_URL, client=http_client, params={
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": str(_MAX_RECORDS), "sort": "dateasc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    })
    try:
        payload = resp.json()
    except ValueError:
        if "limit requests" in resp.text:
            raise RuntimeError("GDELT rate limit hit — increase --sleep")
        return []
    return payload.get("articles", []) if isinstance(payload, dict) else []


def _max_ts(ch_client, source: str) -> datetime | None:
    rows = ch_client.query(
        f"SELECT maxOrNull(published_at) FROM {database_name()}.sentiment_posts FINAL "
        "WHERE source = {s:String}",
        parameters={"s": source},
    ).result_rows
    value = rows[0][0] if rows else None
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def backfill_gdelt(ch_client, http_client: httpx.Client | None = None,
                   start: datetime | None = None, end: datetime | None = None,
                   sleep: float = _MIN_INTERVAL, source: str = "gdelt",
                   query: str | None = None) -> int:
    """Backfill daily GDELT windows into sentiment_posts. Resumable via max ts."""
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    query = query or build_query()
    end = end or (datetime.now(timezone.utc) - timedelta(days=1))
    start = start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    resume = _max_ts(ch_client, source)
    if resume is not None:
        start = max(start, resume + timedelta(seconds=1))
    est_min = max(0, int((end - start).days)) * sleep / 60
    print(f"gdelt: backfilling {start.date()} -> {end.date()} "
          f"(~{int((end - start).days)} requests, ~{est_min:.0f} min at {sleep}s/req)")

    inserted = 0
    windows = 0
    cursor = start
    try:
        while cursor < end:
            window_end = min(cursor + timedelta(days=1), end)
            try:
                articles = fetch_articles(http_client, cursor, window_end, query)
                rows = [r for r in (map_article(a, source) for a in articles) if r]
                inserted += insert_posts(ch_client, rows)
            except Exception as exc:
                print(f"gdelt: {cursor.date()} failed: {exc}")
            windows += 1
            if windows % 30 == 0:
                print(f"gdelt: {cursor.date()} ({inserted} posts so far)")
            cursor = window_end
            if cursor < end:
                time.sleep(sleep)
    finally:
        if owns_client:
            http_client.close()
    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill GDELT crypto news -> sentiment_posts")
    parser.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=_MIN_INTERVAL,
                        help="seconds between GDELT requests (default 5; API limit)")
    args = parser.parse_args(argv)
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else None)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else None)
    from src.ingestion.schemas import get_clickhouse_client
    inserted = backfill_gdelt(get_clickhouse_client(), start=start, end=end, sleep=args.sleep)
    print(f"gdelt: {inserted} posts inserted")


if __name__ == "__main__":
    main()
