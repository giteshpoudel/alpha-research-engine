"""Hacker News historical backfill (Algolia API) -> sentiment_posts (source='hackernews').

Free, fast, and archived back to 2006, with real engagement fields
(points -> likes, num_comments -> replies). Unlike GDELT it is not tightly
rate-limited, so a multi-year backfill takes seconds to minutes.

Usage:
    python -m src.ingestion.hackernews [--start 2024-01-01] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES, extract_tickers

ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_LOOKBACK_DAYS = 365
_HITS_PER_PAGE = 1000
_TAG_RE = re.compile(r"<[^>]+>")


def query_terms() -> dict[str, str]:
    """One search term per ticker: the most descriptive alias."""
    return {ticker: max(aliases, key=len) for ticker, aliases in TICKER_ALIASES.items()}


def parse_created_at(hit: dict) -> datetime | None:
    value = hit.get("created_at")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    epoch = hit.get("created_at_i")
    if isinstance(epoch, (int, float)):
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    return None


def map_hit(hit: dict, source: str = "hackernews") -> dict | None:
    """Map an Algolia HN story hit to a sentiment_posts row, or None to skip."""
    object_id = str(hit.get("objectID") or "").strip()
    title = (hit.get("title") or "").strip()
    published_at = parse_created_at(hit)
    if not object_id or not title or published_at is None:
        return None
    story_text = " ".join(_TAG_RE.sub(" ", hit.get("story_text") or "").split())
    text = f"{title}\n{story_text}".strip() if story_text else title
    return {
        "post_id": f"{source}:{object_id}",
        "source": source,
        "author": (hit.get("author") or "").strip(),
        "text": text,
        "tickers": extract_tickers(text),
        "lang": "en",
        "likes": int(hit.get("points") or 0),
        "retweets": 0,
        "replies": int(hit.get("num_comments") or 0),
        "url": hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
        "published_at": published_at,
    }


def fetch_stories(http_client: httpx.Client, term: str, start: datetime, end: datetime,
                  sleep: float = 0.1, max_pages: int = 10) -> list[dict]:
    """All story hits for ``term`` in [start, end), newest first, paginated."""
    start_i = int(start.timestamp())
    end_i = int(end.timestamp())
    hits: list[dict] = []
    page = 0
    while page < max_pages:
        resp = get_with_backoff(ALGOLIA_URL, client=http_client, params={
            "query": term, "tags": "story", "hitsPerPage": _HITS_PER_PAGE, "page": page,
            "numericFilters": f"created_at_i>{start_i},created_at_i<{end_i}",
        })
        payload = resp.json()
        batch = payload.get("hits", []) if isinstance(payload, dict) else []
        hits.extend(batch)
        nb_pages = int(payload.get("nbPages", 0)) if isinstance(payload, dict) else 0
        page += 1
        if page >= nb_pages or not batch:
            break
        time.sleep(sleep)
    return hits


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


def backfill_hackernews(ch_client, http_client: httpx.Client | None = None,
                        start: datetime | None = None, end: datetime | None = None,
                        sleep: float = 0.1, source: str = "hackernews",
                        terms: dict[str, str] | None = None) -> int:
    """Backfill per-ticker HN stories into sentiment_posts. Resumable via max ts."""
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    resume = _max_ts(ch_client, source)
    if resume is not None:
        start = max(start, resume + timedelta(seconds=1))
    terms = terms or query_terms()

    print(f"hackernews: backfilling {start.date()} -> {end.date()} "
          f"({len(terms)} tickers)")
    inserted = 0
    try:
        for ticker, term in terms.items():
            try:
                hits = fetch_stories(http_client, term, start, end, sleep=sleep)
            except Exception as exc:
                print(f"hackernews: {ticker} ({term}) failed: {exc}")
                continue
            rows = [r for r in (map_hit(h, source) for h in hits) if r and r["tickers"]]
            inserted += insert_posts(ch_client, rows)
            print(f"hackernews: {ticker} ({term}) <- {len(rows)} posts")
    finally:
        if owns_client:
            http_client.close()
    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill Hacker News crypto stories")
    parser.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args(argv)
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else None)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else None)
    from src.ingestion.schemas import get_clickhouse_client
    inserted = backfill_hackernews(get_clickhouse_client(), start=start, end=end,
                                   sleep=args.sleep)
    print(f"hackernews: {inserted} posts inserted")


if __name__ == "__main__":
    main()
