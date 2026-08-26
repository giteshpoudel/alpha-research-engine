"""Reddit poller: /r/{sub}/new.json -> sentiment_posts (source='reddit').

Unauthenticated public JSON endpoints; one request per subreddit per cycle
keeps us well under rate limits. Noise filter: known bot authors and posts
with no text content.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from src.ingestion.http import get_with_backoff
from src.ingestion.posts import insert_posts
from src.ingestion.tickers import extract_tickers

SUBREDDITS = ("CryptoCurrency", "Bitcoin", "ethereum", "solana", "CryptoMarkets")

_BOT_AUTHOR_RE = re.compile(r"(^automoderator$|_bot$|^bot_|bot$)", re.IGNORECASE)


def map_post(post: dict) -> dict | None:
    """Map a Reddit listing child's `data` to a row dict, or None to skip."""
    author = post.get("author") or ""
    if _BOT_AUTHOR_RE.search(author):
        return None
    title = (post.get("title") or "").strip()
    body = (post.get("selftext") or "").strip()
    text = f"{title}\n{body}".strip()
    if not text:
        return None
    return {
        "post_id": f"reddit:{post['name']}",  # fullname, e.g. t3_abc123
        "source": "reddit",
        "author": author,
        "text": text,
        "tickers": extract_tickers(text),
        "lang": "en",
        "likes": max(int(post.get("score") or 0), 0),
        "retweets": 0,
        "replies": max(int(post.get("num_comments") or 0), 0),
        "url": "https://www.reddit.com" + (post.get("permalink") or ""),
        "published_at": datetime.fromtimestamp(float(post["created_utc"]), tz=timezone.utc),
    }


def fetch_new_posts(http_client: httpx.Client, subreddit: str, limit: int = 100) -> list[dict]:
    resp = get_with_backoff(
        f"https://www.reddit.com/r/{subreddit}/new.json",
        client=http_client,
        params={"limit": str(limit), "raw_json": "1"},
    )
    children = resp.json().get("data", {}).get("children", [])
    rows = []
    for child in children:
        row = map_post(child.get("data", {}))
        if row is not None:
            rows.append(row)
    return rows


def poll_subreddits(ch_client, http_client: httpx.Client | None = None,
                    subreddits: tuple[str, ...] = SUBREDDITS) -> int:
    """Fetch the newest posts from each subreddit and insert them.

    A subreddit that fails (after backoff) is skipped, never fatal.
    """
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for sub in subreddits:
            try:
                inserted += insert_posts(ch_client, fetch_new_posts(http_client, sub))
            except RuntimeError as exc:
                print(f"reddit: skipping r/{sub}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted
