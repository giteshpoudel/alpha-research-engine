"""Shared insert path for sentiment_posts rows."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name

SENTIMENT_POST_COLUMNS = (
    "post_id", "source", "author", "text", "tickers", "lang",
    "likes", "retweets", "replies", "url",
    "published_at", "ingested_at", "sentiment_score", "label",
)


def insert_posts(ch_client, rows: list[dict]) -> int:
    """Insert row dicts into sentiment_posts. Returns rows inserted.

    Dedup is structural: the table's ReplacingMergeTree collapses repeated
    (source, post_id) keys on merge, so re-inserting the same post is safe.
    """
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [
        [
            row["post_id"],
            row["source"],
            row.get("author", ""),
            row["text"],
            row.get("tickers", []),
            row.get("lang", ""),
            row.get("likes", 0),
            row.get("retweets", 0),
            row.get("replies", 0),
            row.get("url", ""),
            row["published_at"] or now,  # documented fallback: dateless RSS items
            now,
            None,
            row.get("label"),
        ]
        for row in rows
    ]
    ch_client.insert(
        f"{database_name()}.sentiment_posts",
        data,
        column_names=list(SENTIMENT_POST_COLUMNS),
    )
    return len(rows)
