"""Embedding-distance sentiment scorer.

score = clamp(cos(v, pos_anchor) - cos(v, neg_anchor), -1, +1), where anchors
are mean embeddings of the phrase lists below. Deterministic: same text ->
same score. Also upserts each post's vector into Qdrant with its score as
payload, then applies scores to ClickHouse in one batched mutation.
"""

from __future__ import annotations

import math
import os
import time
from datetime import timezone

import httpx
from qdrant_client.models import PointStruct

from src.ingestion.http import post_with_backoff
from src.ingestion.schemas import QDRANT_COLLECTION, database_name, post_id_to_uuid

EMBED_MODEL = "nomic-embed-text"

POSITIVE_PHRASES = (
    "bitcoin is going to the moon, extremely bullish, buy now",
    "huge breakout, massive gains ahead, so bullish",
    "accumulating here, long term hold, undervalued gem",
    "adoption is growing, institutions are buying, bullish news",
    "great recovery, strong fundamentals, green candles",
    "to the moon, rocket emoji, all time high incoming",
    "buy the dip, this is the bottom, reversal incoming",
    "bull market confirmed, prices will keep rising",
)

NEGATIVE_PHRASES = (
    "bitcoin is crashing, extremely bearish, sell everything",
    "huge dump incoming, massive losses, so bearish",
    "this is a scam, rug pull, ponzi scheme, fraud",
    "dead cat bounce, going to zero, capitulation",
    "panic selling, blood in the streets, red candles everywhere",
    "bear market confirmed, prices will keep falling",
    "get out now, top is in, distribute before the crash",
    "fud, hacks, exchange collapse, funds are gone",
)


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "localhost")


def _embed(http_client: httpx.Client, texts: list[str]) -> list[list[float]]:
    resp = post_with_backoff(
        f"http://{_ollama_host()}:11434/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        client=http_client,
        timeout=120.0,
    )
    return resp.json()["embeddings"]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def sentiment_score(vector, pos_anchor, neg_anchor) -> float:
    raw = _cosine(vector, pos_anchor) - _cosine(vector, neg_anchor)
    return max(-1.0, min(1.0, raw))


def _to_unix(dt) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_unscored(ch_client, limit: int = 256) -> list[tuple]:
    """Oldest-first posts with NULL sentiment_score.

    Returns (post_id, source, text, tickers, published_at) tuples.
    """
    return ch_client.query(
        f"SELECT post_id, source, text, tickers, published_at "
        f"FROM {database_name()}.sentiment_posts "
        "WHERE sentiment_score IS NULL "
        "ORDER BY published_at ASC LIMIT {n:UInt32}",
        parameters={"n": limit},
    ).result_rows


def _wait_for_mutations(ch_client, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = ch_client.query(
            "SELECT count() FROM system.mutations "
            "WHERE database = {db:String} AND table = 'sentiment_posts' AND is_done = 0",
            parameters={"db": database_name()},
        ).result_rows
        if rows[0][0] == 0:
            return
        time.sleep(1.0)
    raise RuntimeError("Timed out waiting for ClickHouse mutation to apply")


def _apply_scores(ch_client, scored: list[tuple[str, str, float]]) -> None:
    """One batched mutation: (source, post_id) -> sentiment_score."""
    branches, conditions, params = [], [], {}
    for i, (source, post_id, score) in enumerate(scored):
        conditions.append(f"(source = {{s{i}:String}} AND post_id = {{p{i}:String}})")
        branches.append(f"source = {{s{i}:String}} AND post_id = {{p{i}:String}}, {{v{i}:Float32}}")
        params[f"s{i}"] = source
        params[f"p{i}"] = post_id
        params[f"v{i}"] = score
    case_expr = "multiIf(" + ", ".join(branches) + ", sentiment_score)"
    where = " OR ".join(conditions)
    ch_client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts "
        f"UPDATE sentiment_score = {case_expr} WHERE {where}",
        parameters=params,
    )
    _wait_for_mutations(ch_client)


def score_pending_posts(ch_client, qd_client, http_client: httpx.Client | None = None,
                        limit: int = 256) -> int:
    """Score up to `limit` unscored posts. Returns how many were scored."""
    rows = fetch_unscored(ch_client, limit)
    if not rows:
        return 0
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    try:
        pos_anchor = _mean_vector(_embed(http_client, list(POSITIVE_PHRASES)))
        neg_anchor = _mean_vector(_embed(http_client, list(NEGATIVE_PHRASES)))
        texts = [text[:4000] for (_, _, text, _, _) in rows]
        vectors = _embed(http_client, texts)
        points, scored = [], []
        for (post_id, source, _text, tickers, published_at), vector in zip(rows, vectors):
            score = sentiment_score(vector, pos_anchor, neg_anchor)
            points.append(PointStruct(
                id=post_id_to_uuid(post_id),
                vector=vector,
                payload={
                    "post_id": post_id,
                    "source": source,
                    "tickers": list(tickers),
                    "published_at": _to_unix(published_at),
                    "score": score,
                },
            ))
            scored.append((source, post_id, score))
        qd_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        _apply_scores(ch_client, scored)
        return len(scored)
    finally:
        if owns_client:
            http_client.close()
