import json
from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.schemas import (
    QDRANT_COLLECTION,
    database_name,
    get_clickhouse_client,
    get_qdrant_client,
    post_id_to_uuid,
)
from src.ingestion.scorer import (
    fetch_unscored,
    sentiment_score,
    score_pending_posts,
)


def test_sentiment_score_pure_math():
    pos = [1.0, 0.0]
    neg = [-1.0, 0.0]
    assert sentiment_score([1.0, 0.0], pos, neg) == pytest.approx(1.0)
    assert sentiment_score([-1.0, 0.0], pos, neg) == pytest.approx(-1.0)
    # orthogonal text scores 0
    assert sentiment_score([0.0, 1.0], pos, neg) == pytest.approx(0.0)
    # never exceeds [-1, 1] even when both cosines are extreme
    assert -1.0 <= sentiment_score([0.9, 0.1], pos, neg) <= 1.0


def _fake_ollama_client(vectors_by_text: dict[str, list[float]]):
    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [vectors_by_text.get(t, [0.0] * 768) for t in texts]
        return httpx.Response(200, json={"embeddings": embeddings})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def clients():
    ch = get_clickhouse_client()
    qd = get_qdrant_client()
    yield ch, qd
    ch.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')"
    )
    qd.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=["00000000-0000-0000-0000-000000000000"],  # no-op placeholder
        wait=True,
    )


def _seed_unscored(ch, post_id, text, published_at):
    ch.insert(
        f"{database_name()}.sentiment_posts",
        [[post_id, "reddit", "tester", text, ["BTC"], "en",
          10, 0, 3, "https://example.com", published_at,
          datetime.now(timezone.utc), None]],
        column_names=["post_id", "source", "author", "text", "tickers", "lang",
                      "likes", "retweets", "replies", "url",
                      "published_at", "ingested_at", "sentiment_score"],
    )


def test_fetch_unscored_ignores_scored(clients):
    ch, _ = clients
    _seed_unscored(ch, "test:unscored-1", "seed text", datetime(2023, 11, 14, tzinfo=timezone.utc))
    rows = fetch_unscored(ch, limit=1000)
    ids = [r[0] for r in rows]
    assert "test:unscored-1" in ids


def test_score_pending_posts_end_to_end(clients):
    ch, qd = clients
    _seed_unscored(ch, "test:score-me", "bullish bitcoin moon", datetime(2023, 11, 14, tzinfo=timezone.utc))

    bull = [1.0] + [0.0] * 767
    bear = [-1.0] + [0.0] * 767
    # Anchor phrases and the post text all get the bullish vector -> score +1
    import src.ingestion.scorer as scorer_mod
    vectors = {t: bull for t in scorer_mod.POSITIVE_PHRASES}
    vectors.update({t: bear for t in scorer_mod.NEGATIVE_PHRASES})
    vectors["bullish bitcoin moon"] = bull
    http_client = _fake_ollama_client(vectors)

    scored = score_pending_posts(ch, qd, http_client=http_client, limit=1000)
    assert scored >= 1

    rows = ch.query(
        f"SELECT sentiment_score FROM {database_name()}.sentiment_posts FINAL "
        "WHERE post_id = 'test:score-me'"
    ).result_rows
    assert rows[0][0] == pytest.approx(1.0, abs=1e-5)

    points = qd.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[post_id_to_uuid("test:score-me")],
        with_payload=True,
    )
    assert len(points) == 1
    assert points[0].payload["post_id"] == "test:score-me"
    assert points[0].payload["tickers"] == ["BTC"]
    assert points[0].payload["score"] == pytest.approx(1.0, abs=1e-5)
    qd.delete(collection_name=QDRANT_COLLECTION,
              points_selector=[post_id_to_uuid("test:score-me")], wait=True)
