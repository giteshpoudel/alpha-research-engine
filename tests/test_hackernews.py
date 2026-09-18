from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.hackernews import (
    backfill_hackernews,
    fetch_stories,
    map_hit,
    parse_created_at,
    query_terms,
)
from src.ingestion.schemas import database_name, get_clickhouse_client

HIT_BTC = {"objectID": "111", "title": "Bitcoin breaks $100k", "author": "alice",
           "created_at": "2024-01-01T00:00:00Z", "created_at_i": 1704067200,
           "points": 500, "num_comments": 200, "url": "https://ex.com/btc"}
HIT_ETH = {"objectID": "222", "title": "Ask HN: best way to learn Ethereum?",
           "story_text": "<p>ETH question</p>", "author": "bob",
           "created_at": "2024-01-02T00:00:00Z", "points": 10, "num_comments": 5,
           "url": None}
HIT_OTHER = {"objectID": "333", "title": "Ask HN: favorite text editor",
             "author": "carol", "created_at": "2024-01-03T00:00:00Z",
             "points": 3, "num_comments": 1, "url": None}


def _mock_client():
    def handler(req):
        page = int(req.url.params.get("page", "0"))
        pages = {0: [HIT_BTC, HIT_ETH], 1: [HIT_OTHER]}
        return httpx.Response(200, json={"hits": pages.get(page, []), "nbPages": 2})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_created_at_iso_and_epoch():
    assert parse_created_at(HIT_BTC) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_created_at({"created_at_i": 1704067200}) == \
        datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert parse_created_at({}) is None


def test_query_terms_prefers_descriptive_alias():
    assert query_terms()["BTC"] == "bitcoin"
    assert query_terms()["LINK"] == "chainlink"


def test_map_hit_fields_and_skips():
    row = map_hit(HIT_BTC)
    assert row["post_id"] == "hackernews:111"
    assert row["source"] == "hackernews"
    assert row["author"] == "alice"
    assert row["tickers"] == ["BTC"]
    assert row["likes"] == 500 and row["replies"] == 200 and row["retweets"] == 0
    assert row["published_at"] == datetime(2024, 1, 1, tzinfo=timezone.utc)

    eth = map_hit(HIT_ETH)
    assert eth["tickers"] == ["ETH"]
    assert "ETH question" in eth["text"]  # story_text tags stripped
    assert eth["url"] == "https://news.ycombinator.com/item?id=222"  # fallback
    assert map_hit({"objectID": "", "title": "x", "created_at": "2024-01-01T00:00:00Z"}) is None


def test_fetch_stories_paginates():
    hits = fetch_stories(_mock_client(), "bitcoin",
                         datetime(2024, 1, 1, tzinfo=timezone.utc),
                         datetime(2024, 1, 10, tzinfo=timezone.utc), sleep=0.0)
    assert [h["objectID"] for h in hits] == ["111", "222", "333"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE source = 'TEST_hackernews'",
        settings={"mutations_sync": 1},
    )


def test_backfill_inserts_and_is_idempotent(ch_client):
    kwargs = dict(start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                  end=datetime(2024, 1, 10, tzinfo=timezone.utc),
                  sleep=0.0, source="TEST_hackernews", terms={"BTC": "bitcoin"})
    n1 = backfill_hackernews(ch_client, http_client=_mock_client(), **kwargs)
    assert n1 == 2  # BTC + ETH rows (editor story has no ticker)
    backfill_hackernews(ch_client, http_client=_mock_client(), **kwargs)
    count = ch_client.query(
        f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
        "WHERE source = 'TEST_hackernews'"
    ).result_rows[0][0]
    assert count == 2
