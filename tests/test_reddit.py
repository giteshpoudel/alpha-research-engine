from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.posts import insert_posts
from src.ingestion.reddit import fetch_new_posts, map_post, poll_subreddits
from src.ingestion.schemas import database_name, get_clickhouse_client

LISTING_FIXTURE = {
    "data": {
        "children": [
            {"data": {
                "name": "t3_abc", "author": "trader1",
                "title": "Bitcoin breaking out", "selftext": "BTC looks bullish",
                "score": 42, "num_comments": 7, "created_utc": 1700000000.0,
                "permalink": "/r/CryptoCurrency/comments/abc/x/",
            }},
            {"data": {
                "name": "t3_bot", "author": "AutoModerator",
                "title": "Daily discussion", "selftext": "thread",
                "score": 1, "num_comments": 100, "created_utc": 1700000060.0,
                "permalink": "/r/CryptoCurrency/comments/bot/y/",
            }},
            {"data": {
                "name": "t3_empty", "author": "lurker",
                "title": "", "selftext": "",
                "score": 0, "num_comments": 0, "created_utc": 1700000120.0,
                "permalink": "/r/CryptoCurrency/comments/empty/z/",
            }},
        ]
    }
}


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_map_post_fields():
    row = map_post(LISTING_FIXTURE["data"]["children"][0]["data"])
    assert row["post_id"] == "reddit:t3_abc"
    assert row["source"] == "reddit"
    assert row["text"] == "Bitcoin breaking out\nBTC looks bullish"
    assert row["tickers"] == ["BTC"]
    assert row["likes"] == 42
    assert row["replies"] == 7
    assert row["url"] == "https://www.reddit.com/r/CryptoCurrency/comments/abc/x/"
    assert row["published_at"] == datetime.fromtimestamp(1700000000.0, tz=timezone.utc)


def test_map_post_skips_bots_and_empty():
    assert map_post(LISTING_FIXTURE["data"]["children"][1]["data"]) is None
    assert map_post(LISTING_FIXTURE["data"]["children"][2]["data"]) is None


def test_fetch_new_posts_filters():
    client = _mock_client(lambda req: httpx.Response(200, json=LISTING_FIXTURE))
    rows = fetch_new_posts(client, "CryptoCurrency")
    assert [r["post_id"] for r in rows] == ["reddit:t3_abc"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')",
        settings={"mutations_sync": 1},
    )


def test_poll_subreddits_inserts_and_is_idempotent(ch_client):
    fixture = {
        "data": {"children": [{"data": {
            "name": "t3_test1", "author": "tester",
            "title": "test: bitcoin news", "selftext": "",
            "score": 5, "num_comments": 2, "created_utc": 1700000000.0,
            "permalink": "/r/Bitcoin/comments/test1/t/",
        }}]}
    }
    # map_post keys post_id off the fullname; force the test: prefix
    def handler(req):
        return httpx.Response(200, json=fixture)

    http_client = _mock_client(handler)
    import src.ingestion.reddit as reddit_mod
    original = reddit_mod.map_post
    reddit_mod.map_post = lambda post: {**original(post), "post_id": "test:" + original(post)["post_id"]}
    try:
        assert poll_subreddits(ch_client, http_client=http_client, subreddits=("Bitcoin",)) == 1
        assert poll_subreddits(ch_client, http_client=http_client, subreddits=("Bitcoin",)) == 1
        rows = ch_client.query(
            f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
            "WHERE post_id = 'test:reddit:t3_test1'"
        ).result_rows
        assert rows[0][0] == 1
    finally:
        reddit_mod.map_post = original
