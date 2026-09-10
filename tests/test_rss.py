from datetime import datetime, timezone

import feedparser
import httpx
import pytest

from src.ingestion.rss import fetch_feed, map_entry, poll_feeds
from src.ingestion.schemas import database_name, get_clickhouse_client

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>CoinTelegraph</title>
<item>
  <title>Ethereum ETF approved, ETH rallies</title>
  <link>https://example.com/eth-etf</link>
  <guid>https://example.com/eth-etf</guid>
  <pubDate>Tue, 14 Nov 2023 22:00:00 +0000</pubDate>
  <description>ETH jumps on &lt;b&gt;ETF&lt;/b&gt; approval</description>
</item>
<item>
  <title></title>
  <link>https://example.com/empty</link>
  <guid>https://example.com/empty</guid>
</item>
</channel></rss>
"""


def _entries():
    return feedparser.parse(FEED_XML).entries


def test_map_entry_fields():
    row = map_entry("CoinTelegraph", _entries()[0])
    assert row["post_id"] == "news:https://example.com/eth-etf"
    assert row["source"] == "news"
    assert row["author"] == "CoinTelegraph"
    assert row["text"] == "Ethereum ETF approved, ETH rallies\nETH jumps on ETF approval"
    assert row["tickers"] == ["ETH"]
    assert row["likes"] == 0 and row["retweets"] == 0 and row["replies"] == 0
    assert row["published_at"] == datetime(2023, 11, 14, 22, 0, tzinfo=timezone.utc)


def test_map_entry_skips_empty():
    assert map_entry("CoinTelegraph", _entries()[1]) is None


def test_fetch_feed_parses_and_filters():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=FEED_XML)))
    rows = fetch_feed("CoinTelegraph", "https://example.com/rss", http_client=client)
    assert [r["post_id"] for r in rows] == ["news:https://example.com/eth-etf"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')",
        settings={"mutations_sync": 1},
    )


def test_poll_feeds_inserts_and_is_idempotent(ch_client):
    http_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=FEED_XML)))
    import src.ingestion.rss as rss_mod
    original = rss_mod.map_entry
    rss_mod.map_entry = lambda name, entry: (
        None if original(name, entry) is None
        else {**original(name, entry), "post_id": "test:" + original(name, entry)["post_id"]}
    )
    try:
        assert poll_feeds(ch_client, http_client=http_client,
                          feeds=(("CoinTelegraph", "https://example.com/rss"),)) == 1
        assert poll_feeds(ch_client, http_client=http_client,
                          feeds=(("CoinTelegraph", "https://example.com/rss"),)) == 1
        rows = ch_client.query(
            f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
            "WHERE post_id = 'test:news:https://example.com/eth-etf'"
        ).result_rows
        assert rows[0][0] == 1
    finally:
        rss_mod.map_entry = original
