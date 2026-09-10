from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.stocktwits import fetch_stream, map_message, poll_symbols

MESSAGE_FIXTURE = {
    "id": 663469073,
    "body": "$BTC looking strong, breakout incoming",
    "created_at": "2026-09-02T18:58:51Z",
    "user": {"id": 9565930, "username": "Agonas"},
    "symbols": [{"id": 11418, "symbol": "BTC.X"}],
    "entities": {"media": [], "sentiment": {"basic": "Bullish"}},
    "likes": {"total": 7},
}

UNLABELED_FIXTURE = {
    "id": 663469074,
    "body": "BTC sideways all day, eth too",
    "created_at": "2026-09-02T19:01:00Z",
    "user": {"id": 42, "username": "watcher"},
    "symbols": [{"id": 11418, "symbol": "BTC.X"}, {"id": 11419, "symbol": "ETH.X"}],
    "entities": {"media": [], "sentiment": None},
}

EMPTY_FIXTURE = {
    "id": 663469075,
    "body": "   ",
    "created_at": "2026-09-02T19:02:00Z",
    "user": {"id": 1, "username": "nobody"},
    "symbols": [],
    "entities": {"sentiment": None},
}

STREAM_FIXTURE = {"messages": [MESSAGE_FIXTURE, UNLABELED_FIXTURE, EMPTY_FIXTURE]}


def test_map_message_fields():
    row = map_message(MESSAGE_FIXTURE)
    assert row["post_id"] == "stocktwits:663469073"
    assert row["source"] == "stocktwits"
    assert row["author"] == "Agonas"
    assert row["text"] == "$BTC looking strong, breakout incoming"
    assert row["tickers"] == ["BTC"]
    assert row["likes"] == 7
    assert row["retweets"] == 0
    assert row["replies"] == 0
    assert row["url"] == "https://stocktwits.com/Agonas/message/663469073"
    assert row["published_at"] == datetime(2026, 9, 2, 18, 58, 51, tzinfo=timezone.utc)
    assert row["label"] == "Bullish"


def test_map_message_unlabeled_and_multi_ticker():
    row = map_message(UNLABELED_FIXTURE)
    assert row["label"] is None
    assert row["tickers"] == ["BTC", "ETH"]


def test_map_message_skips_empty():
    assert map_message(EMPTY_FIXTURE) is None


def test_fetch_stream_filters():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=STREAM_FIXTURE)))
    rows = fetch_stream(client, "BTC.X")
    assert [r["post_id"] for r in rows] == ["stocktwits:663469073", "stocktwits:663469074"]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')",
        settings={"mutations_sync": 1},
    )


def test_poll_symbols_inserts_idempotently_and_stores_label(ch_client):
    http_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=STREAM_FIXTURE)))
    import src.ingestion.stocktwits as st_mod
    original = st_mod.map_message
    st_mod.map_message = lambda msg: (
        None if original(msg) is None
        else {**original(msg), "post_id": "test:" + original(msg)["post_id"]}
    )
    try:
        assert poll_symbols(ch_client, http_client=http_client, symbols=("BTC.X",)) == 2
        assert poll_symbols(ch_client, http_client=http_client, symbols=("BTC.X",)) == 2
        rows = ch_client.query(
            f"SELECT post_id, label FROM {database_name()}.sentiment_posts FINAL "
            "WHERE startsWith(post_id, 'test:stocktwits:') ORDER BY post_id"
        ).result_rows
        assert rows == [
            ("test:stocktwits:663469073", "Bullish"),
            ("test:stocktwits:663469074", None),
        ]
    finally:
        st_mod.map_message = original
