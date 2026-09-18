from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.gdelt import (
    backfill_gdelt,
    build_query,
    fetch_articles,
    map_article,
    parse_seendate,
)
from src.ingestion.schemas import database_name, get_clickhouse_client

PAYLOAD = {
    "articles": [
        {"url": "https://ex.com/a", "title": "Bitcoin ETF approved, BTC rallies",
         "seendate": "20240101T000000Z", "domain": "ex.com", "language": "English"},
        {"url": "https://ex.com/b", "title": "Ethereum upgrade ships",
         "seendate": "20240102T000000Z", "domain": "ex.com", "language": "English"},
    ]
}


def _dt(day):
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def test_parse_seendate():
    assert parse_seendate("20240101T000000Z") == _dt(1)
    assert parse_seendate("garbage") is None


def test_build_query_includes_tickers_and_english():
    query = build_query()
    assert "bitcoin" in query and "btc" in query
    assert "sourcelang:english" in query


def test_map_article_fields_and_skips():
    row = map_article(PAYLOAD["articles"][0])
    assert row["post_id"] == "gdelt:https://ex.com/a"
    assert row["source"] == "gdelt"
    assert row["author"] == "ex.com"
    assert row["tickers"] == ["BTC"]
    assert row["published_at"] == _dt(1)
    assert map_article({"url": "", "title": "x", "seendate": "20240101T000000Z"}) is None
    assert map_article({"url": "https://x", "title": "", "seendate": "20240101T000000Z"}) is None


def test_fetch_articles_parses_json():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=PAYLOAD)))
    articles = fetch_articles(client, _dt(1), _dt(2), "bitcoin")
    assert [a["url"] for a in articles] == ["https://ex.com/a", "https://ex.com/b"]


def test_fetch_articles_rate_limit_raises():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, text="Please limit requests to one every 5 seconds")))
    with pytest.raises(RuntimeError, match="rate limit"):
        fetch_articles(client, _dt(1), _dt(2), "bitcoin")


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.sentiment_posts DELETE WHERE source = 'TEST_gdelt'",
        settings={"mutations_sync": 1},
    )


def test_backfill_inserts_and_is_idempotent(ch_client):
    http_client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=PAYLOAD)))
    n1 = backfill_gdelt(ch_client, http_client=http_client, start=_dt(1), end=_dt(3),
                        sleep=0.0, source="TEST_gdelt", query="bitcoin")
    assert n1 == 4  # 2 windows x 2 articles
    backfill_gdelt(ch_client, http_client=http_client, start=_dt(1), end=_dt(3),
                   sleep=0.0, source="TEST_gdelt", query="bitcoin")
    count = ch_client.query(
        f"SELECT count() FROM {database_name()}.sentiment_posts FINAL "
        "WHERE source = 'TEST_gdelt'"
    ).result_rows[0][0]
    assert count == 2  # Replace keys collapse duplicates
