from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.ingestion.hyperliquid import backfill_funding, fetch_funding, map_funding
from src.ingestion.schemas import database_name, get_clickhouse_client

T0 = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)


def _rec(hours_after_t0, rate="0.0001"):
    ms = int((T0 + timedelta(hours=hours_after_t0)).timestamp() * 1000)
    return {"coin": "TEST", "fundingRate": rate, "premium": "0.00009", "time": ms}


def test_map_funding_fields():
    next_ts = T0 + timedelta(hours=8)
    row = map_funding("TEST", _rec(0), next_ts)
    assert row["exchange"] == "hyperliquid"
    assert row["symbol"] == "TEST"
    assert row["ts"] == T0
    assert row["funding_rate"] == 0.0001
    assert row["mark_price"] == 0.0
    assert row["next_funding_ts"] == next_ts


def test_fetch_funding_pagination_and_next_ts_chain():
    calls = []
    page1 = [_rec(i * 8) for i in range(500)]
    page2 = [_rec(500 * 8), _rec(501 * 8)]

    def handler(req):
        import json as _json
        body = _json.loads(req.content)
        calls.append(body["startTime"])
        batch = page1 if len(calls) == 1 else page2
        start = body["startTime"]
        return httpx.Response(200, json=[r for r in batch if r["time"] >= start])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_funding(client, "TEST", T0)
    assert len(calls) == 2
    assert calls[1] == page1[-1]["time"] + 1
    assert len(rows) == 502
    # next_funding_ts chains to the following record...
    assert rows[0]["next_funding_ts"] == rows[1]["ts"]
    # ...and the last record gets ts + 8h
    assert rows[-1]["next_funding_ts"] == rows[-1]["ts"] + timedelta(hours=8)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.funding_rates DELETE WHERE symbol = 'TEST'",
                   settings={"mutations_sync": 1})


def test_backfill_funding_inserts_and_resumes(ch_client):
    records = [_rec(0), _rec(8), _rec(16)]

    def handler(req):
        import json as _json
        start = _json.loads(req.content)["startTime"]
        return httpx.Response(200, json=[r for r in records if r["time"] >= start])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert backfill_funding(ch_client, http_client=client, coins=("TEST",), start=T0) == 3
    # resume: max_ts is the last record -> window start passes it -> 0 new
    assert backfill_funding(ch_client, http_client=client, coins=("TEST",), start=T0) == 0
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.funding_rates FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 3
