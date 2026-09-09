from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.binance_us import backfill_ohlcv, fetch_klines, map_kline
from src.ingestion.schemas import database_name, get_clickhouse_client

# [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, ...]
KLINE = [1640995200000, "46192.43", "46715.27", "46192.43", "46670.23",
         "17.363974", 1640998799999, "806563.32", 543, "12.21", "567219.24", "0"]

T0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_map_kline_fields():
    row = map_kline("BTC", "1h", KLINE)
    assert row["exchange"] == "binance_us"
    assert row["symbol"] == "BTC"
    assert row["interval"] == "1h"
    assert row["ts"] == T0
    assert row["open"] == 46192.43
    assert row["high"] == 46715.27
    assert row["low"] == 46192.43
    assert row["close"] == 46670.23
    assert row["volume"] == 17.363974
    assert row["quote_volume"] == 806563.32
    assert row["num_trades"] == 543


def _kline_at(ms):
    k = list(KLINE)
    k[0] = ms
    return k


def test_fetch_klines_paginates():
    calls = []

    def handler(req):
        calls.append(dict(req.url.params))
        start = int(req.url.params["startTime"])
        if start == 1640995200000:
            # full page of 1000 -> forces a second request
            return httpx.Response(200, json=[_kline_at(start + i * 3600000) for i in range(1000)])
        return httpx.Response(200, json=[_kline_at(start)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    end = datetime(2022, 5, 9, tzinfo=timezone.utc)  # beyond the mocked pages
    rows = fetch_klines(client, "TEST", "1h", T0, end)
    assert len(calls) == 2
    assert int(calls[1]["startTime"]) == 1640995200000 + 1000 * 3600000  # advanced past last candle
    assert rows[0]["ts"] == T0
    assert all(r["symbol"] == "TEST" for r in rows)


def test_fetch_klines_falls_back_to_usd_pair():
    def handler(req):
        if req.url.params["symbol"] == "TESTUSDT":
            return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
        return httpx.Response(200, json=[KLINE])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_klines(client, "TEST", "1h", T0, datetime(2022, 1, 2, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert rows[0]["ts"] == T0


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TEST'")


def test_backfill_inserts_and_resumes(ch_client):
    candles = [_kline_at(1640995200000 + i * 3600000) for i in range(3)]
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=candles)))
    end = datetime(2022, 1, 1, 3, 0, tzinfo=timezone.utc)
    assert backfill_ohlcv(ch_client, http_client=client, symbols=("TEST",),
                          intervals=("1h",), start=T0, end=end) == 3
    # resume: max_ts is candle 2 -> window start would be candle 3 == end -> nothing to do
    assert backfill_ohlcv(ch_client, http_client=client, symbols=("TEST",),
                          intervals=("1h",), start=T0, end=end) == 0
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 3
