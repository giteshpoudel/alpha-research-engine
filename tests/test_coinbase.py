from datetime import datetime, timezone

import httpx
import pytest

from src.ingestion.coinbase import (
    MATIC_MIGRATION,
    backfill_matic,
    fetch_candles,
    map_candle,
)
from src.ingestion.schemas import database_name, get_clickhouse_client

# [time, low, high, open, close, volume] -- Coinbase puts low BEFORE high
CANDLE = [1640995200, 46192.43, 46715.27, 46205.0, 46670.23, 686.07385399]

T0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_map_candle_low_high_order():
    row = map_candle("1h", CANDLE)
    assert row["exchange"] == "coinbase"
    assert row["symbol"] == "MATIC"
    assert row["ts"] == T0
    assert row["low"] == 46192.43
    assert row["high"] == 46715.27
    assert row["open"] == 46205.0
    assert row["close"] == 46670.23
    assert row["volume"] == 686.07385399
    assert row["quote_volume"] == 0.0
    assert row["num_trades"] == 0


def test_fetch_candles_reverses_newest_first():
    batch = [[1640998800, 1, 2, 1.5, 1.8, 10], [1640995200, 1, 2, 1.2, 1.5, 20]]
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=batch)))
    rows = fetch_candles(client, "TEST-USD", "1h", T0,
                         datetime(2022, 1, 1, 2, 0, tzinfo=timezone.utc), symbol="TEST")
    assert [r["ts"] for r in rows] == [T0, datetime(2022, 1, 1, 1, 0, tzinfo=timezone.utc)]


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TEST'",
                   settings={"mutations_sync": 1})


def test_backfill_splits_at_migration(ch_client):
    requested = []

    def handler(req):
        product = req.url.path.split("/products/")[1].split("/")[0]
        requested.append(product)
        # Return a candle at the requested window start so the two migration
        # segments produce distinct ts (same-ts rows would collapse under
        # ReplacingMergeTree FINAL).
        candle = list(CANDLE)
        candle[0] = int(datetime.fromisoformat(req.url.params["start"]).timestamp())
        return httpx.Response(200, json=[candle])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    start = MATIC_MIGRATION - timedelta_days(1)
    end = MATIC_MIGRATION + timedelta_days(1)
    inserted = backfill_matic(ch_client, http_client=client, intervals=("1d",),
                              start=start, end=end, symbol="TEST")
    assert requested == ["MATIC-USD", "POL-USD"]
    assert inserted == 2
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 2


def timedelta_days(n):
    from datetime import timedelta
    return timedelta(days=n)
