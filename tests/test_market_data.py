from datetime import datetime, timezone

import pytest

from src.ingestion.market_data import (
    insert_funding,
    insert_ohlcv,
    max_ts,
)
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    client.command(f"ALTER TABLE {db}.ohlcv DELETE WHERE symbol = 'TEST'")
    client.command(f"ALTER TABLE {db}.funding_rates DELETE WHERE symbol = 'TEST'")


def _ohlcv_row(ts):
    return {
        "exchange": "binance_us", "symbol": "TEST", "interval": "1h", "ts": ts,
        "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
        "volume": 12.5, "quote_volume": 1300.0, "num_trades": 42,
    }


def test_insert_ohlcv_and_max_ts(ch_client):
    t1 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2022, 1, 1, 1, 0, tzinfo=timezone.utc)
    assert insert_ohlcv(ch_client, [_ohlcv_row(t1), _ohlcv_row(t2)]) == 2
    assert insert_ohlcv(ch_client, []) == 0
    latest = max_ts(ch_client, "ohlcv", "binance_us", "TEST", "1h")
    assert latest == t2
    assert latest.tzinfo is not None  # must be tz-aware


def test_max_ts_empty_series_returns_none(ch_client):
    assert max_ts(ch_client, "ohlcv", "binance_us", "TEST", "1h") is None
    assert max_ts(ch_client, "funding_rates", "hyperliquid", "TEST") is None


def test_insert_funding_and_max_ts(ch_client):
    t1 = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2023, 6, 1, 8, 0, tzinfo=timezone.utc)
    rows = [
        {"exchange": "hyperliquid", "symbol": "TEST", "ts": t1,
         "funding_rate": 0.0001, "mark_price": 0.0, "next_funding_ts": t2},
        {"exchange": "hyperliquid", "symbol": "TEST", "ts": t2,
         "funding_rate": -0.0002, "mark_price": 0.0,
         "next_funding_ts": datetime(2023, 6, 1, 16, 0, tzinfo=timezone.utc)},
    ]
    assert insert_funding(ch_client, rows) == 2
    assert max_ts(ch_client, "funding_rates", "hyperliquid", "TEST") == t2


def test_insert_ohlcv_idempotent(ch_client):
    t1 = datetime(2022, 2, 1, 0, 0, tzinfo=timezone.utc)
    insert_ohlcv(ch_client, [_ohlcv_row(t1)])
    insert_ohlcv(ch_client, [_ohlcv_row(t1)])
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.ohlcv FINAL WHERE symbol = 'TEST'"
    ).result_rows
    assert rows[0][0] == 1
