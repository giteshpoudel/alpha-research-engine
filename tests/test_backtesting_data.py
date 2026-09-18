from datetime import datetime, timezone

import pytest

from src.backtesting.data import (
    WINDOWS,
    load_funding,
    load_ohlcv,
    load_sentiment,
    slice_window,
)
from src.ingestion.schemas import database_name, get_clickhouse_client

IS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ch_client():
    return get_clickhouse_client()


def test_windows_definition():
    assert WINDOWS["IS"] == (
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc),
    )
    assert WINDOWS["OOS"] == (datetime(2025, 1, 1, tzinfo=timezone.utc), None)
    assert WINDOWS["PRELIM"] == (datetime(2026, 8, 25, tzinfo=timezone.utc), None)


def test_load_ohlcv_btc_is_window(ch_client):
    df = load_ohlcv(ch_client, "BTC", interval="1h",
                    start=IS_START, end=datetime(2022, 2, 1, tzinfo=timezone.utc))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 744  # 31 days * 24h
    assert df.index[0] == IS_START
    assert df.index.is_monotonic_increasing
    assert df.index.tz is not None
    first = df.iloc[0]
    assert first["close"] == pytest.approx(46670.23)  # verified reference candle


def test_load_funding_btc(ch_client):
    s = load_funding(ch_client, "BTC",
                     start=datetime(2023, 5, 12, tzinfo=timezone.utc),
                     end=datetime(2023, 5, 15, tzinfo=timezone.utc))
    assert s.name == "funding_rate"
    assert len(s) > 0
    assert s.index.tz is not None
    assert (s.abs() < 0.01).all()  # funding rates are small fractions


def test_load_sentiment_and_slice_window(ch_client):
    df = load_sentiment(ch_client, "BTC", bucket="1h")
    assert list(df.columns) == ["weighted_score", "velocity", "engagement_ratio"]
    prelim = slice_window(df, "PRELIM")
    assert len(prelim) == len(df)  # all sentiment data is in the PRELIM window
    is_slice = slice_window(df, "IS")
    assert len(is_slice) == 0  # nothing in the IS window (guardrail #1 sanity)


def test_load_ohlcv_index_unique_and_exchange_scoped(ch_client):
    df = load_ohlcv(ch_client, "BTC", interval="1h",
                    start=IS_START, end=datetime(2022, 1, 2, tzinfo=timezone.utc))
    assert df.index.is_unique

    ts = datetime(2026, 9, 1, tzinfo=timezone.utc)
    columns = ["exchange", "symbol", "interval", "ts", "open", "high", "low",
               "close", "volume", "quote_volume", "num_trades"]
    rows = [
        ["binance_us", "TEST", "1h", ts, 111.0, 112.0, 110.0, 111.5, 10.0, 1110.0, 100],
        ["coinbase", "TEST", "1h", ts, 222.0, 223.0, 221.0, 222.5, 20.0, 2220.0, 200],
    ]
    ch_client.insert(f"{database_name()}.ohlcv", rows, column_names=columns)
    try:
        df = load_ohlcv(ch_client, "TEST", interval="1h",
                        start=datetime(2026, 8, 31, tzinfo=timezone.utc),
                        end=datetime(2026, 9, 2, tzinfo=timezone.utc))
        assert df.index.is_unique
        assert len(df) == 1
        assert df.iloc[0]["open"] == pytest.approx(111.0)  # binance_us row wins
    finally:
        ch_client.command(
            f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TEST'",
            settings={"mutations_sync": 1},
        )


def test_load_funding_is_hyperliquid_only(ch_client):
    s = load_funding(ch_client, "BTC",
                     start=datetime(2023, 5, 12, tzinfo=timezone.utc),
                     end=datetime(2023, 5, 13, tzinfo=timezone.utc))
    assert len(s) > 0


def test_load_sentiment_indexes_at_bucket_end(ch_client):
    import pandas as pd

    raw = ch_client.query(
        f"SELECT min(bucket_start) FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE ticker = 'BTC' AND bucket_size = '1h'"
    ).result_rows[0][0]
    assert raw is not None
    expected = raw + pd.Timedelta(hours=1)
    if expected.tzinfo is None:
        expected = expected.replace(tzinfo=timezone.utc)
    df = load_sentiment(ch_client, "BTC", bucket="1h")
    assert df.index.min() == expected  # knowable only at bucket end (no lookahead)
