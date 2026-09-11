from datetime import datetime, timezone

import pytest

from src.backtesting.data import (
    WINDOWS,
    load_funding,
    load_ohlcv,
    load_sentiment,
    slice_window,
)
from src.ingestion.schemas import get_clickhouse_client

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
