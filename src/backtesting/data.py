"""Data access for backtesting: ClickHouse -> pandas, with window discipline.

All window slicing lives here (guardrail #1): strategies only ever receive
pre-sliced data and cannot reach outside their window by accident.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.ingestion.schemas import database_name

WINDOWS: dict[str, tuple[datetime, datetime | None]] = {
    "IS": (datetime(2022, 1, 1, tzinfo=timezone.utc),
           datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc)),
    "OOS": (datetime(2025, 1, 1, tzinfo=timezone.utc), None),
    "PRELIM": (datetime(2026, 8, 25, tzinfo=timezone.utc), None),
}


def _time_filter(start, end):
    clauses, params = [], {}
    if start is not None:
        clauses.append("ts >= {start:DateTime64(3)}")
        params["start"] = start
    if end is not None:
        clauses.append("ts < {end:DateTime64(3)}")
        params["end"] = end
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _utc_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    return idx


def _pick_ohlcv_exchange(ch_client, symbol: str) -> str:
    """Single exchange for a symbol, preferring binance_us over coinbase."""
    rows = ch_client.query(
        f"SELECT DISTINCT exchange FROM {database_name()}.ohlcv "
        "WHERE symbol = {s:String}",
        parameters={"s": symbol},
    ).result_rows
    available = {r[0] for r in rows}
    for preferred in ("binance_us", "coinbase"):
        if preferred in available:
            return preferred
    return ""


def load_ohlcv(ch_client, symbol: str, interval: str = "1h",
               start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """Load OHLCV for one symbol from a single exchange.

    If the symbol exists on multiple exchanges, binance_us is preferred over
    coinbase; only rows from that one exchange are returned. ``end`` is
    exclusive.
    """
    tf, params = _time_filter(start, end)
    params["s"] = symbol
    params["i"] = interval
    params["e"] = _pick_ohlcv_exchange(ch_client, symbol)
    rows = ch_client.query(
        f"SELECT ts, open, high, low, close, volume FROM {database_name()}.ohlcv FINAL "
        f"WHERE symbol = {{s:String}} AND interval = {{i:String}} AND exchange = {{e:String}}{tf} ORDER BY ts",
        parameters=params,
    ).result_rows
    if not rows:
        raise ValueError(f"no ohlcv data for {symbol} {interval} in [{start}, {end}]")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = _utc_index(df.pop("ts"))
    if not df.index.is_unique:
        raise ValueError(f"duplicate timestamps for {symbol} {interval} ohlcv — exchange scoping failed")
    return df


def load_funding(ch_client, symbol: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.Series:
    tf, params = _time_filter(start, end)
    params["s"] = symbol
    params["e"] = "hyperliquid"
    rows = ch_client.query(
        f"SELECT ts, funding_rate FROM {database_name()}.funding_rates FINAL "
        f"WHERE symbol = {{s:String}} AND exchange = {{e:String}}{tf} ORDER BY ts",
        parameters=params,
    ).result_rows
    if not rows:
        raise ValueError(f"no funding data for {symbol} in [{start}, {end}]")
    idx = _utc_index([r[0] for r in rows])
    series = pd.Series([float(r[1]) for r in rows], index=idx, name="funding_rate")
    if not series.index.is_unique:
        raise ValueError(f"duplicate timestamps for {symbol} funding — exchange scoping failed")
    return series


def load_sentiment(ch_client, symbol: str, bucket: str = "1h") -> pd.DataFrame:
    rows = ch_client.query(
        f"SELECT bucket_start, weighted_score, velocity, engagement_ratio "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE ticker = {s:String} AND bucket_size = {b:String} ORDER BY bucket_start",
        parameters={"s": symbol, "b": bucket},
    ).result_rows
    if not rows:
        raise ValueError(f"no sentiment metrics for {symbol} {bucket}")
    df = pd.DataFrame(rows, columns=["ts", "weighted_score", "velocity", "engagement_ratio"])
    df.index = _utc_index(df.pop("ts"))
    return df


def slice_window(data, window: str):
    """Rows of a UTC-indexed DataFrame/Series within WINDOWS[window]."""
    start, end = WINDOWS[window]
    sliced = data.loc[start:]
    if end is not None:
        sliced = sliced.loc[:end]
    return sliced
