"""Shared insert path and resume cursor for market-data tables."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name

OHLCV_COLUMNS = (
    "exchange", "symbol", "interval", "ts",
    "open", "high", "low", "close", "volume", "quote_volume", "num_trades",
)

FUNDING_COLUMNS = (
    "exchange", "symbol", "ts", "funding_rate", "mark_price", "next_funding_ts",
)


def insert_ohlcv(ch_client, rows: list[dict]) -> int:
    """Insert ohlcv row dicts. Dedup is structural (ReplacingMergeTree)."""
    if not rows:
        return 0
    data = [
        [r["exchange"], r["symbol"], r["interval"], r["ts"],
         r["open"], r["high"], r["low"], r["close"], r["volume"],
         r["quote_volume"], r["num_trades"]]
        for r in rows
    ]
    ch_client.insert(f"{database_name()}.ohlcv", data, column_names=list(OHLCV_COLUMNS))
    return len(rows)


def insert_funding(ch_client, rows: list[dict]) -> int:
    """Insert funding_rates row dicts. Dedup is structural (ReplacingMergeTree)."""
    if not rows:
        return 0
    data = [
        [r["exchange"], r["symbol"], r["ts"], r["funding_rate"],
         r["mark_price"], r["next_funding_ts"]]
        for r in rows
    ]
    ch_client.insert(f"{database_name()}.funding_rates", data, column_names=list(FUNDING_COLUMNS))
    return len(rows)


def max_ts(ch_client, table: str, exchange: str, symbol: str,
           interval: str | None = None) -> datetime | None:
    """Latest stored ts for a series (tz-aware UTC), or None for an empty series.

    maxOrNull returns NULL for empty sets (plain max returns the type default,
    epoch 0). ClickHouse may return naive datetimes; normalize to aware UTC
    because callers compare against tz-aware resume windows.
    """
    sql = (f"SELECT maxOrNull(ts) FROM {database_name()}.{table} "
           "WHERE exchange = {e:String} AND symbol = {s:String}")
    params = {"e": exchange, "s": symbol}
    if interval is not None:
        sql += " AND interval = {i:String}"
        params["i"] = interval
    rows = ch_client.query(sql, parameters=params).result_rows
    latest = rows[0][0] if rows else None
    if latest is None:
        return None
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest
