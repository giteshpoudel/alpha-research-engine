"""Hyperliquid funding-rate history backfill (exchange='hyperliquid').

POST /info {"type":"fundingHistory"} returns up to 500 records per call,
each {coin, fundingRate, premium, time}. mark_price is not available from
this endpoint and is stored as 0.0 (documented gap). next_funding_ts is
derived from the next record in the series (ts + 8h for the latest).
History starts at each coin's listing; empty responses are normal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.http import post_with_backoff
from src.ingestion.market_data import insert_funding, max_ts
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES

_INFO_URL = "https://api.hyperliquid.xyz/info"
_PAGE = 500
_PERP_COLUMNS = ("exchange", "symbol", "ts", "mark_price", "oracle_price",
                 "mid_price", "open_interest")


def map_funding(coin: str, rec: dict, next_ts: datetime) -> dict:
    """Hyperliquid funding record -> funding_rates row dict."""
    return {
        "exchange": "hyperliquid",
        "symbol": coin,
        "ts": datetime.fromtimestamp(rec["time"] / 1000, tz=timezone.utc),
        "funding_rate": float(rec["fundingRate"]),
        "mark_price": 0.0,
        "next_funding_ts": next_ts,
    }


def fetch_funding(http_client: httpx.Client, coin: str, start: datetime) -> list[dict]:
    """All funding records for coin from `start` to now, oldest-first."""
    records: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    while True:
        resp = post_with_backoff(
            _INFO_URL,
            json={"type": "fundingHistory", "coin": coin, "startTime": cursor},
            client=http_client,
            timeout=30.0,
        )
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        if len(batch) < _PAGE:
            break
        cursor = batch[-1]["time"] + 1
    rows = []
    for i, rec in enumerate(records):
        if i + 1 < len(records):
            next_ts = datetime.fromtimestamp(records[i + 1]["time"] / 1000, tz=timezone.utc)
        else:
            next_ts = datetime.fromtimestamp(rec["time"] / 1000, tz=timezone.utc) + timedelta(hours=8)
        rows.append(map_funding(coin, rec, next_ts))
    return rows


def backfill_funding(ch_client, http_client: httpx.Client | None = None,
                     coins: tuple[str, ...] | None = None,
                     start: datetime | None = None) -> int:
    """Backfill (or top up) funding history per coin. Resumable via max_ts.

    A failing coin is logged and skipped, never fatal.
    """
    coins = coins or tuple(TICKER_ALIASES)
    start = start or datetime(2023, 1, 1, tzinfo=timezone.utc)
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for coin in coins:
            resume = max_ts(ch_client, "funding_rates", "hyperliquid", coin)
            window_start = max(start, resume + timedelta(milliseconds=1)) if resume else start
            try:
                rows = fetch_funding(http_client, coin, window_start)
                inserted += insert_funding(ch_client, rows)
            except Exception as exc:
                print(f"hyperliquid: skipping {coin}: {exc}")
    finally:
        if owns_client:
            http_client.close()
    return inserted


def map_asset_ctxs(payload: list, coins: tuple[str, ...] | None = None,
                   ts: datetime | None = None) -> list[dict]:
    """metaAndAssetCtxs -> perp_marks rows. ``payload`` is [meta, assetCtxs]."""
    meta, ctxs = payload[0], payload[1]
    ts = ts or datetime.now(timezone.utc)
    rows = []
    for entry, ctx in zip(meta.get("universe", []), ctxs):
        symbol = entry.get("name")
        if coins is not None and symbol not in coins:
            continue
        rows.append({
            "exchange": "hyperliquid", "symbol": symbol, "ts": ts,
            "mark_price": float(ctx.get("markPx") or 0.0),
            "oracle_price": float(ctx.get("oraclePx") or 0.0),
            "mid_price": float(ctx.get("midPx") or 0.0),
            "open_interest": float(ctx.get("openInterest") or 0.0),
        })
    return rows


def snapshot_perp_marks(ch_client, coins: tuple[str, ...] | None = None,
                        http_client: httpx.Client | None = None) -> int:
    """One live perp mark/oracle snapshot for all (or selected) coins."""
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    try:
        resp = post_with_backoff(_INFO_URL, json={"type": "metaAndAssetCtxs"},
                                 client=http_client, timeout=30.0)
        rows = map_asset_ctxs(resp.json(), coins=coins)
        if not rows:
            return 0
        ch_client.insert(f"{database_name()}.perp_marks",
                         [[r[c] for c in _PERP_COLUMNS] for r in rows],
                         column_names=list(_PERP_COLUMNS))
        return len(rows)
    finally:
        if owns_client:
            http_client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hyperliquid perp mark snapshots")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--coins", default=None, help="comma-separated; default all")
    args = parser.parse_args(argv)

    from src.ingestion.schemas import get_clickhouse_client

    coins = tuple(args.coins.split(",")) if args.coins else tuple(TICKER_ALIASES)
    print(f"perp_marks inserted: {snapshot_perp_marks(get_clickhouse_client(), coins=coins)}")


if __name__ == "__main__":
    main()
