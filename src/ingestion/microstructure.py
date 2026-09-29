"""Binance.US microstructure ingestion: top-of-book snapshots + public trades.

L2 snapshots and the trade tape are only available as *current* data (no
historical endpoint), so this collects forward — the tables fill up over time.
Rows are keyed to be idempotent (snapshot ts, venue trade id).

The immediate research uses are market making (spread/queue/inventory) and
order-flow imbalance (signed trade volume -> short-horizon drift).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import httpx
import pandas as pd

from src.ingestion.http import get_with_backoff
from src.ingestion.schemas import database_name, get_clickhouse_client

_BASE = "https://api.binance.us/api/v3"
_DEPTH_URL = f"{_BASE}/depth"
_AGG_TRADES_URL = f"{_BASE}/aggTrades"
EXCHANGE = "binance_us"

_ORDER_BOOK_COLUMNS = ("exchange", "symbol", "ts", "best_bid", "best_ask",
                       "bid_depth", "ask_depth", "imbalance", "spread_bps")
_TRADE_COLUMNS = ("exchange", "symbol", "trade_id", "ts", "price", "qty",
                  "is_buyer_maker")


def map_depth(symbol: str, payload: dict, ts: datetime | None = None) -> dict:
    """Depth payload -> snapshot row (top-of-book + aggregate imbalance)."""
    bids = [(float(p), float(q)) for p, q in payload.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in payload.get("asks", [])]
    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    bid_depth = sum(q for _, q in bids)
    ask_depth = sum(q for _, q in asks)
    total = bid_depth + ask_depth
    mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
    spread_bps = ((best_ask - best_bid) / mid * 1e4) if mid else 0.0
    return {
        "exchange": EXCHANGE, "symbol": symbol,
        "ts": ts or datetime.now(timezone.utc),
        "best_bid": best_bid, "best_ask": best_ask,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "imbalance": (bid_depth - ask_depth) / total if total else 0.0,
        "spread_bps": spread_bps,
    }


def map_trade(symbol: str, raw: dict) -> dict:
    """aggTrades record -> trade row. ``m`` is is_buyer_maker."""
    return {"exchange": EXCHANGE, "symbol": symbol, "trade_id": int(raw["a"]),
            "ts": datetime.fromtimestamp(raw["T"] / 1000, tz=timezone.utc),
            "price": float(raw["p"]), "qty": float(raw["q"]),
            "is_buyer_maker": 1 if raw["m"] else 0}


def order_flow_imbalance(trades: pd.DataFrame, window: int | None = None) -> float:
    """Signed-volume imbalance: (buy - sell) / total.  ``trades`` needs columns
    qty and is_buyer_maker; optional trailing ``window`` rows."""
    if window is not None:
        trades = trades.tail(window)
    total = float(trades["qty"].sum())
    if total <= 0:
        return 0.0
    # a taker buy has is_buyer_maker == 0 (aggressor lifts the ask)
    buy = float(trades.loc[trades["is_buyer_maker"] == 0, "qty"].sum())
    sell = float(trades.loc[trades["is_buyer_maker"] == 1, "qty"].sum())
    return (buy - sell) / total


def insert_rows(client, table: str, columns: tuple, rows: list[dict]) -> int:
    if not rows:
        return 0
    client.insert(f"{database_name()}.{table}",
                  [[r[c] for c in columns] for r in rows], column_names=list(columns))
    return len(rows)


def snapshot_depth(client, symbols, http_client: httpx.Client | None = None) -> int:
    """Collect one top-of-book snapshot per symbol right now."""
    owns = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for symbol in symbols:
            try:
                resp = get_with_backoff(_DEPTH_URL, client=http_client,
                                        params={"symbol": f"{symbol}USDT", "limit": "100"})
                inserted += insert_rows(client, "order_book_snapshots",
                                        _ORDER_BOOK_COLUMNS,
                                        [map_depth(symbol, resp.json())])
            except Exception as exc:
                print(f"microstructure: depth {symbol} failed: {exc}")
    finally:
        if owns:
            http_client.close()
    return inserted


def harvest_trades(client, symbols, http_client: httpx.Client | None = None,
                   limit: int = 1000) -> int:
    """Harvest the most recent public trades per symbol (forward collection)."""
    owns = http_client is None
    http_client = http_client or httpx.Client()
    inserted = 0
    try:
        for symbol in symbols:
            try:
                resp = get_with_backoff(_AGG_TRADES_URL, client=http_client,
                                        params={"symbol": f"{symbol}USDT", "limit": str(limit)})
                rows = [map_trade(symbol, t) for t in resp.json()]
                inserted += insert_rows(client, "market_trades", _TRADE_COLUMNS, rows)
            except Exception as exc:
                print(f"microstructure: trades {symbol} failed: {exc}")
    finally:
        if owns:
            http_client.close()
    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Binance.US depth + trade ingestion")
    parser.add_argument("--symbols", default="BTC,ETH,SOL,DOGE,XRP,ADA,AVAX,LINK,LTC,BNB,DOT")
    parser.add_argument("--what", choices=["depth", "trades", "both"], default="both")
    args = parser.parse_args(argv)

    client = get_clickhouse_client()
    symbols = tuple(args.symbols.split(","))
    if args.what in ("depth", "both"):
        print(f"depth snapshots inserted: {snapshot_depth(client, symbols)}")
    if args.what in ("trades", "both"):
        print(f"trades inserted: {harvest_trades(client, symbols)}")


if __name__ == "__main__":
    main()
