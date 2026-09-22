"""Symbol classification: price bucket, category, and pump-and-dump risk.

Rule-based with an optional per-symbol override. This is the metadata the
optimizer uses to cluster the universe (majors vs alts vs memes; cheap vs
expensive) and to focus intraday vs swing strategies.

Usage:
    python -m src.agents.symbols
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES

# Ordered (upper_bound, label); the last bucket is the fallback.
PRICE_BUCKETS: tuple[tuple[float, str], ...] = ((2.0, "$0-2"), (20.0, "$2-20"))
TOP_BUCKET = "$20+"

# Static defaults; an override map (symbol -> category) wins over these.
MEME_SYMBOLS = frozenset({
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "BRETT", "TRUMP",
})
MAJOR_SYMBOLS = frozenset({"BTC", "ETH", "BNB"})

_METADATA_COLUMNS = ("symbol", "price", "price_bucket", "category", "ann_vol",
                     "pump_risk", "updated_at")


def price_bucket(price: float) -> str:
    for upper, label in PRICE_BUCKETS:
        if price < upper:
            return label
    return TOP_BUCKET


def category(symbol: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and symbol in overrides:
        return overrides[symbol]
    if symbol in MEME_SYMBOLS:
        return "meme"
    if symbol in MAJOR_SYMBOLS:
        return "major"
    return "alt"


def pump_risk(category: str, ann_vol: float) -> str:
    if category == "meme" or ann_vol >= 0.90:
        return "high"
    if ann_vol >= 0.70:
        return "medium"
    return "low"


def classify(symbol: str, price: float, ann_vol: float,
             overrides: dict[str, str] | None = None) -> dict:
    cat = category(symbol, overrides)
    return {
        "symbol": symbol,
        "price": float(price),
        "price_bucket": price_bucket(price),
        "category": cat,
        "ann_vol": float(ann_vol),
        "pump_risk": pump_risk(cat, ann_vol),
    }


def classify_universe(ch_client, symbols: tuple[str, ...] | None = None,
                      overrides: dict[str, str] | None = None) -> list[dict]:
    rows = []
    for symbol in (symbols or tuple(TICKER_ALIASES)):
        try:
            close = load_ohlcv(ch_client, symbol)["close"]
        except ValueError:
            continue
        ann_vol = float(close.pct_change().dropna().std() * np.sqrt(24 * 365))
        rows.append(classify(symbol, float(close.iloc[-1]), ann_vol, overrides))
    return rows


def store_metadata(ch_client, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [[r["symbol"], r["price"], r["price_bucket"], r["category"],
             r["ann_vol"], r["pump_risk"], now] for r in rows]
    ch_client.insert(f"{database_name()}.symbol_metadata", data,
                     column_names=list(_METADATA_COLUMNS))
    return len(rows)


def load_metadata(ch_client) -> list[dict]:
    rows = ch_client.query(
        f"SELECT symbol, price, price_bucket, category, ann_vol, pump_risk, updated_at "
        f"FROM {database_name()}.symbol_metadata FINAL ORDER BY symbol"
    ).result_rows
    return [dict(zip(_METADATA_COLUMNS, row)) for row in rows]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify universe symbols")
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args(argv)
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))

    from src.ingestion.schemas import get_clickhouse_client
    ch = get_clickhouse_client()
    rows = classify_universe(ch, symbols)
    if not args.no_store:
        store_metadata(ch, rows)
    print(f"{'sym':<6}{'price':>12}{'bucket':>8}{'category':>9}{'vol':>7}{'risk':>8}")
    for r in rows:
        print(f"{r['symbol']:<6}{r['price']:>12.4f}{r['price_bucket']:>8}"
              f"{r['category']:>9}{r['ann_vol']:>7.2f}{r['pump_risk']:>8}")


if __name__ == "__main__":
    main()
