"""Backfill historical trades into market_trades (same rows as the websocket collector).

    PYTHONPATH=. python -m src.ingestion.trade_backfill --days 90
    PYTHONPATH=. python -m src.ingestion.trade_backfill --days 90 --venues kraken --symbols BTC

Order-book snapshots can't be recovered from the past, but every trade can, so
trade-flow features (signed volume, aggressor imbalance, trade bursts) get months of
history now instead of waiting for the live tape.

- Kraken: /0/public/Trades (1,000 trades per call, cursor ``last``; ~1 call/s).
- Binance.US: /api/v3/aggTrades (``fromId`` pagination), USDT and USD pairs.

Resumable: progress per (venue, symbol) is saved to data/trade_backfill_state.json.
Rows dedupe with the live collector on (exchange, symbol, trade_id).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.ws_collector import venue_label

STATE = Path("data/trade_backfill_state.json")
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
COLUMNS = ["exchange", "symbol", "trade_id", "ts", "price", "qty", "is_buyer_maker"]
BATCH = 20_000


# --- pure parsers (unit-tested) -------------------------------------------------

def kraken_rows(symbol: str, trades: list[list]) -> list[list]:
    """REST rows [price, volume, time, side, type, misc, trade_id]; side = taker side."""
    return [[venue_label("kraken", "USD"), symbol, int(t[6]),
             datetime.fromtimestamp(float(t[2]), tz=timezone.utc),
             float(t[0]), float(t[1]), 1 if t[3] == "s" else 0] for t in trades]


def binance_rows(exchange: str, symbol: str, trades: list[dict]) -> list[list]:
    return [[exchange, symbol, int(t["a"]), datetime.fromtimestamp(t["T"] / 1000, tz=timezone.utc),
             float(t["p"]), float(t["q"]), 1 if t["m"] else 0] for t in trades]


# --- state ----------------------------------------------------------------------

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


# --- fetchers -------------------------------------------------------------------

def _get(http: httpx.Client, url: str, params: dict, pause: float) -> dict | list:
    for attempt in range(8):
        try:
            resp = http.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {resp.status_code}")
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                if any("Too many" in e or "Busy" in e for e in data["error"]):
                    raise httpx.HTTPError(str(data["error"]))
                raise RuntimeError(str(data["error"]))
            time.sleep(pause)
            return data
        except (httpx.HTTPError, ValueError) as exc:
            wait = min(60, 2 ** attempt * pause * 2)
            print(f"  retry in {wait:.0f}s: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def backfill_kraken(ch, http, symbol: str, start: datetime, end: datetime, state: dict) -> None:
    key = f"kraken:{symbol}"
    cursor = state.get(key, {}).get("cursor") or str(int(start.timestamp() * 1e9))
    end_ns = int(end.timestamp() * 1e9)
    buf, total = [], state.get(key, {}).get("rows", 0)
    while int(cursor) < end_ns:
        data = _get(http, "https://api.kraken.com/0/public/Trades",
                    {"pair": KRAKEN_PAIRS[symbol], "since": cursor, "count": 1000}, pause=1.1)
        result = data["result"]
        trades = next(v for k, v in result.items() if k != "last")
        if not trades:
            break
        buf += kraken_rows(symbol, trades)
        cursor = result["last"]
        if len(buf) >= BATCH:
            total = _flush(ch, buf, state, key, cursor, total)
            buf = []
            print(f"  {key}: {total:,} rows, at {buf_ts(cursor)}", flush=True)
    _flush(ch, buf, state, key, cursor, total, done=True)


def backfill_binance(ch, http, symbol: str, quote: str, start: datetime, end: datetime,
                     state: dict) -> None:
    exchange = venue_label("binance_us", quote)
    key = f"{exchange}:{symbol}"
    url, pair = "https://api.binance.us/api/v3/aggTrades", f"{symbol}{quote}"
    end_ms = int(end.timestamp() * 1000)
    from_id = state.get(key, {}).get("cursor")
    if from_id is None:  # find the first trade at/after start
        first = _get(http, url, {"symbol": pair, "startTime": int(start.timestamp() * 1000),
                                 "endTime": int(start.timestamp() * 1000) + 3_600_000,
                                 "limit": 1}, pause=0.2)
        if not first:
            # nothing in the first hour: walk forward an hour at a time
            t = start
            while not first and t < end:
                t += timedelta(hours=1)
                first = _get(http, url, {"symbol": pair, "startTime": int(t.timestamp() * 1000),
                                         "endTime": int(t.timestamp() * 1000) + 3_600_000,
                                         "limit": 1}, pause=0.2)
        if not first:
            return
        from_id = first[0]["a"]
    buf, total = [], state.get(key, {}).get("rows", 0)
    while True:
        page = _get(http, url, {"symbol": pair, "fromId": from_id, "limit": 1000}, pause=0.2)
        page = [t for t in page if t["T"] < end_ms]
        if not page:
            break
        buf += binance_rows(exchange, symbol, page)
        from_id = page[-1]["a"] + 1
        if len(buf) >= BATCH:
            total = _flush(ch, buf, state, key, from_id, total)
            buf = []
            print(f"  {key}: {total:,} rows, at {datetime.fromtimestamp(page[-1]['T'] / 1000, timezone.utc):%F %H:%M}",
                  flush=True)
        if len(page) < 1000:
            break
    _flush(ch, buf, state, key, from_id, total, done=True)


def buf_ts(cursor_ns: str) -> str:
    return f"{datetime.fromtimestamp(int(cursor_ns) / 1e9, timezone.utc):%F %H:%M}"


def _flush(ch, buf, state, key, cursor, total, done: bool = False) -> int:
    if buf:
        ch.insert(f"{database_name()}.market_trades", buf, column_names=COLUMNS)
    total += len(buf)
    state[key] = {"cursor": cursor, "rows": total, "done": done,
                  "updated": datetime.now(timezone.utc).isoformat()}
    save_state(state)
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill historical trades")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--venues", nargs="+", default=["binance_us", "kraken"])
    args = parser.parse_args(argv)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    ch, http, state = get_clickhouse_client(), httpx.Client(timeout=30), load_state()
    for venue in args.venues:
        for symbol in args.symbols:
            if venue == "kraken":
                if state.get(f"kraken:{symbol}", {}).get("done"):
                    continue
                print(f"kraken {symbol}: {start:%F} -> {end:%F}", flush=True)
                backfill_kraken(ch, http, symbol, start, end, state)
            else:
                for quote in ("USDT", "USD"):
                    if state.get(f"{venue_label(venue, quote)}:{symbol}", {}).get("done"):
                        continue
                    print(f"binance_us {symbol}{quote}: {start:%F} -> {end:%F}", flush=True)
                    backfill_binance(ch, http, symbol, quote, start, end, state)
    print("backfill complete", {k: v["rows"] for k, v in state.items()}, flush=True)


if __name__ == "__main__":
    main()
