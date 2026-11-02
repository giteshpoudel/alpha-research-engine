"""Websocket microstructure collector: full trade tape + 1-second book samples.

Replaces the 5-minute REST poller, which saw only the last 1,000 trades per run
(gaps on busy pairs) and one book snapshot every 5 minutes. This keeps a live
connection to each venue and records:

- every public trade -> ``market_trades`` (idempotent on venue trade id)
- a top-of-book + depth sample per instrument per second -> ``book_samples``
- any break in a venue's sequential trade ids -> ``tape_gaps``

Venues: Binance.US (USDT and USD pairs) and Kraken (USD). The ``exchange`` column
carries the quote when it isn't the venue default: ``binance_us`` (USDT, same as
the REST poller), ``binance_us:USD`` and ``kraken:USD``. Research code compares
venues on this data before we pick where to trade live.

Usage:
    python -m src.ingestion.ws_collector                 # run forever
    python -m src.ingestion.ws_collector --seconds 60 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import websockets

from src.ingestion.schemas import database_name, get_clickhouse_client

DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "LINK", "LTC")
BINANCE_URL = "wss://stream.binance.us:9443/stream?streams="
KRAKEN_URL = "wss://ws.kraken.com/v2"
BOOK_DEPTH = 10
SAMPLE_SECONDS = 1.0
FLUSH_SECONDS = 2.0
STALE_SECONDS = 60.0  # reconnect if a connection goes quiet this long

TRADE_COLUMNS = ("exchange", "symbol", "trade_id", "ts", "price", "qty", "is_buyer_maker")
BOOK_COLUMNS = ("exchange", "symbol", "ts", "bid_px", "bid_qty", "ask_px", "ask_qty",
                "bid_depth5", "ask_depth5", "bid_depth10", "ask_depth10",
                "spread_bps", "microprice")
GAP_COLUMNS = ("exchange", "symbol", "detected_at", "from_id", "to_id", "missing")


def venue_label(venue: str, quote: str) -> str:
    """``binance_us`` for its default USDT quote, else ``venue:QUOTE``."""
    return venue if (venue, quote) == ("binance_us", "USDT") else f"{venue}:{quote}"


# --- pure transforms (unit-tested) ------------------------------------------

def book_row(exchange: str, symbol: str, ts: datetime,
             bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> dict | None:
    """One book sample. ``bids`` best-first (descending), ``asks`` ascending."""
    if not bids or not asks:
        return None
    (bid_px, bid_qty), (ask_px, ask_qty) = bids[0], asks[0]
    if bid_px <= 0 or ask_px <= bid_px:  # empty or crossed book: skip, don't store junk
        return None
    mid = (bid_px + ask_px) / 2.0
    size = bid_qty + ask_qty
    micro = (bid_px * ask_qty + ask_px * bid_qty) / size if size > 0 else mid
    return {
        "exchange": exchange, "symbol": symbol, "ts": ts,
        "bid_px": bid_px, "bid_qty": bid_qty, "ask_px": ask_px, "ask_qty": ask_qty,
        "bid_depth5": sum(q for _, q in bids[:5]), "ask_depth5": sum(q for _, q in asks[:5]),
        "bid_depth10": sum(q for _, q in bids[:10]), "ask_depth10": sum(q for _, q in asks[:10]),
        "spread_bps": (ask_px - bid_px) / mid * 1e4, "microprice": micro,
    }


def parse_binance(stream: str, data: dict) -> tuple[str, str, str, object] | None:
    """Combined-stream message -> (kind, pair, quote, payload).

    kind ``trade`` -> trade row dict; kind ``book`` -> (bids, asks).
    """
    pair = stream.split("@", 1)[0].upper()
    if "@aggTrade" in stream:
        return ("trade", pair, "", {
            "trade_id": int(data["a"]),
            "ts": datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
            "price": float(data["p"]), "qty": float(data["q"]),
            "is_buyer_maker": 1 if data["m"] else 0,
        })
    if "@depth" in stream:
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
        return ("book", pair, "", (bids, asks))
    return None


def parse_kraken_trade(item: dict) -> dict:
    """Kraken v2 trade -> trade row. ``side`` is the taker (aggressor) side."""
    ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
    return {"trade_id": int(item["trade_id"]), "ts": ts, "price": float(item["price"]),
            "qty": float(item["qty"]), "is_buyer_maker": 1 if item["side"] == "sell" else 0}


@dataclass
class Book:
    """Local L2 book maintained from Kraken snapshot + incremental updates."""

    depth: int = BOOK_DEPTH
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    def apply(self, bids: list[dict], asks: list[dict], snapshot: bool = False) -> None:
        if snapshot:
            self.bids.clear()
            self.asks.clear()
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for level in levels:
                price, qty = float(level["price"]), float(level["qty"])
                if qty == 0:
                    side.pop(price, None)
                else:
                    side[price] = qty
        # Kraken only maintains the subscribed depth; drop levels pushed past it.
        for price in sorted(self.bids, reverse=True)[self.depth:]:
            del self.bids[price]
        for price in sorted(self.asks)[self.depth:]:
            del self.asks[price]

    def top(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        return (sorted(self.bids.items(), reverse=True)[: self.depth],
                sorted(self.asks.items())[: self.depth])

    def crossed(self) -> bool:
        return bool(self.bids and self.asks) and max(self.bids) >= min(self.asks)


class GapTracker:
    """Detects breaks in a venue's sequential trade ids."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, str], int] = {}

    def check(self, exchange: str, symbol: str, trade_id: int) -> dict | None:
        key = (exchange, symbol)
        last = self._last.get(key)
        if last is not None and trade_id <= last:
            return None  # duplicate / replay after reconnect: ignore
        self._last[key] = trade_id
        if last is None or trade_id == last + 1:
            return None
        return {"exchange": exchange, "symbol": symbol,
                "detected_at": datetime.now(timezone.utc),
                "from_id": last + 1, "to_id": trade_id - 1, "missing": trade_id - last - 1}


# --- I/O ---------------------------------------------------------------------

class Sink:
    """Buffers rows and flushes them to ClickHouse in batches off the event loop."""

    def __init__(self, client, dry_run: bool = False) -> None:
        self.client = client
        self.dry_run = dry_run
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, table: str, row: dict) -> None:
        self.buffers[table].append(row)
        self.counts[table] += 1

    def _insert(self, table: str, columns: tuple, rows: list[dict]) -> None:
        self.client.insert(f"{database_name()}.{table}",
                           [[r[c] for c in columns] for r in rows],
                           column_names=list(columns))

    async def flush(self) -> None:
        batches = {t: rows for t, rows in self.buffers.items() if rows}
        self.buffers = defaultdict(list)
        if self.dry_run:
            return
        columns = {"market_trades": TRADE_COLUMNS, "book_samples": BOOK_COLUMNS,
                   "tape_gaps": GAP_COLUMNS}
        for table, rows in batches.items():
            try:
                await asyncio.to_thread(self._insert, table, columns[table], rows)
            except Exception as exc:  # keep collecting; a failed batch is logged, not fatal
                print(f"ws_collector: insert {table} ({len(rows)} rows) failed: {exc}")


class Collector:
    def __init__(self, sink: Sink, symbols: tuple[str, ...]) -> None:
        self.sink = sink
        self.symbols = symbols
        self.gaps = GapTracker()
        self.kraken_books: dict[str, Book] = {}
        self.last_message: dict[str, float] = {}

    def _trade(self, exchange: str, symbol: str, row: dict) -> None:
        gap = self.gaps.check(exchange, symbol, row["trade_id"])
        if gap:
            self.sink.add("tape_gaps", gap)
        self.sink.add("market_trades", {"exchange": exchange, "symbol": symbol, **row})

    # Binance.US: depth10@1000ms is already a 1s snapshot, so store each one.
    async def binance(self) -> None:
        pairs = {}
        for sym in self.symbols:
            for quote in ("USDT", "USD"):
                pairs[f"{sym}{quote}"] = (sym, venue_label("binance_us", quote))
        streams = "/".join(f"{p.lower()}@aggTrade/{p.lower()}@depth{BOOK_DEPTH}@1000ms"
                           for p in pairs)
        async with websockets.connect(BINANCE_URL + streams, max_size=2**22,
                                      ping_interval=20) as ws:
            print(f"ws_collector: binance_us connected ({len(pairs)} pairs)")
            async for raw in ws:
                self.last_message["binance_us"] = time.monotonic()
                msg = json.loads(raw)
                parsed = parse_binance(msg.get("stream", ""), msg.get("data", {}))
                if not parsed or parsed[1] not in pairs:
                    continue
                kind, pair, _, payload = parsed
                symbol, exchange = pairs[pair]
                if kind == "trade":
                    self._trade(exchange, symbol, payload)
                else:
                    row = book_row(exchange, symbol, datetime.now(timezone.utc), *payload)
                    if row:
                        self.sink.add("book_samples", row)

    async def kraken(self) -> None:
        pairs = {f"{s}/USD": s for s in self.symbols}
        exchange = venue_label("kraken", "USD")
        async with websockets.connect(KRAKEN_URL, max_size=2**22, ping_interval=20) as ws:
            for channel, extra in (("trade", {}), ("book", {"depth": BOOK_DEPTH})):
                await ws.send(json.dumps({"method": "subscribe", "params": {
                    "channel": channel, "symbol": list(pairs), **extra}}))
            print(f"ws_collector: kraken connected ({len(pairs)} pairs)")
            self.kraken_books = {s: Book() for s in pairs.values()}
            async for raw in ws:
                self.last_message["kraken"] = time.monotonic()
                msg = json.loads(raw)
                channel = msg.get("channel")
                if channel == "trade":
                    for item in msg.get("data", []):
                        symbol = pairs.get(item.get("symbol"))
                        if symbol:
                            self._trade(exchange, symbol, parse_kraken_trade(item))
                elif channel == "book":
                    for item in msg.get("data", []):
                        symbol = pairs.get(item.get("symbol"))
                        if not symbol:
                            continue
                        book = self.kraken_books[symbol]
                        book.apply(item.get("bids", []), item.get("asks", []),
                                   snapshot=msg.get("type") == "snapshot")
                        if book.crossed():  # out of sync: force a fresh snapshot
                            raise ConnectionError(f"kraken {symbol} book crossed")

    async def sample_kraken(self) -> None:
        exchange = venue_label("kraken", "USD")
        while True:
            await asyncio.sleep(SAMPLE_SECONDS)
            now = datetime.now(timezone.utc)
            for symbol, book in list(self.kraken_books.items()):
                row = book_row(exchange, symbol, now, *book.top())
                if row:
                    self.sink.add("book_samples", row)

    async def flusher(self) -> None:
        last_report = time.monotonic()
        reported: dict[str, int] = {}
        while True:
            await asyncio.sleep(FLUSH_SECONDS)
            await self.sink.flush()
            if time.monotonic() - last_report >= 60:
                delta = {t: n - reported.get(t, 0) for t, n in self.sink.counts.items()}
                reported = dict(self.sink.counts)
                last_report = time.monotonic()
                print(f"{datetime.now():%F %T} ws_collector: last 60s {delta}", flush=True)


async def _supervise(name: str, factory, collector: Collector) -> None:
    """Run one venue forever: reconnect with backoff, and on a quiet connection."""
    backoff = 1.0
    while True:
        started = time.monotonic()
        collector.last_message[name] = started
        task = asyncio.create_task(factory())
        try:
            while not task.done():
                await asyncio.sleep(5)
                if time.monotonic() - collector.last_message.get(name, 0) > STALE_SECONDS:
                    task.cancel()
                    print(f"ws_collector: {name} quiet for {STALE_SECONDS:.0f}s, reconnecting")
            await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception as exc:
            print(f"ws_collector: {name} disconnected: {exc!r}")
        finally:
            if not task.done():
                task.cancel()
        backoff = 1.0 if time.monotonic() - started > 60 else min(backoff * 2, 60.0)
        await asyncio.sleep(backoff)


async def run(symbols: tuple[str, ...], venues: tuple[str, ...], seconds: float | None,
              dry_run: bool) -> dict:
    sink = Sink(None if dry_run else get_clickhouse_client(), dry_run=dry_run)
    collector = Collector(sink, symbols)
    tasks = [asyncio.create_task(collector.flusher())]
    if "binance_us" in venues:
        tasks.append(asyncio.create_task(_supervise("binance_us", collector.binance, collector)))
    if "kraken" in venues:
        tasks.append(asyncio.create_task(_supervise("kraken", collector.kraken, collector)))
        tasks.append(asyncio.create_task(collector.sample_kraken()))
    try:
        if seconds:
            await asyncio.sleep(seconds)
        else:
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await sink.flush()
    return dict(sink.counts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Websocket trade tape + 1s book samples")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--venues", default="binance_us,kraken")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds (default: run forever)")
    parser.add_argument("--dry-run", action="store_true", help="collect but don't insert")
    args = parser.parse_args(argv)
    counts = asyncio.run(run(tuple(args.symbols.split(",")), tuple(args.venues.split(",")),
                             args.seconds, args.dry_run))
    print(f"ws_collector: totals {counts}")


if __name__ == "__main__":
    main()
