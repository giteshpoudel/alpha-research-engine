from datetime import datetime, timezone

import pytest

from src.ingestion import ws_collector as ws

TS = datetime(2026, 11, 2, tzinfo=timezone.utc)


def test_venue_label():
    assert ws.venue_label("binance_us", "USDT") == "binance_us"
    assert ws.venue_label("binance_us", "USD") == "binance_us:USD"
    assert ws.venue_label("kraken", "USD") == "kraken:USD"


def test_book_row_metrics():
    row = ws.book_row("kraken:USD", "BTC", TS,
                      bids=[(100.0, 3.0), (99.0, 1.0)], asks=[(101.0, 1.0), (102.0, 2.0)])
    assert row["spread_bps"] == pytest.approx(1 / 100.5 * 1e4)
    # microprice leans toward the thin side (ask has less size -> closer to ask)
    assert row["microprice"] == pytest.approx((100 * 1 + 101 * 3) / 4)
    assert row["bid_depth5"] == 4.0 and row["ask_depth10"] == 3.0


def test_book_row_rejects_empty_and_crossed():
    assert ws.book_row("x", "BTC", TS, [], [(1.0, 1.0)]) is None
    assert ws.book_row("x", "BTC", TS, [(101.0, 1.0)], [(100.0, 1.0)]) is None


def test_parse_binance_trade_and_book():
    kind, pair, _, row = ws.parse_binance("btcusd@aggTrade", {
        "a": 21268661, "p": "76764.09", "q": "0.00066", "T": 1793653908794, "m": False})
    assert (kind, pair) == ("trade", "BTCUSD")
    assert row["trade_id"] == 21268661 and row["is_buyer_maker"] == 0
    kind, pair, _, (bids, asks) = ws.parse_binance(
        "ethusdt@depth10@1000ms", {"bids": [["10.0", "2.0"]], "asks": [["10.5", "1.0"]]})
    assert (kind, pair) == ("book", "ETHUSDT") and bids == [(10.0, 2.0)]
    assert ws.parse_binance("btcusd@kline_1m", {}) is None


def test_parse_kraken_trade_aggressor_side():
    row = ws.parse_kraken_trade({"trade_id": 7, "price": 1.5, "qty": 2, "side": "sell",
                                 "timestamp": "2026-11-02T21:11:39.794835Z"})
    assert row["is_buyer_maker"] == 1  # taker sold into a resting bid
    assert row["ts"].tzinfo is not None


def test_kraken_book_snapshot_update_and_truncate():
    book = ws.Book(depth=2)
    book.apply([{"price": 100, "qty": 1}, {"price": 99, "qty": 1}],
               [{"price": 101, "qty": 1}, {"price": 102, "qty": 1}], snapshot=True)
    book.apply([{"price": 100, "qty": 0}, {"price": 98, "qty": 5}], [])  # level removed
    book.apply([{"price": 99.5, "qty": 2}], [{"price": 100.5, "qty": 1}])
    bids, asks = book.top()
    assert bids == [(99.5, 2.0), (99.0, 1.0)]  # 98 pushed beyond depth 2
    assert asks == [(100.5, 1.0), (101.0, 1.0)]
    assert not book.crossed()
    book.apply([{"price": 101.0, "qty": 1}], [])
    assert book.crossed()


def test_gap_tracker():
    gaps = ws.GapTracker()
    assert gaps.check("k", "BTC", 10) is None
    assert gaps.check("k", "BTC", 11) is None
    assert gaps.check("k", "BTC", 11) is None  # replayed duplicate
    gap = gaps.check("k", "BTC", 15)
    assert (gap["from_id"], gap["to_id"], gap["missing"]) == (12, 14, 3)
    assert gaps.check("k", "ETH", 1) is None  # tracked per symbol
