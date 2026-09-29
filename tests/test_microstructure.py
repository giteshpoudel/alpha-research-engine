from datetime import datetime, timezone

import pandas as pd

from src.ingestion.microstructure import map_depth, map_trade, order_flow_imbalance


def test_map_depth_computes_spread_and_imbalance():
    payload = {"bids": [["100.0", "2.0"], ["99.9", "1.0"]],
               "asks": [["100.1", "1.0"], ["100.2", "0.5"]]}
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    row = map_depth("BTC", payload, ts=ts)
    assert row["best_bid"] == 100.0 and row["best_ask"] == 100.1
    assert row["bid_depth"] == 3.0 and row["ask_depth"] == 1.5
    assert row["imbalance"] == (3.0 - 1.5) / 4.5
    assert row["spread_bps"] > 0


def test_map_trade_direction_flag():
    raw = {"a": 7, "p": "100.5", "q": "0.3", "T": 1735689600000, "m": True}
    row = map_trade("BTC", raw)
    assert row["trade_id"] == 7 and row["price"] == 100.5
    assert row["is_buyer_maker"] == 1  # aggressor was the seller


def test_order_flow_imbalance_sign():
    trades = pd.DataFrame({"qty": [2.0, 1.0], "is_buyer_maker": [0, 1]})
    assert order_flow_imbalance(trades) == (2.0 - 1.0) / 3.0
    buys = pd.DataFrame({"qty": [1.0, 1.0], "is_buyer_maker": [0, 0]})
    assert order_flow_imbalance(buys) == 1.0
    assert order_flow_imbalance(pd.DataFrame({"qty": [], "is_buyer_maker": []})) == 0.0
