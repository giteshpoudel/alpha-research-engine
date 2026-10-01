from datetime import datetime, timezone

import pandas as pd
import pytest

from src.ingestion.hyperliquid import map_asset_ctxs
from src.meta_learning.basis import basis_stats


def test_map_asset_ctxs_filters_and_maps():
    payload = [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "ZZZ"}]},
        [{"markPx": "100", "oraclePx": "99", "midPx": "100.5", "openInterest": "12"},
         {"markPx": "50", "oraclePx": "49.5", "midPx": "50.1", "openInterest": "34"},
         {"markPx": "1", "oraclePx": "1", "midPx": "1", "openInterest": "1"}],
    ]
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = map_asset_ctxs(payload, coins=("BTC", "ETH"), ts=ts)
    assert [r["symbol"] for r in rows] == ["BTC", "ETH"]
    assert rows[0]["mark_price"] == 100.0 and rows[0]["mid_price"] == 100.5
    assert rows[1]["open_interest"] == 34.0


def test_basis_stats_shape():
    idx = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    basis = pd.Series([0.001] * 100, index=idx)
    stats = basis_stats(basis)
    assert stats["n"] == 100
    assert stats["mean"] == pytest.approx(0.001)
    assert stats["current"] == pytest.approx(0.001)
    assert basis_stats(pd.Series(dtype=float)) == {"n": 0}
