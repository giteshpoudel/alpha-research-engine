import numpy as np
import pandas as pd

from src.research import batch4 as b4


def test_hold_weights_fixed_hold_no_overlap():
    idx = pd.date_range("2025-01-01", periods=10, freq="1D")
    sig = pd.Series([0, 1, 1, 0, 1, 0, 0, 0, 0, 0], index=idx).astype(bool)
    w = b4.hold_weights(sig, pd.Series(0.5, index=idx), 3)
    assert list(w) == [0, .5, .5, .5, .5, .5, .5, 0, 0, 0]


def test_signals_do_not_use_the_future():
    idx = pd.date_range("2024-01-01", periods=300, freq="1D")
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300))), index=idx)
    fund = pd.Series(rng.normal(0, 1e-5, 300), index=idx)
    a = b4.dip_signal(close, 50, 3, 1.5)
    f = b4.funding_signal(fund, 3, 10)
    close2, fund2 = close.copy(), fund.copy()
    close2.iloc[200:] *= 0.5
    fund2.iloc[200:] -= 1
    pd.testing.assert_series_equal(a.iloc[:200], b4.dip_signal(close2, 50, 3, 1.5).iloc[:200])
    pd.testing.assert_series_equal(f.iloc[:200], b4.funding_signal(fund2, 3, 10).iloc[:200])


def _tape(prices, dt=10.0, qty=1000.0):
    return pd.DataFrame({"t": np.arange(len(prices)) * dt, "price": prices, "qty": qty})


def test_mm_fills_only_when_trades_print_through():
    flat = [1.0] * 200
    assert b4.simulate_mm(_tape(flat), 0.001, 1, 0.0)["fills"] == 0       # touching never fills
    r = b4.simulate_mm(_tape(flat + [0.995] + flat), 0.001, 1, 0.0)
    assert r["buys"] == 1 and r["sells"] == 0


def test_mm_round_trip_earns_the_spread_and_respects_inventory_cap():
    flat = [1.0] * 100
    r = b4.simulate_mm(_tape(flat + [0.99, 0.99, 0.99] + flat + [1.01] + flat), 0.002, 1, 0.0)
    assert r["buys"] == 1                    # capped at one lot long
    assert r["sells"] == 1 and r["pnl"] > 0


def test_lagged_fair_ignores_trades_newer_than_delay():
    t = np.array([0.0, 10.0, 11.0])
    p = np.array([1.0, 1.0, 2.0])
    fair = b4.lagged_fair(t, p)
    assert fair[2] == 1.0                    # the 2.0 print at t=11 is not visible at t=11
