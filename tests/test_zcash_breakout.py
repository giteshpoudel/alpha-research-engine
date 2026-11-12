import numpy as np
import pandas as pd

from src.research.zcash_breakout import COST, simulate, trade_list


def _idx(n):
    return pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")


def test_simulate_holds_next_bar_and_charges_both_sides():
    idx = _idx(5)
    opens = pd.Series([100, 100, 110, 110, 110], index=idx, dtype=float)
    w = pd.Series([1, 0, 0, 0, 0], index=idx, dtype=float)        # decided at bar 0 close
    net, held, _ = simulate(opens, w, pd.Series(False, index=idx))
    assert held.tolist() == [0, 1, 0, 0, 0]
    assert net.iloc[1] == 0.10 - COST and net.iloc[2] == -COST


def test_stress_triples_cost():
    idx = _idx(3)
    opens = pd.Series([100.0] * 3, index=idx)
    w = pd.Series([1.0, 1.0, 1.0], index=idx)
    net, _, _ = simulate(opens, w, pd.Series([True, False, False], index=idx))
    assert abs(net.iloc[1] + 3 * COST) < 1e-12


def test_trade_list_counts_trades_by_start():
    idx = _idx(8)
    held = pd.Series([0, 1, 1, 0, 0, 1, 0, 0], index=idx, dtype=float)
    net = pd.Series([0, 0.1, 0.0, -0.001, 0, -0.05, -0.001, 0], index=idx)
    tr = trade_list(net, held, idx[0], idx[4])
    assert len(tr) == 1 and abs(tr[0] - (1.1 * 0.999 - 1)) < 1e-12
    assert len(trade_list(net, held, idx[0], idx[-1] + pd.Timedelta("1h"))) == 2
