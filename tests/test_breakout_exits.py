import numpy as np
import pandas as pd

from src.research.breakout_exits import breakout_exit, decide, trades
from src.research.intraday_strategies import vol_breakout


def _prices(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * (1 + rng.uniform(0, 0.005, n)), c * (1 - rng.uniform(0, 0.005, n))
    return h, l, c


def test_baseline_matches_original_rule():
    h, l, c = _prices()
    for p in (dict(N=24, vratio=1.0, k=3.0, T=24), dict(N=48, vratio=1.3, k=2.0, T=96)):
        orig = vol_breakout(c, h, l, c, np.ones_like(c), **p)
        assert np.array_equal(breakout_exit(h, l, c, **p), orig)


def test_tighter_exits_never_hold_longer():
    h, l, c = _prices(1)
    p = dict(N=24, vratio=1.0, k=3.0, T=48)
    base = breakout_exit(h, l, c, **p)
    for v in (1.5, 1.0, "lock"):
        x = breakout_exit(h, l, c, **p, variant=v)
        assert x.sum() <= base.sum()
        assert x.sum() > 0
        # same entries: every entry in the variant is an entry signal of the rule
        ent = np.flatnonzero((x == 1) & (np.r_[0, x[:-1]] == 0))
        assert all(base[e] == 1 or base[e - 1] == 0 for e in ent)


def test_lock_exits_after_half_giveback():
    # up 1 ATR-ish then fall back: lock exits before the 3-ATR stop would
    c = np.r_[np.tile([100.0, 100.1], 15), 101, 103, 105, 104, 103, 102.4, 102.0, 101, 100]
    h, l = c + 0.25, c - 0.25
    p = dict(N=4, vratio=0.0, k=3.0, T=100)
    base = breakout_exit(h, l, c, **p)
    lock = breakout_exit(h, l, c, **p, variant="lock")
    assert base[30] == 1 and lock[30] == 1
    assert lock.sum() < base.sum()


def test_trade_stats():
    idx = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    held = pd.DataFrame({"X": [0, 0.5, 0.5, 0, 0, 0.5, 0, 0]}, index=idx)
    r = pd.DataFrame({"X": [0, 0.10, 0.0, 0, 0, -0.05, 0, 0]}, index=idx)
    sc = pd.DataFrame({"X": [0.001] * 8}, index=idx)
    t = trades(held, r, sc, idx[0], idx[-1] + pd.Timedelta(hours=1))
    assert t["trades"] == 2 and t["win_rate"] == 0.5
    assert abs(t["avg_win"] - (0.10 - 0.002)) < 1e-12
    assert abs(t["avg_loss"] - (-0.05 - 0.002)) < 1e-12


def test_decide_requires_all_checks():
    def row(ret, sh, c2, pre):
        return {"oos_2025_26": {"return": ret, "sharpe": sh, "costs_x2_return": c2},
                "pre_2022_24": {"return": pre}}
    res = {"baseline": row(0.10, 0.5, 0.05, 0.0), "tight15": row(0.2, 0.7, 0.06, 0.01),
           "tight10": row(0.3, 0.9, 0.04, 0.1), "lock": row(0.2, 0.55, 0.1, 0.1)}
    checks, winner = decide(res)
    assert checks["tight15"]["pass"] and not checks["tight10"]["pass"] and not checks["lock"]["pass"]
    assert winner == "tight15"
