import numpy as np
import pandas as pd
import pytest

from src.research import intraday_strategies as strat
from src.research import intraday_study as study


def _df(close, vol=None):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2023-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": vol if vol is not None else np.ones(len(close))},
                        index=idx)


def test_run_applies_one_bar_lag_and_costs():
    df = _df([100, 100, 110, 121, 121])
    pos = np.array([1, 1, 1, 0, 0], dtype=float)  # decided at close of bar 0
    r = study.run(df, pos, side_cost=0.001, stress=np.zeros(5, bool))
    # enters at open of bar 1 (100); held bars 1..3 earn open->open 100->110->121->121
    assert r.net.iloc[0] == 0
    assert r.net.iloc[1] == pytest.approx(0.10 - 0.001)  # entry cost
    assert r.net.iloc[2] == pytest.approx(0.10)
    assert r.net.iloc[3] == pytest.approx(0.0)
    assert r.net.iloc[4] == pytest.approx(-0.001)        # exit at open of bar 4
    assert len(r.trades) == 1
    assert r.trades[0] == pytest.approx(1.21 * (1 - 0.001) ** 2 - 1)


def test_stress_multiplies_cost_using_prior_bar_flag():
    df = _df([100] * 4)
    pos = np.array([1, 0, 0, 0], dtype=float)
    stress = np.array([True, False, False, False])
    r = study.run(df, pos, side_cost=0.001, stress=stress, stress_mult=3.0)
    assert r.net.iloc[1] == pytest.approx(-0.003)  # entry at bar 1, flag known at bar 0
    assert r.net.iloc[2] == pytest.approx(-0.001)


def test_no_lookahead_signal_change_at_last_bar_has_no_effect():
    df = _df([100, 101, 102, 103])
    a = study.run(df, np.array([0, 0, 0, 0.0]), 0.0, np.zeros(4, bool)).net
    b = study.run(df, np.array([0, 0, 0, 1.0]), 0.0, np.zeros(4, bool)).net
    pd.testing.assert_series_equal(a, b)


def test_vol_breakout_enters_on_breakout_and_exits_on_trail():
    rng = np.random.default_rng(0)
    calm = 100 + rng.normal(0, 0.05, 200).cumsum() * 0.1
    surge = calm[-1] * np.cumprod(np.full(30, 1.01))
    crash = surge[-1] * np.cumprod(np.full(30, 0.98))
    c = np.concatenate([calm, surge, crash])
    df = _df(c)
    pos = strat.vol_breakout(*[df[k].to_numpy() for k in ("open", "high", "low", "close", "volume")],
                             N=24, vratio=1.0, k=2.0, T=200)
    assert pos[:200].sum() == 0 or pos[:200].mean() < 0.2
    assert pos[205:230].mean() > 0.8      # in during the surge
    assert pos[-1] == 0                   # trailed out in the crash


def test_overshoot_fade_buys_capitulation_on_volume_and_takes_profit():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 0.2, 120).cumsum() * 0.2
    flush = [c[-1] * 0.90]                                   # one-bar 10% flush
    rebound = flush[0] * np.cumprod(np.full(12, 1.01))
    c = np.concatenate([c, flush, rebound])
    vol = np.ones(len(c))
    vol[120] = 20.0
    df = _df(c, vol)
    pos = strat.overshoot_fade(*[df[k].to_numpy() for k in ("open", "high", "low", "close", "volume")],
                               L=4, m=3.0, z=1.0, H=48)
    assert pos[:120].sum() == 0
    assert pos[120] == 1                  # bought the flush bar's close
    assert pos[-1] == 0                   # exited on the half-recovery target


def test_overshoot_fade_stops_out_of_a_falling_knife():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 0.2, 120).cumsum() * 0.2
    c = np.concatenate([c, c[-1] * np.cumprod(np.full(6, 0.97))])
    vol = np.ones(len(c))
    vol[120:126] = 20.0
    df = _df(c, vol)
    pos = strat.overshoot_fade(*[df[k].to_numpy() for k in ("open", "high", "low", "close", "volume")],
                               L=4, m=3.0, z=1.0, H=48)
    assert pos[120] == 1 and pos[121] == 0


def test_grids_match_preregistration():
    assert len(strat.GRIDS["vol_breakout"]) == 24
    assert len(strat.GRIDS["overshoot_fade"]) == 16
    assert len(study.BARS) * 40 * len(study.SYMBOLS) == study.N_TRIALS == 240


def test_costs_match_preregistration():
    assert study.cost_per_side("BTC") == pytest.approx(5e-4)
    assert study.cost_per_side("SOL") == pytest.approx(6e-4)
    assert study.cost_per_side("ETH", mult=2.0) == pytest.approx(10e-4)
