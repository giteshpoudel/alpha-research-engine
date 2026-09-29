import numpy as np
import pandas as pd
import pytest

from src.meta_learning.cross_sectional import (
    apply_vol_target,
    backtest_cross_sectional,
    select_oos,
)

IDX = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")


def test_momentum_buys_the_trending_up_asset():
    # A rises steadily; B and C fall. Long-only top-1 should be profitable.
    prices = pd.DataFrame({
        "A": np.linspace(100, 200, len(IDX)),
        "B": np.linspace(100, 60, len(IDX)),
        "C": np.linspace(100, 80, len(IDX)),
    }, index=IDX)
    result = backtest_cross_sectional(prices, lookback=24, hold=24, top_k=1, fee=0.0)
    assert result.total_return > 0


def test_no_position_before_lookback():
    prices = pd.DataFrame({"A": np.ones(len(IDX)), "B": np.ones(len(IDX))}, index=IDX)
    result = backtest_cross_sectional(prices, lookback=48, hold=24, top_k=1, fee=0.0)
    assert result.total_return == 0.0
    assert result.equity_curve.iloc[0] == 1.0


def test_long_short_beats_long_only_on_dispersion():
    prices = pd.DataFrame({
        "A": np.linspace(100, 250, len(IDX)),
        "B": np.linspace(100, 40, len(IDX)),
    }, index=IDX)
    long_only = backtest_cross_sectional(prices, lookback=24, hold=24, top_k=1, fee=0.0)
    long_short = backtest_cross_sectional(prices, lookback=24, hold=24, top_k=1,
                                          fee=0.0, long_short=True)
    assert long_short.total_return > long_only.total_return


def test_insufficient_columns_raises():
    prices = pd.DataFrame({"A": np.ones(len(IDX))}, index=IDX)
    with pytest.raises(ValueError):
        backtest_cross_sectional(prices, top_k=2)


def test_vol_target_scales_down_and_is_causal():
    rng = np.random.default_rng(0)
    net = pd.Series(rng.normal(0.0, 0.05, 2000), index=pd.date_range(
        "2024-01-01", periods=2000, freq="1h", tz="UTC"))
    scaled = apply_vol_target(net, target_annual=0.2, vol_window=168,
                              periods_per_year=8760)
    assert scaled.abs().mean() < net.abs().mean()
    assert scaled.iloc[0] == 0.0  # no exposure before history exists


def test_select_oos_picks_from_grid_and_evaluates_out_of_sample():
    prices = pd.DataFrame({
        "A": np.linspace(100, 200, len(IDX)),
        "B": np.linspace(100, 120, len(IDX)),
        "C": np.linspace(100, 60, len(IDX)),
    }, index=IDX)
    grid = [{"lookback": 24, "hold": 24, "top_k": 1},
            {"lookback": 72, "hold": 24, "top_k": 2}]
    best, is_sharpe, oos = select_oos(prices, IDX[250], grid, fee=0.0)
    assert best in grid
    assert np.isfinite(is_sharpe)
    assert np.isfinite(oos.total_return)
