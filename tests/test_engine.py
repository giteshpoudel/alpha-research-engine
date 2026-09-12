import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import run_funding_backtest, run_signal_backtest

IDX = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")


def test_signal_backtest_monotonic_up_no_trades():
    close = pd.Series(np.linspace(100, 200, 100), index=IDX)
    entries = pd.Series([True] + [False] * 99, index=IDX)
    exits = pd.Series([False] * 99 + [True], index=IDX)
    result = run_signal_backtest(close, entries, exits, fee=0.0)
    assert result.total_return == pytest.approx(1.0, rel=1e-3)  # 100 -> 200
    assert result.max_drawdown <= 0.0
    assert abs(result.max_drawdown) < 0.05
    assert result.num_trades == 1
    assert result.win_rate == 1.0
    assert len(result.equity_curve) == 100
    # metrics are finite, never NaN
    for v in (result.sharpe, result.sortino, result.calmar):
        assert np.isfinite(v)


def test_signal_backtest_losing_trade():
    close = pd.Series(np.linspace(200, 100, 100), index=IDX)
    entries = pd.Series([True] + [False] * 99, index=IDX)
    exits = pd.Series([False] * 99 + [True], index=IDX)
    result = run_signal_backtest(close, entries, exits, fee=0.0)
    assert result.total_return == pytest.approx(-0.5, rel=1e-2)
    assert result.win_rate == 0.0
    assert result.max_drawdown == pytest.approx(-0.5, abs=0.02)


def test_funding_backtest_hand_computed():
    # 10 hourly funding periods: 6 above threshold (0.001), 4 below
    rates = [0.001] * 6 + [0.00001] * 4
    funding = pd.Series(rates, index=IDX[:10])
    result = run_funding_backtest(funding, threshold=0.0005, fee=0.0)
    # positioned exactly for the 6 high bars: accrual = 6 * 0.001, zero fees
    assert result.total_return == pytest.approx(0.006, rel=1e-6)
    assert result.num_trades == 1
    assert result.win_rate == 1.0


def test_funding_backtest_fees_and_undefined_metrics():
    rates = [0.001] * 6 + [0.00001] * 4
    funding = pd.Series(rates, index=IDX[:10])
    result = run_funding_backtest(funding, threshold=0.0005, fee=0.001)
    # entry costs 2*fee, exit costs 2*fee -> total_return = 0.006 - 0.004
    assert result.total_return == pytest.approx(0.002, rel=1e-3)
    # flat funding series -> zero variance -> sharpe must be 0.0, not NaN
    flat = pd.Series([0.001] * 10, index=IDX[:10])
    flat_result = run_funding_backtest(flat, threshold=0.0005, fee=0.0)
    assert flat_result.sharpe == 0.0
    assert flat_result.num_trades == 1
