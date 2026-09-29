import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (
    run_funding_backtest,
    run_maker_backtest,
    run_signal_backtest,
)

IDX = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
MIDX = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")


def _ohlc(close):
    s = pd.Series(close, index=MIDX, dtype=float)
    return pd.DataFrame({"high": s + 1.0, "low": s - 1.0, "close": s})


def test_maker_buy_fills_on_dip_and_sells_into_strength():
    df = _ohlc([100, 100, 110, 110, 110, 110])
    entries = pd.Series([True, False, False, False, False, False], index=MIDX)
    exits = pd.Series([False, False, True, False, False, False], index=MIDX)
    result = run_maker_backtest(df, entries, exits, fee=0.0, limit_bps=5.0, max_wait=1)
    # buy limit 100*(1-5bps)=99.95 fills on bar1 (low 99 <= 99.95);
    # sell limit 110*(1+5bps)=110.055 fills on bar3 (high 111 >= 110.055)
    assert result.num_trades == 1
    assert result.total_return == pytest.approx(110.055 / 99.95 - 1.0, rel=1e-9)
    assert result.equity_curve.iloc[0] == 1.0


def test_maker_entry_not_filled_when_price_never_dips():
    df = _ohlc([100, 101, 102, 103, 104, 105])  # lows stay above the 99.95 bid
    entries = pd.Series([True, False, False, False, False, False], index=MIDX)
    exits = pd.Series([False, False, True, False, False, False], index=MIDX)
    result = run_maker_backtest(df, entries, exits, fee=0.0)
    assert result.num_trades == 0
    assert result.total_return == 0.0


def test_maker_exit_falls_back_to_taker_after_max_wait():
    df = pd.DataFrame({
        "high": pd.Series([100.04] * 6, index=MIDX),
        "low": pd.Series([99.9] * 6, index=MIDX),
        "close": pd.Series([100.0] * 6, index=MIDX),
    })
    entries = pd.Series([True, False, False, False, False, False], index=MIDX)
    exits = pd.Series([False, False, True, False, False, False], index=MIDX)
    result = run_maker_backtest(df, entries, exits, fee=0.0, limit_bps=5.0,
                                max_wait=1, taker_fee=0.001)
    # entry at 99.95 fills; sell limit 100.05 never touched (high 100.04) ->
    # fallback taker sell at close 100 with 0.1% fee
    assert result.num_trades == 1
    assert result.total_return == pytest.approx((100.0 * 0.999) / 99.95 - 1.0, rel=1e-9)


def test_maker_applies_maker_fee_on_both_sides():
    df = _ohlc([100, 100, 110, 110, 110, 110])
    entries = pd.Series([True, False, False, False, False, False], index=MIDX)
    exits = pd.Series([False, False, True, False, False, False], index=MIDX)
    result = run_maker_backtest(df, entries, exits, fee=0.001, limit_bps=5.0, max_wait=1)
    expected = (110.055 * (1 - 0.001)) / (99.95 * (1 + 0.001)) - 1.0
    assert result.total_return == pytest.approx(expected, rel=1e-9)


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


def test_funding_flat_series_sharpe_zero_at_any_rate():
    # flat funding at 0.003/bar: eps guard must catch near-zero float std
    flat = pd.Series([0.003] * 10, index=IDX[:10])
    result = run_funding_backtest(flat, threshold=0.0005, fee=0.0)
    assert result.sharpe == 0.0


def test_funding_total_return_matches_equity_curve():
    rates = [0.001] * 6 + [0.00001] * 4
    funding = pd.Series(rates, index=IDX[:10])
    result = run_funding_backtest(funding, threshold=0.0005, fee=0.001)
    assert result.equity_curve.iloc[0] == 1.0
    assert result.total_return == pytest.approx(
        result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1,
        rel=1e-9,
    )
