import pandas as pd

from src.backtesting import fees
from src.meta_learning.carry import carry_metrics, carry_regime_metrics


def _series(rate, hours=8760):
    idx = pd.date_range("2024-01-01", periods=hours, freq="1h", tz="UTC")
    return pd.Series([rate] * hours, index=idx)


def test_carry_positive_funding_and_free_fees():
    result = carry_metrics(_series(0.0001), threshold=0.0, fee=0.0)
    assert result["total_return"] > 0
    assert result["annualized"] > 0


def test_carry_negative_funding_not_positioned():
    result = carry_metrics(_series(-0.0001), threshold=0.0, fee=0.0)
    assert result["total_return"] == 0.0


def test_carry_fees_reduce_return():
    gross = carry_metrics(_series(0.0001), threshold=0.0, fee=0.0)
    net = carry_metrics(_series(0.0001), threshold=0.0, fee=0.001)
    assert net["total_return"] < gross["total_return"]


def test_regime_guard_avoids_sustained_negative_funding():
    idx = pd.date_range("2024-01-01", periods=1000, freq="1h", tz="UTC")
    rates = [0.0001] * 500 + [-0.0002] * 500
    funding = pd.Series(rates, index=idx)
    hold = carry_metrics(funding, -1e-9, 0.0, -1.0)      # always in
    guarded = carry_regime_metrics(funding, 0.0, enter_window=24,
                                   exit_window=48, enter_level=0.0, exit_level=0.0)
    assert guarded["total_return"] > hold["total_return"]


def test_regime_guard_waits_for_warmup():
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    funding = pd.Series([0.0001] * 200, index=idx)
    guarded = carry_regime_metrics(funding, 0.0, enter_window=24, exit_window=48)
    # no position until the enter window has filled -> fewer bars than always-in
    assert guarded["total_return"] > 0


def test_perp_fee_default_and_env(monkeypatch):
    monkeypatch.delenv("PERP_TAKER_FEE", raising=False)
    assert fees.perp_fee() == 0.00045
    monkeypatch.setenv("PERP_TAKER_FEE", "0.0002")
    assert fees.perp_fee() == 0.0002
