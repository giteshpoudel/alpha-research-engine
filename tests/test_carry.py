import numpy as np
import pandas as pd

from src.backtesting import fees
from src.meta_learning.carry import carry_metrics


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


def test_perp_fee_default_and_env(monkeypatch):
    monkeypatch.delenv("PERP_TAKER_FEE", raising=False)
    assert fees.perp_fee() == 0.00045
    monkeypatch.setenv("PERP_TAKER_FEE", "0.0002")
    assert fees.perp_fee() == 0.0002
