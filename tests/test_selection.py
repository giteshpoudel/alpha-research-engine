import numpy as np
import pandas as pd
import pytest

from src.meta_learning import selection


def test_psr_zero_sharpe_is_half():
    assert selection.probabilistic_sharpe_ratio(0.0, 1000) == pytest.approx(0.5)


def test_psr_increases_with_sharpe_and_sample():
    low = selection.probabilistic_sharpe_ratio(0.05, 500)
    high = selection.probabilistic_sharpe_ratio(0.10, 500)
    more = selection.probabilistic_sharpe_ratio(0.10, 5000)
    assert low < high < more


def test_deflated_equals_psr_with_one_trial():
    sr, n = 0.08, 2000
    assert selection.deflated_sharpe_ratio(sr, 1, n) == pytest.approx(
        selection.probabilistic_sharpe_ratio(sr, n), abs=1e-9)


def test_deflation_lowers_confidence_with_more_trials():
    sr, n = 0.08, 2000
    few = selection.deflated_sharpe_ratio(sr, 10, n, sr_std=0.05)
    many = selection.deflated_sharpe_ratio(sr, 1000, n, sr_std=0.05)
    assert many < few


def test_per_bar_sharpe_conversion():
    assert selection.per_bar_sharpe(8760 ** 0.5, 8760) == pytest.approx(1.0)


def test_per_bar_moments_on_normal_returns():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 1.0, 20000))
    sr, skew, kurt = selection.per_bar_moments(returns)
    assert abs(sr) < 0.05
    assert abs(skew) < 0.1
    assert abs(kurt - 3.0) < 0.2  # non-excess, normal == 3.0


def test_min_dsr_prob_gate(monkeypatch):
    monkeypatch.delenv("MIN_DSR_PROB", raising=False)
    assert selection.min_dsr_prob() == 0.0
    assert selection.passes_dsr(None) is True          # gate off by default
    monkeypatch.setenv("MIN_DSR_PROB", "0.95")
    assert selection.passes_dsr(0.97) is True
    assert selection.passes_dsr(0.90) is False
    assert selection.passes_dsr(None) is False


def test_expected_max_sharpe_is_positive_and_grows():
    assert selection.expected_max_sharpe(1, 0.05) == 0.0
    assert selection.expected_max_sharpe(100, 0.05) > 0
    assert selection.expected_max_sharpe(1000, 0.05) > selection.expected_max_sharpe(100, 0.05)
