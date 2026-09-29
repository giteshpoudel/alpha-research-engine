"""Selection control: probabilistic / deflated Sharpe ratio.

Bailey & Lopez de Prado (2014). Guards against (a) selecting a strategy simply
because it was the best of many trials (multiplicity), and (b) judging non-normal
returns by a normal Sharpe. All Sharpe inputs here are *per-observation*; convert
an annualized Sharpe with :func:`per_bar_sharpe`.
"""

from __future__ import annotations

import math
import os
from statistics import NormalDist

_EULER = 0.5772156649015329


def min_dsr_prob() -> float:
    """Adoption floor for the Deflated Sharpe probability (0 = gate disabled)."""
    return float(os.environ.get("MIN_DSR_PROB", "0"))


def passes_dsr(dsr: float | None, threshold: float | None = None) -> bool:
    """True when the DSR gate is off or the candidate clears it."""
    if threshold is None:
        threshold = min_dsr_prob()
    if threshold <= 0:
        return True
    return dsr is not None and dsr >= threshold


def per_bar_sharpe(annualized_sharpe: float, periods_per_year: float) -> float:
    return annualized_sharpe / math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float = 0.0,
                               kurt: float = 3.0) -> float:
    """P(true SR > 0) given the observed per-bar Sharpe and return moments."""
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    z = sr * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return NormalDist().cdf(z)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected best per-bar Sharpe under the null when testing ``n_trials``."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    a = NormalDist().inv_cdf(1.0 - 1.0 / n_trials)
    b = NormalDist().inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER) * a + _EULER * b)


def deflated_sharpe_ratio(sr: float, n_trials: int, n_obs: int, skew: float = 0.0,
                          kurt: float = 3.0, sr_std: float | None = None) -> float:
    """P(true SR > E[max SR | n_trials]). ``sr``/``sr_std`` are per-bar."""
    if sr_std is None or sr_std <= 0:
        sr_std = max(abs(sr), 1e-9)
    sr0 = expected_max_sharpe(n_trials, sr_std)
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return NormalDist().cdf(z)
