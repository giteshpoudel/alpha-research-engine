"""Per-interval fee model.

Intraday strategies trade far more often, so a flat 0.1%/side fee makes them
unviable by construction. Fees are per-interval and env-overridable
(``FEE_5M``, ``FEE_1H``, ``FEE_1D``, or ``DEFAULT_FEE``); the 5m default is a
maker-like 2 bps.
"""

from __future__ import annotations

import os

from src.backtesting.strategies.registry import interval_of

_DEFAULTS = {"5m": 0.0002, "1h": 0.001, "1d": 0.001}


def fee_for_interval(interval: str) -> float:
    env = os.environ.get(f"FEE_{interval.upper()}")
    if env is not None:
        return float(env)
    if interval in _DEFAULTS:
        return _DEFAULTS[interval]
    return float(os.environ.get("DEFAULT_FEE", "0.001"))


def fee_for_strategy(strategy: str) -> float:
    return fee_for_interval(interval_of(strategy))
