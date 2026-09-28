"""Time-series momentum (spot, long-only): long while trailing return is positive."""

from __future__ import annotations

import pandas as pd

MOMENTUM_DEFAULTS = {"window": 24, "threshold": 0.0}


def signals(close: pd.Series, window: int = 24,
            threshold: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Enter when trailing ``window``-bar return crosses above ``threshold``; exit below."""
    momentum = close.pct_change(window)
    condition = momentum > threshold
    entries = condition & ~condition.shift(1, fill_value=False)
    exits = ~condition & condition.shift(1, fill_value=False)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
