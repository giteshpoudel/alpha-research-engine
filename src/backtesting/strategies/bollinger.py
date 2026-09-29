"""Bollinger-band reversion (spot, long-only): buy below the lower band, exit at the mean."""

from __future__ import annotations

import pandas as pd

BOLLINGER_DEFAULTS = {"window": 20, "k": 2.0}


def signals(close: pd.Series, window: int = 20, k: float = 2.0
            ) -> tuple[pd.Series, pd.Series]:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    lower = ma - k * sd
    entries = (close < lower) & ~(close.shift(1) < lower.shift(1))
    exits = (close > ma) & ~(close.shift(1) > ma.shift(1))
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
