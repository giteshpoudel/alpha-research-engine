"""EMA crossover trend (spot, long-only): long while fast EMA is above slow EMA."""

from __future__ import annotations

import pandas as pd

EMA_CROSS_DEFAULTS = {"fast": 12, "slow": 26}


def signals(close: pd.Series, fast: int = 12, slow: int = 26
            ) -> tuple[pd.Series, pd.Series]:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    condition = fast_ema > slow_ema
    entries = condition & ~condition.shift(1, fill_value=False)
    exits = ~condition & condition.shift(1, fill_value=False)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
