"""MACD signal-line crossover (spot, long-only)."""

from __future__ import annotations

import pandas as pd

MACD_DEFAULTS = {"fast": 12, "slow": 26, "signal": 9}


def signals(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
            ) -> tuple[pd.Series, pd.Series]:
    macd = (close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean())
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    condition = macd > signal_line
    entries = condition & ~condition.shift(1, fill_value=False)
    exits = ~condition & condition.shift(1, fill_value=False)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
