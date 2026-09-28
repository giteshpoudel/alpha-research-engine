"""Donchian breakout (spot, long-only): buy new highs, sell new lows.

Uses shifted rolling extremes so the breakout level at bar ``t`` is computed
only from bars before ``t`` (no lookahead).
"""

from __future__ import annotations

import pandas as pd

BREAKOUT_DEFAULTS = {"entry_window": 24, "exit_window": 12}


def signals(close: pd.Series, entry_window: int = 24,
            exit_window: int = 12) -> tuple[pd.Series, pd.Series]:
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(exit_window).min().shift(1)
    entries = (close > upper) & ~(close.shift(1) > upper.shift(1))
    exits = (close < lower) & ~(close.shift(1) < lower.shift(1))
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
