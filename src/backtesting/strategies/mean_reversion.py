"""Mean reversion (spot, long-only): z-score crossing signals."""

from __future__ import annotations

import pandas as pd

MR_DEFAULTS = {"window": 24, "z_entry": -2.0, "z_exit": 0.0}


def signals(close: pd.Series, window: int = 24,
            z_entry: float = -2.0, z_exit: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Enter when z crosses below z_entry; exit when z crosses above z_exit."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = (close - ma) / sd
    entries = (z <= z_entry) & ~(z.shift(1) <= z_entry)
    exits = (z >= z_exit) & (z.shift(1) < z_exit)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
