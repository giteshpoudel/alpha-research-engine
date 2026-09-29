"""RSI mean-reversion (spot, long-only): buy oversold, exit on recovery toward neutral."""

from __future__ import annotations

import pandas as pd

RSI_DEFAULTS = {"window": 14, "oversold": 30.0, "overbought": 55.0}


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(100.0)


def signals(close: pd.Series, window: int = 14, oversold: float = 30.0,
            overbought: float = 55.0) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, window)
    entries = (rsi <= oversold) & ~(rsi.shift(1) <= oversold)
    exits = (rsi >= overbought) & ~(rsi.shift(1) >= overbought)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
