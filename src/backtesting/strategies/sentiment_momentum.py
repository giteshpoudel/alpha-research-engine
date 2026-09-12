"""Sentiment-adjusted momentum (spot, long-only): momentum gated by sentiment."""

from __future__ import annotations

import pandas as pd

SM_DEFAULTS = {"mom_window": 24, "mom_threshold": 0.0, "sent_threshold": 0.1}


def signals(close: pd.Series, sentiment: pd.Series, mom_window: int = 24,
            mom_threshold: float = 0.0, sent_threshold: float = 0.1) -> tuple[pd.Series, pd.Series]:
    """Long only when 24h momentum and sentiment both exceed their thresholds.

    Sentiment is forward-filled onto the price index (last known score, never
    future data).
    """
    mom = close.pct_change(mom_window)
    sent = sentiment.reindex(close.index).ffill()
    condition = (mom > mom_threshold) & (sent > sent_threshold)
    condition = condition.fillna(False).astype(bool)
    entries = condition & ~condition.shift(1, fill_value=False)
    exits = ~condition & condition.shift(1, fill_value=False)
    return entries, exits
