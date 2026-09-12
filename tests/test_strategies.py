import numpy as np
import pandas as pd

from src.backtesting.strategies import mean_reversion, sentiment_momentum

IDX = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")


def _series(values):
    return pd.Series(values, index=IDX[: len(values)], dtype=float)


def test_mean_reversion_crossings():
    # 24 bars at 100 -> z=0; bar 24 crashes to 80 (z very negative); then reverts
    values = [100.0] * 24 + [80.0, 81.0, 82.0, 95.0] + [100.0] * 20
    close = _series(values)
    entries, exits = mean_reversion.signals(close, window=24, z_entry=-2.0, z_exit=0.0)
    assert entries.dtype == bool and exits.dtype == bool
    assert entries.iloc[24]  # first bar below z_entry
    assert entries.sum() == 1  # crossing-based: no repeated entries while depressed
    assert exits.sum() >= 1
    assert exits.idxmax() > entries.idxmax()  # exit comes after entry


def test_mean_reversion_no_signal_in_flat_market():
    close = _series([100.0] * 48)
    entries, exits = mean_reversion.signals(close)
    assert not entries.any()
    assert not exits.any()


def test_sentiment_momentum_gating():
    close = _series([100.0] * 24 + [101.0] * 24)  # +1% momentum after 24 bars
    bullish = _series([0.5] * 48)
    bearish = _series([-0.5] * 48)
    e_bull, _ = sentiment_momentum.signals(close, bullish)
    e_bear, _ = sentiment_momentum.signals(close, bearish)
    assert e_bull.sum() == 1  # momentum + bullish sentiment -> one entry
    assert e_bull.iloc[24]
    assert not e_bear.any()  # bearish sentiment blocks the entry


def test_sentiment_momentum_exit_on_sentiment_flip():
    close = _series([100.0] * 24 + [101.0] * 24)
    sent = _series([0.5] * 30 + [-0.5] * 18)  # flips bearish at bar 30
    entries, exits = sentiment_momentum.signals(close, sent)
    assert entries.sum() == 1
    assert exits.iloc[30]  # exit exactly when sentiment flips below threshold
