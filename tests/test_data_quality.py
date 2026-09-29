import numpy as np
import pandas as pd

from src.backtesting.data import clean_ohlcv

IDX = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")


def _df(closes):
    c = pd.Series(closes, index=IDX, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c})


def test_clean_ohlcv_neutralises_glitch_bar():
    df = _df([100, 100, 120, 100, 100])  # bar 2 is a +20% bad print
    out = clean_ohlcv(df, "5m")          # 5m cap is 5%
    assert out["close"].iloc[2] == 100.0
    assert (out["close"].pct_change().abs().dropna() <= 0.05 + 1e-9).all()


def test_clean_ohlcv_leaves_normal_data_untouched():
    df = _df([100, 100.2, 100.1, 100.3, 100.2])
    out = clean_ohlcv(df, "5m")
    assert out["close"].equals(df["close"])


def test_clean_ohlcv_disabled(monkeypatch):
    monkeypatch.setenv("GLITCH_FILTER", "0")
    df = _df([100, 100, 120, 100, 100])
    assert clean_ohlcv(df, "5m")["close"].iloc[2] == 120.0


def test_clean_ohlcv_env_cap(monkeypatch):
    monkeypatch.setenv("GLITCH_MAX_MOVE_5M", "0.5")
    df = _df([100, 100, 120, 100, 100])
    assert clean_ohlcv(df, "5m")["close"].iloc[2] == 120.0  # 20% < 50% cap
