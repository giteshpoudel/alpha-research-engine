import pandas as pd
import pytest

from src.backtesting.data import clean_ohlcv

IDX = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")


def _df(closes):
    c = pd.Series(closes, index=IDX, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c})


def test_glitch_filter_defaults_off():
    # Audit showed 5m "glitches" are real crash bars -> keep data as-is.
    df = _df([100, 100, 120, 100, 100])
    assert clean_ohlcv(df, "5m")["close"].iloc[2] == 120.0


def test_clean_ohlcv_neutralises_glitch_bar(monkeypatch):
    monkeypatch.setenv("GLITCH_FILTER", "1")
    df = _df([100, 100, 120, 100, 100])  # +20% then -17% bad print pair
    out = clean_ohlcv(df, "5m")          # 5m cap 5% -> both flattened
    assert (out["close"].pct_change().abs().dropna() <= 0.05 + 1e-9).all()


def test_clean_ohlcv_disabled(monkeypatch):
    monkeypatch.setenv("GLITCH_FILTER", "0")
    df = _df([100, 100, 120, 100, 100])
    assert clean_ohlcv(df, "5m")["close"].iloc[2] == 120.0


def test_clean_ohlcv_env_cap(monkeypatch):
    monkeypatch.setenv("GLITCH_FILTER", "1")
    monkeypatch.setenv("GLITCH_MAX_MOVE_5M", "0.5")
    df = _df([100, 100, 120, 100, 100])
    assert clean_ohlcv(df, "5m")["close"].iloc[2] == 120.0  # 20% < 50% cap
