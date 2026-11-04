import numpy as np
import pandas as pd

from src.research import btc_lead_lag as ll


def _setup(n=800, shock_at=(750,), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rb = rng.normal(0, 0.002, n)
    for t in shock_at:
        rb[t] = 0.03                        # ~15 sigma
    btc = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    ra = pd.DataFrame(rng.normal(0, 0.002, (n, len(ll.ALTS))), index=idx, columns=ll.ALTS)
    for t in shock_at:
        ra.iloc[t] = 0.0                    # every alt lags...
        ra.iloc[t, 0] = 0.03                # ...except ETH, which moved with BTC
    closes = 10 * (1 + ra).cumprod()
    return btc, closes


def test_shock_detected_only_after_warmup():
    btc, _ = _setup(shock_at=(100, 750))
    s = ll.shocks(btc, 3.0)
    assert not s.iloc[100] and s.iloc[750]


def test_laggards_bought_for_H_hours_and_movers_skipped():
    btc, closes = _setup()
    w, n = ll.weights(closes, btc, z=3.0, H=3)
    assert n >= 1
    sol = w["SOL"].to_numpy()
    assert list(sol[750:754]) == [1 / 9] * 3 + [0.0]
    assert w["ETH"].iloc[750:754].sum() == 0


def test_new_shock_restarts_timer():
    btc, closes = _setup(shock_at=(750, 752))
    w, _ = ll.weights(closes, btc, z=3.0, H=3)
    assert (w["SOL"].iloc[750:755] > 0).all() and w["SOL"].iloc[755] == 0


def test_missing_coin_never_bought():
    btc, closes = _setup()
    closes["XRP"] = np.nan
    w, _ = ll.weights(closes, btc, z=3.0, H=3)
    assert w["XRP"].sum() == 0
