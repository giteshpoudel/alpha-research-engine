"""Pre-registered Phase 2 intraday strategies (docs/prereg/2026-11-02-phase2-intraday.md).

Each function takes OHLCV numpy arrays and returns a long/flat position array
``pos`` where ``pos[t]`` is the position decided at the close of bar t, using
only data up to and including bar t. The harness applies the one-bar lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def realized_vol(close: np.ndarray, n: int) -> np.ndarray:
    lr = np.diff(np.log(close), prepend=np.nan)
    return pd.Series(lr).rolling(n, min_periods=n).std().to_numpy()


def vol_breakout(o, h, l, c, v, N: int, vratio: float, k: float, T: int) -> np.ndarray:
    """Family A: enter on an N-bar high when short-term vol has expanded vs long-term;
    exit on an ATR trailing stop from the highest close since entry, or after T bars."""
    n = len(c)
    prior_high = pd.Series(h).shift(1).rolling(N, min_periods=N).max().to_numpy()
    short = max(2, N // 4)
    ratio = realized_vol(c, short) / realized_vol(c, 4 * N)
    a = atr(h, l, c, N)
    pos = np.zeros(n)
    inpos, peak, held = False, 0.0, 0
    for t in range(n):
        if inpos:
            held += 1
            peak = max(peak, c[t])
            if c[t] < peak - k * a[t] or held >= T:
                inpos = False
        elif (not np.isnan(prior_high[t]) and not np.isnan(ratio[t]) and not np.isnan(a[t])
              and c[t] > prior_high[t] and ratio[t] > vratio):
            inpos, peak, held = True, c[t], 0
        pos[t] = 1.0 if inpos else 0.0
    return pos


def overshoot_fade(o, h, l, c, v, L: int, m: float, z: float, H: int,
                   atr_n: int = 48, stop_atr: float = 3.0) -> np.ndarray:
    """Family B: buy a sharp ATR-normalized drop on a volume spike; exit when half the
    drop is recovered, on a 3-ATR stop below entry, or after H bars."""
    n = len(c)
    a = atr(h, l, c, atr_n)
    vs = pd.Series(v)
    vz = ((vs - vs.rolling(atr_n, min_periods=atr_n).mean())
          / vs.rolling(atr_n, min_periods=atr_n).std()).to_numpy()
    pos = np.zeros(n)
    inpos, entry, target, stop, held = False, 0.0, 0.0, 0.0, 0
    for t in range(n):
        if inpos:
            held += 1
            if c[t] >= target or c[t] <= stop or held >= H:
                inpos = False
        elif t >= L and not np.isnan(a[t]) and a[t] > 0 and not np.isnan(vz[t]):
            drop = c[t] - c[t - L]
            if drop / a[t] < -m and vz[t] > z:
                inpos, entry, held = True, c[t], 0
                target = entry - 0.5 * drop  # drop is negative: target above entry
                stop = entry - stop_atr * a[t]
        pos[t] = 1.0 if inpos else 0.0
    return pos


GRIDS = {
    "vol_breakout": [dict(N=N, vratio=vr, k=k, T=T)
                     for N in (24, 48, 96) for vr in (1.0, 1.3) for k in (2.0, 3.0)
                     for T in (N, 2 * N)],
    "overshoot_fade": [dict(L=L, m=m, z=z, H=H)
                       for L in (4, 12) for m in (3.0, 5.0) for z in (1.0, 2.0)
                       for H in (12, 48)],
}
FUNCS = {"vol_breakout": vol_breakout, "overshoot_fade": overshoot_fade}
