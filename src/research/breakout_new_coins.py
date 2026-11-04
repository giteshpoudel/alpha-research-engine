"""The frozen hourly breakout rule on coins it was never designed on.

Pre-registered in docs/prereg/2026-11-04-breakout-new-coins.md (commit 233c760).

    PYTHONPATH=. python -m src.research.breakout_new_coins
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import get_clickhouse_client
from src.research import batch1 as b1
from src.research.intraday_strategies import vol_breakout
from src.research.intraday_study import IS_START, OOS_END, OOS_START, STRESS_MULT, max_drawdown, sharpe, stress_flags, window

UTC = timezone.utc
COINS = ["ADA", "AVAX", "BNB", "LTC"]
PARAMS = dict(N=48, vratio=1.0, k=3.0, T=48)
COST = {"ADA": 8e-4, "AVAX": 9e-4, "BNB": 8e-4, "LTC": 8e-4}
VOL_TARGET, VOL_WINDOW = 0.80, 720
PRE = (IS_START, OOS_START)
OUT = Path("data/research/breakout_new_coins.json")


def weights(df: pd.DataFrame, n_coins: int) -> np.ndarray:
    """Vol-sized breakout weight decided at each hourly close (size fixed at entry)."""
    pos = vol_breakout(*[df[x].to_numpy() for x in ("open", "high", "low", "close", "volume")], **PARAMS)
    vol = (df["close"].pct_change().rolling(VOL_WINDOW).std() * math.sqrt(8760)).to_numpy()
    size = np.minimum(1.0, VOL_TARGET / np.where(vol > 0, vol, np.nan))
    w = np.zeros(len(df))
    for t in range(len(df)):
        if pos[t] == 1:
            w[t] = w[t - 1] if t and pos[t - 1] == 1 else (0.0 if np.isnan(size[t]) else size[t])
    return w / n_coins


def simulate(opens: pd.DataFrame, w: pd.DataFrame, stress: pd.DataFrame, mult: float = 1.0):
    """Same convention as batch 1: weight decided at bar t's close is held over bar t+1."""
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    held = held.where(opens.notna() & opens.shift(-1).notna(), 0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = np.where(stress.shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0)
    cost = change * pd.Series(COST)[opens.columns].to_numpy() * smult * mult
    per_coin = held * r - cost
    return per_coin, held


def summarize(per_coin: pd.DataFrame, per_coin2: pd.DataFrame, held: pd.DataFrame, start, end) -> dict:
    net, net2 = window(per_coin.sum(axis=1), start, end), window(per_coin2.sum(axis=1), start, end)
    h = held[(held.index >= start) & (held.index < end)]
    out = b1.stats(net, h, 8760)
    out["costs_x2_return"] = float((1 + net2).prod() - 1)
    out["per_coin"] = {}
    for c in per_coin.columns:
        s, s2 = window(per_coin[c], start, end), window(per_coin2[c], start, end)
        hc = h[c]
        entries = int(((hc > 0) & (hc.shift(1).fillna(0) == 0)).sum())
        # each coin's stand-alone return at full weight (x n coins) for a per-coin verdict
        k = len(per_coin.columns)
        out["per_coin"][c] = {"return": float((1 + k * s).prod() - 1), "costs_x2_return": float((1 + k * s2).prod() - 1),
                              "max_dd": max_drawdown(k * s), "entries": entries}
    return out


def main() -> None:
    ch = get_clickhouse_client()
    px = {c: load_ohlcv(ch, c, "1h", start=IS_START, end=OOS_END) for c in COINS}
    opens = pd.concat({c: px[c]["open"] for c in COINS}, axis=1).sort_index()
    stress = pd.concat({c: pd.Series(stress_flags(px[c], "1h"), index=px[c].index) for c in COINS},
                       axis=1).reindex(opens.index).fillna(False).astype(bool)
    W = pd.concat({c: pd.Series(weights(px[c], len(COINS)), index=px[c].index) for c in COINS},
                  axis=1).reindex(opens.index).fillna(0.0)
    pc, held = simulate(opens, W, stress)
    pc2, _ = simulate(opens, W, stress, mult=2.0)

    r = opens.shift(-1) / opens - 1
    bh_all = r.mean(axis=1, skipna=True).fillna(0.0)
    res = {}
    for label, (a, b) in {"oos_2025_26": (OOS_START, OOS_END), "pre_2022_24": PRE}.items():
        s = summarize(pc, pc2, held, a, b)
        bh = window(bh_all, a, b)
        s["buy_and_hold"] = {"return": float((1 + bh).prod() - 1), "max_dd": max_drawdown(bh), "sharpe": sharpe(bh, 8760)}
        res[label] = s
    o = res["oos_2025_26"]
    t = b1.tier(o, o["buy_and_hold"]["max_dd"])
    if t != "retire" and res["pre_2022_24"]["return"] <= 0:
        t = "watch"
    shadow = [c for c, v in o["per_coin"].items() if v["return"] > 0 and v["costs_x2_return"] > 0] if t != "retire" else []
    out = {"params": PARAMS, "costs": COST, "results": res, "tier": t, "shadow_list": shadow,
           "trials_added": 4, "trial_total": 399}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    for label, s in res.items():
        bh = s["buy_and_hold"]
        print(f"{label}: return {s['return']:+.1%} sharpe {s['sharpe']:+.2f} maxDD {s['max_dd']:.1%} "
              f"underwater {s['underwater_days']:.0f}d exposure {s['exposure']:.2f} costs x2 {s['costs_x2_return']:+.1%} "
              f"| buy-and-hold {bh['return']:+.1%} maxDD {bh['max_dd']:.1%}")
        for c, v in s["per_coin"].items():
            print(f"    {c}: {v['return']:+.1%} (costs x2 {v['costs_x2_return']:+.1%}) maxDD {v['max_dd']:.1%} entries {v['entries']}")
    print(f"\nTIER: {t.upper()}  shadow list: {shadow}")


if __name__ == "__main__":
    main()
