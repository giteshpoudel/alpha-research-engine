"""BTC leads, altcoins follow: buy the laggards after a BTC hourly shock.

Pre-registered in docs/prereg/2026-11-04-btc-lead-lag.md.

    PYTHONPATH=. python -m src.research.btc_lead_lag
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import get_clickhouse_client
from src.research import batch1 as b1
from src.research.breakout_exits import trades
from src.research.intraday_study import IS_START, OOS_END, OOS_START, STRESS_MULT, max_drawdown, sharpe, stress_flags, window

ALTS = ["ETH", "SOL", "XRP", "DOGE", "LINK", "ADA", "AVAX", "BNB", "LTC"]
COST = {"ETH": 5e-4, "SOL": 6e-4, "XRP": 5e-4, "DOGE": 7e-4, "LINK": 8e-4,
        "ADA": 8e-4, "AVAX": 9e-4, "BNB": 8e-4, "LTC": 8e-4}
GRID = [dict(z=z, H=H) for z in (2.0, 3.0) for H in (3, 12)]
SIGMA_WINDOW, LAG_RATIO = 720, 0.5
SEL, TEST = (IS_START, OOS_START), (OOS_START, OOS_END)
OUT = Path("data/research/btc_lead_lag.json")


def shocks(btc_close: pd.Series, z: float) -> pd.Series:
    """True at hourly close t when BTC's hour-t return > z x its prior 720-hour std."""
    r = btc_close.pct_change()
    sigma = r.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std().shift(1)
    return (r > z * sigma).fillna(False)


def weights(closes: pd.DataFrame, btc_close: pd.Series, z: float, H: int) -> tuple[pd.DataFrame, int]:
    """Weight per alt decided at each close: 1/9 for H hours after a shock in which it lagged."""
    shock = shocks(btc_close, z)
    rb = btc_close.pct_change()
    ra = closes.pct_change()
    lag = ra.lt(LAG_RATIO * rb, axis=0) & ra.notna()
    trigger = lag & shock.to_numpy()[:, None]
    w = np.zeros(closes.shape)
    left = np.zeros(closes.shape[1], dtype=int)
    trig = trigger.to_numpy()
    for t in range(len(closes)):
        left = np.where(trig[t], H, left)
        w[t] = np.where(left > 0, 1.0 / len(ALTS), 0.0)
        left = np.maximum(left - 1, 0)
    return pd.DataFrame(w, index=closes.index, columns=closes.columns), int(shock.sum())


def simulate(opens: pd.DataFrame, w: pd.DataFrame, stress: pd.DataFrame, mult: float = 1.0):
    """Batch-1 convention: weight decided at bar t's close is held over bar t+1."""
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    held = held.where(opens.notna() & opens.shift(-1).notna(), 0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = pd.DataFrame(np.where(stress.shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0),
                         index=opens.index, columns=opens.columns)
    side_cost = smult * pd.Series(COST)[opens.columns].to_numpy() * mult
    return (held * r - change * side_cost).sum(axis=1), held, r, side_cost


def evaluate(opens, closes, btc, stress, p) -> dict:
    w, n_shocks = weights(closes, btc, **p)
    net, held, r, sc = simulate(opens, w, stress)
    net2, *_ = simulate(opens, w, stress, mult=2.0)
    out = {"params": p}
    for label, (a, b) in {"selection": SEL, "test": TEST}.items():
        n, n2 = window(net, a, b), window(net2, a, b)
        h = held[(held.index >= a) & (held.index < b)]
        s = b1.stats(n, h, 8760)
        s["costs_x2_return"] = float((1 + n2).prod() - 1)
        s["shocks"] = int(shocks(btc, p["z"])[(btc.index >= a) & (btc.index < b)].sum())
        s.update(trades(held, r, sc, a, b))
        out[label] = s
    return out


def main() -> None:
    ch = get_clickhouse_client()
    btc_px = load_ohlcv(ch, "BTC", "1h", start=IS_START, end=OOS_END)
    px = {c: load_ohlcv(ch, c, "1h", start=IS_START, end=OOS_END) for c in ALTS}
    idx = btc_px.index
    opens = pd.concat({c: px[c]["open"] for c in ALTS}, axis=1).reindex(idx)
    closes = pd.concat({c: px[c]["close"] for c in ALTS}, axis=1).reindex(idx)
    stress = pd.concat({c: pd.Series(stress_flags(px[c], "1h"), index=px[c].index) for c in ALTS},
                       axis=1).reindex(idx).fillna(False).astype(bool)
    btc = btc_px["close"]

    results = [evaluate(opens, closes, btc, stress, p) for p in GRID]
    best = max(results, key=lambda x: x["selection"]["sharpe"])
    r = opens.shift(-1) / opens - 1
    bh_all = r.mean(axis=1, skipna=True).fillna(0.0)
    bh = window(bh_all, *TEST)
    bh_stats = {"return": float((1 + bh).prod() - 1), "max_dd": max_drawdown(bh), "sharpe": sharpe(bh, 8760)}
    if best["selection"]["return"] <= 0:
        tier = "retire (selection period not profitable; test not judged)"
    else:
        tier = b1.tier(best["test"], bh_stats["max_dd"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"grid": results, "chosen": best["params"], "buy_and_hold_test": bh_stats,
                               "tier": tier, "trials_added": 4, "trial_total": 406}, indent=2, default=str))
    for x in results:
        for label in ("selection", "test"):
            s = x[label]
            print(f"z={x['params']['z']} H={x['params']['H']:>2} {label:9s}: return {s['return']:+.1%} "
                  f"sharpe {s['sharpe']:+.2f} maxDD {s['max_dd']:.1%} costs x2 {s['costs_x2_return']:+.1%} | "
                  f"shocks {s['shocks']} trades {s['trades']} win {s['win_rate']:.0%} avg {s['avg_trade']:+.2%} "
                  f"underwater {s['underwater_days']:.0f}d")
    print(f"buy-and-hold test: {bh_stats['return']:+.1%} maxDD {bh_stats['max_dd']:.1%}")
    print(f"\nCHOSEN {best['params']}  TIER: {tier}")


if __name__ == "__main__":
    main()
