"""Strategy batch 1. Implements docs/prereg/2026-11-03-batch1.md.

    PYTHONPATH=. python -m src.research.batch1

Output: data/research/batch1.json and a printed summary. OOS runs once.

Implementation choices fixed before the run:
- Weights are a fraction of one budget. A weight decided on bar t's close is held from
  bar t+1's open (same model as intraday_study.run), cost per side on |weight change|.
- trend: vol-scaled weights are re-set every day (standard volatility targeting).
- breakout_vol: the weight is fixed at entry (vol measured on the entry bar) and held
  until exit, as live trading would place one order per entry.
- Daily-bar stress flags come from the 1h flag of the hour before each daily open.
- Funding: Hyperliquid hourly rate, 72-hour mean × 8760 = annualized; daily value at day
  D uses funding up to D 00:00 (one day before the decision on D's close). No data →
  no filter.
- Parameters are chosen jointly for the whole portfolio (one setting per strategy).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_funding, load_ohlcv
from src.ingestion.schemas import get_clickhouse_client
from src.research.intraday_strategies import vol_breakout
from src.research.intraday_study import (
    FOLD_TESTS, IS_END, IS_START, OOS_END, OOS_START, UTC, max_drawdown, sharpe,
    stress_flags, window,
)

COINS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
COST = {"BTC": 5e-4, "ETH": 5e-4, "SOL": 6e-4, "XRP": 5e-4, "DOGE": 7e-4, "LINK": 8e-4}
STRESS_MULT = 3.0
BREAKOUT_PARAMS = {  # frozen selections: Phase 2 (b6ac068) and alt-coin study (51f703e)
    "BTC": dict(N=96, vratio=1.0, k=3.0, T=192), "ETH": dict(N=48, vratio=1.3, k=3.0, T=96),
    "SOL": dict(N=24, vratio=1.3, k=3.0, T=48), "XRP": dict(N=24, vratio=1.0, k=3.0, T=24),
    "DOGE": dict(N=48, vratio=1.0, k=2.0, T=48), "LINK": dict(N=48, vratio=1.0, k=3.0, T=48),
}
GRIDS = {
    "trend": [dict(L=L, target=t) for L in (20, 50, 100) for t in (0.4, 0.6, None)],
    "rotation": [dict(L=L, k=k) for L in (7, 14, 28, 56) for k in (2, 3)],
    "breakout_vol": [dict(target=t) for t in (0.4, 0.6, 0.8)],
    "trend_funding": [dict(F=f) for f in (0.2, 0.4, 0.6)],
}
BPY = {"1d": 365, "1h": 8760}
EVENTS = {"2025-10-10 crash": "2025-10-10", "Binance hack": "2026-10-29"}
Y2024 = (datetime(2024, 1, 1, tzinfo=UTC), IS_END)
OUT = Path("data/research/batch1.json")


# --- data ---------------------------------------------------------------------------

def panel(frames: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return pd.concat({c: f[col] for c, f in frames.items()}, axis=1).sort_index()


def load(ch) -> dict:
    h = {c: load_ohlcv(ch, c, "1h", start=IS_START, end=OOS_END) for c in COINS}
    d = {c: load_ohlcv(ch, c, "1d", start=IS_START, end=OOS_END) for c in COINS}
    stress_h = pd.concat({c: pd.Series(stress_flags(h[c], "1h"), index=h[c].index)
                          for c in COINS}, axis=1).sort_index()
    daily_idx = panel(d, "open").index
    stress_d = stress_h.shift(1).reindex(daily_idx, method="ffill").fillna(False).astype(bool)
    funding = {}
    for c in COINS:
        f = load_funding(ch, c, start=IS_START, end=OOS_END)
        funding[c] = (f.rolling(72, min_periods=48).mean() * 8760)
    fund_d = pd.concat(funding, axis=1).sort_index()
    fund_d = fund_d.reindex(daily_idx, method="ffill")
    return {"h": h, "d": d, "open_h": panel(h, "open"), "open_d": panel(d, "open"),
            "close_d": panel(d, "close"), "stress_h": stress_h.reindex(panel(h, "open").index)
            .fillna(False).astype(bool), "stress_d": stress_d, "fund_d": fund_d}


# --- simulator ------------------------------------------------------------------------

def simulate(opens: pd.DataFrame, w: pd.DataFrame, stress: pd.DataFrame,
             mult: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """Weight decided on bar t's close is held during bar t+1 (open t+1 -> open t+2)."""
    w = w.reindex(columns=opens.columns).fillna(0.0)
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    held = held.where(opens.notna() & opens.shift(-1).notna(), 0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = np.where(stress.reindex(opens.index).shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0)
    cost = change * pd.Series(COST)[opens.columns].to_numpy() * smult * mult
    per_coin = held * r - cost
    return per_coin.sum(axis=1), held


# --- strategies -----------------------------------------------------------------------

def w_trend(data: dict, L: int, target: float | None) -> pd.DataFrame:
    close = data["close_d"]
    long = (close / close.shift(L) - 1) > 0
    if target is None:
        size = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    else:
        vol = close.pct_change().rolling(30).std() * math.sqrt(365)
        size = (target / vol).clip(upper=1.0).fillna(0.0)
    return (long * size / len(COINS)).fillna(0.0)


def w_rotation(data: dict, L: int, k: int) -> pd.DataFrame:
    close = data["close_d"]
    mom = close / close.shift(L) - 1
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in close.index[close.index.dayofweek == 6]:   # Sunday close -> Monday open
        row = mom.loc[t].dropna()
        pick = row[row > 0].sort_values(ascending=False).index[:k]
        w.loc[t] = 0.0
        w.loc[t, list(pick)] = 1.0 / k
    return w.ffill().fillna(0.0)


def w_breakout_vol(data: dict, target: float, cache: dict) -> pd.DataFrame:
    opens = data["open_h"]
    out = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    for c in COINS:
        df = data["h"][c]
        if c not in cache:
            cache[c] = vol_breakout(*[df[x].to_numpy() for x in ("open", "high", "low", "close", "volume")],
                                    **BREAKOUT_PARAMS[c])
        pos = cache[c]
        vol = (df["close"].pct_change().rolling(720).std() * math.sqrt(8760)).to_numpy()
        size = np.minimum(1.0, target / np.where(vol > 0, vol, np.nan))
        w = np.zeros(len(df))
        for t in range(len(df)):
            if pos[t] == 1:
                w[t] = w[t - 1] if t and pos[t - 1] == 1 else (0.0 if np.isnan(size[t]) else size[t])
        out.loc[df.index, c] = w / len(COINS)
    return out


def w_trend_funding(data: dict, F: float, base: dict) -> pd.DataFrame:
    w = w_trend(data, **base)
    blocked = (data["fund_d"].reindex(w.index) > F).fillna(False)
    return w.where(~blocked, 0.0)


# --- evaluation -------------------------------------------------------------------------

def underwater_days(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    peak_t, worst, peak = ret.index[0], 0.0, -np.inf
    for t, v in eq.items():
        if v >= peak:
            peak, peak_t = v, t
        else:
            worst = max(worst, (t - peak_t).total_seconds() / 86400)
    return worst


def stats(net: pd.Series, held: pd.DataFrame, bpy: int) -> dict:
    years = len(net) / bpy
    turnover = float((held - held.shift(1).fillna(0)).abs().sum().sum() / years) if years else 0.0
    return {"return": float((1 + net).prod() - 1), "sharpe": sharpe(net, bpy),
            "max_dd": max_drawdown(net), "underwater_days": underwater_days(net),
            "exposure": float(held.sum(axis=1).mean()), "turnover_per_year": turnover}


def walk_forward(weights_for, opens, stress, grid, bpy) -> dict:
    nets = [simulate(opens, weights_for(p), stress)[0] for p in grid]
    parts, chosen = [], []
    for test_start in FOLD_TESTS:
        train_start = test_start - pd.DateOffset(months=12)
        best = max(range(len(grid)), key=lambda i: sharpe(window(nets[i], train_start, test_start), bpy))
        chosen.append(grid[best])
        parts.append(window(nets[best], test_start, test_start + pd.DateOffset(months=3)))
    stitched = pd.concat(parts)
    folds = [float((1 + window(stitched, t, t + pd.DateOffset(months=3))).prod() - 1) for t in FOLD_TESTS]
    final = max(range(len(grid)), key=lambda i: sharpe(window(nets[i], *Y2024), bpy))
    return {"return": float((1 + stitched).prod() - 1), "sharpe": sharpe(stitched, bpy),
            "max_dd": max_drawdown(stitched), "folds_positive": sum(f > 0 for f in folds),
            "folds": folds, "chosen": chosen, "final": grid[final],
            "grid_2024_sharpe": [sharpe(window(n, *Y2024), bpy) for n in nets]}


def oos_eval(weights, opens, stress, bpy, bh) -> dict:
    w_oos = weights
    net, held = simulate(opens, w_oos, stress)
    net2, _ = simulate(opens, w_oos, stress, mult=2.0)
    net, net2 = window(net, OOS_START, OOS_END), window(net2, OOS_START, OOS_END)
    held = held[(held.index >= OOS_START) & (held.index < OOS_END)]
    out = stats(net, held, bpy)
    out["costs_x2_return"] = float((1 + net2).prod() - 1)
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    contrib = (held * r)[(r.index >= OOS_START) & (r.index < OOS_END)].sum()
    out["per_coin_contribution"] = {c: float(v) for c, v in contrib.items()}
    out["bh_exposure_matched"] = float((1 + bh[bpy] * out["exposure"]).prod() - 1)
    out["events"] = {}
    for name, day in EVENTS.items():
        d = pd.Timestamp(day, tz=UTC)
        w = lambda s: window(s, d - pd.Timedelta(days=1), d + pd.Timedelta(days=3))
        out["events"][name] = {"strategy": float((1 + w(net)).prod() - 1),
                               "buy_and_hold": float((1 + w(bh[bpy])).prod() - 1)}
    out["_net"] = net
    return out


def tier(o: dict, bh_dd: float, cap_watch: bool = False) -> str:
    if o["return"] <= 0 or o["sharpe"] < 0.5 or o["costs_x2_return"] <= 0:
        return "retire"
    candidate = (o["max_dd"] > -0.15 and abs(o["max_dd"]) <= 0.5 * abs(bh_dd)
                 and o["underwater_days"] <= 182)
    return "candidate" if candidate and not cap_watch else "watch"


def main() -> None:
    ch = get_clickhouse_client()
    data = load(ch)
    od, oh, sd, sh = data["open_d"], data["open_h"], data["stress_d"], data["stress_h"]
    bh = {}
    for bpy, opens in ((365, od), (8760, oh)):
        r = (opens.shift(-1) / opens - 1)
        bh[bpy] = window(r.mean(axis=1, skipna=True).fillna(0.0), OOS_START, OOS_END)
    bh_stats = {"return": float((1 + bh[365]).prod() - 1), "sharpe": sharpe(bh[365], 365),
                "max_dd": max_drawdown(bh[365])}

    bcache: dict = {}
    specs = {
        "trend": (lambda p: w_trend(data, **p), od, sd, 365),
        "rotation": (lambda p: w_rotation(data, **p), od, sd, 365),
        "breakout_vol": (lambda p: w_breakout_vol(data, p["target"], bcache), oh, sh, 8760),
    }
    results = {}
    for name, (fn, opens, stress, bpy) in specs.items():
        wf = walk_forward(fn, opens, stress, GRIDS[name], bpy)
        o = oos_eval(fn(wf["final"]), opens, stress, bpy, bh)
        results[name] = {"in_sample": wf, "oos": o,
                         "tier": tier(o, bh_stats["max_dd"], cap_watch=name == "breakout_vol")}

    base = results["trend"]["in_sample"]["final"]
    grid = GRIDS["trend_funding"]
    nets = [simulate(od, w_trend_funding(data, p["F"], base), sd)[0] for p in grid]
    s24 = [sharpe(window(n, *Y2024), 365) for n in nets]
    final = grid[int(np.argmax(s24))]
    o = oos_eval(w_trend_funding(data, final["F"], base), od, sd, 365, bh)
    results["trend_funding"] = {"in_sample": {"final": final, "base": base, "grid_2024_sharpe": s24,
                                              "trend_2024_sharpe": sharpe(window(
                                                  simulate(od, w_trend(data, **base), sd)[0], *Y2024), 365)},
                                "oos": o, "tier": tier(o, bh_stats["max_dd"])}

    nets_oos = {k: v["oos"].pop("_net") for k, v in results.items()}
    corr = pd.concat({k: (v.resample("1D").apply(lambda x: (1 + x).prod() - 1) if k == "breakout_vol" else v)
                      for k, v in nets_oos.items()}, axis=1).corr().round(2)
    out = {"coins": COINS, "costs": COST, "buy_and_hold": bh_stats, "results": results,
           "oos_daily_correlation": corr.to_dict(), "trials_added": 23, "trial_total": 337}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"buy-and-hold (6 coins, OOS): return {bh_stats['return']:+.1%} sharpe {bh_stats['sharpe']:+.2f} "
          f"maxDD {bh_stats['max_dd']:.1%}")
    for name, r in results.items():
        i, o = r["in_sample"], r["oos"]
        if "folds_positive" in i:
            print(f"\n{name}: IS return {i['return']:+.1%} sharpe {i['sharpe']:+.2f} maxDD {i['max_dd']:.1%} "
                  f"folds+ {i['folds_positive']}/8 | final {i['final']}")
        else:
            print(f"\n{name}: final {i['final']} on trend {i['base']} | 2024 sharpe {max(i['grid_2024_sharpe']):+.2f} "
                  f"vs trend alone {i['trend_2024_sharpe']:+.2f}")
        print(f"  OOS return {o['return']:+.1%} sharpe {o['sharpe']:+.2f} maxDD {o['max_dd']:.1%} "
              f"underwater {o['underwater_days']:.0f}d costs x2 {o['costs_x2_return']:+.1%} "
              f"exposure {o['exposure']:.0%} turnover {o['turnover_per_year']:.1f}/yr "
              f"| exposure-matched B&H {o['bh_exposure_matched']:+.1%}")
        print("  per coin:", {c: f"{v:+.1%}" for c, v in o["per_coin_contribution"].items()})
        print("  events:", {k: f"{v['strategy']:+.1%} vs {v['buy_and_hold']:+.1%}" for k, v in o["events"].items()})
        print(f"  TIER: {r['tier'].upper()}")
    print("\nOOS daily correlation:\n", corr)


if __name__ == "__main__":
    main()
