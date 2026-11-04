"""Strategy batch 2 (machine learning). Implements docs/prereg/2026-11-03-batch2-ml.md.

    PYTHONPATH=. python -m src.research.batch2

Output: data/research/batch2.json and a printed summary. OOS runs once.

Implementation choices fixed before the run:
- Quarterly retraining from 2023Q1 to 2026Q4 on samples whose label time is at or before
  the quarter start. Predictions for every quarter are made in the same pass; only the
  2023-2024 quarters are used to pick hyperparameters.
- Breakout trade label: open of the bar after entry to open of the bar after exit, with
  entry and exit costs (stress multiplier at the fill bars). Label time = exit fill time.
- Hourly funding feature: 72h mean of Hyperliquid hourly funding × 8760, as of the bar open.
- A feature with no values at all in a quarter's training set (funding before 2023-05) is
  left out of that quarter's model; infinite values (division by zero) count as missing.
  (Fixed after a crash on the first run, before any result was produced.)
- Feature importances: permutation importance (ROC AUC, 5 repeats) of the model trained
  for 2024Q4, scored on 2024Q4 samples.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

from src.backtesting.data import load_funding
from src.ingestion.schemas import get_clickhouse_client
from src.research import batch1 as b
from src.research.intraday_study import IS_END, IS_START, OOS_END, OOS_START, UTC, sharpe, window

QUARTERS = [pd.Timestamp(f"{y}-{m:02d}-01", tz=UTC) for y in (2023, 2024, 2025, 2026) for m in (1, 4, 7, 10)]
IS_TEST = (pd.Timestamp("2023-01-01", tz=UTC), IS_END)
GRIDS = {
    "meta_breakout": [dict(theta=t, depth=d) for t in (0.40, 0.50, 0.60) for d in (2, 3)],
    "gbm_daily": [dict(theta=t, depth=d) for t in (0.50, 0.55, 0.60) for d in (2, 3)],
}
UNFILTERED = {"sharpe": 0.90, "max_dd": -0.201}   # batch 1 vol-sized breakout, OOS
OUT = Path("data/research/batch2.json")


def model(depth: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=depth, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def walk_predict(X: pd.DataFrame, y: pd.Series, t_sample: pd.Series, t_label: pd.Series,
                 depth: int) -> tuple[pd.Series, dict]:
    """Quarterly purged walk-forward: probability for every sample in 2023Q1..2026Q4."""
    p = pd.Series(np.nan, index=X.index)
    models = {}
    for q in QUARTERS:
        train = t_label <= q
        test = (t_sample >= q) & (t_sample < q + pd.DateOffset(months=3))
        if not test.any() or train.sum() < 50 or y[train].nunique() < 2:
            continue
        cols = [c for c in X.columns if X.loc[train, c].notna().any()]   # e.g. funding before 2023-05
        m = model(depth).fit(X.loc[train, cols], y[train])
        p[test] = m.predict_proba(X.loc[test, cols])[:, 1]
        m.cols_ = cols
        models[q] = m
    return p, models


def scores(p: pd.Series, y: pd.Series, mask: pd.Series) -> dict:
    ok = mask & p.notna()
    if ok.sum() < 10 or y[ok].nunique() < 2:
        return {"auc": None, "accuracy": None, "n": int(ok.sum())}
    return {"auc": float(roc_auc_score(y[ok], p[ok])),
            "accuracy": float(accuracy_score(y[ok], p[ok] >= 0.5)), "n": int(ok.sum()),
            "base_rate": float(y[ok].mean())}


def importances(models: dict, X, y, t_sample) -> dict:
    q = pd.Timestamp("2024-10-01", tz=UTC)
    if q not in models:
        return {}
    test = (t_sample >= q) & (t_sample < IS_END)
    if y[test].nunique() < 2:
        return {}
    cols = models[q].cols_
    r = permutation_importance(models[q], X.loc[test, cols], y[test], scoring="roc_auc", n_repeats=5, random_state=0)
    return dict(sorted(((c, round(float(v), 4)) for c, v in zip(cols, r.importances_mean)),
                       key=lambda kv: -kv[1]))


# --- strategy 5: meta-labeling the breakout ------------------------------------------------

def hourly_funding(ch, idx_by_coin: dict) -> dict:
    out = {}
    for c, idx in idx_by_coin.items():
        f = load_funding(ch, c, start=IS_START, end=OOS_END)
        out[c] = (f.rolling(72, min_periods=48).mean() * 8760).reindex(idx, method="ffill")
    return out


def breakout_samples(data: dict, funding: dict) -> tuple[pd.DataFrame, dict]:
    pos_cache: dict = {}
    b.w_breakout_vol(data, 0.8, pos_cache)
    closes = data["open_h"].columns
    close_h = pd.concat({c: data["h"][c]["close"] for c in closes}, axis=1).sort_index()
    breadth = (close_h / close_h.shift(168) - 1 > 0).sum(axis=1) / close_h.notna().sum(axis=1).clip(lower=1)
    btc = data["h"]["BTC"]["close"]
    btc24, btc168 = (btc / btc.shift(24) - 1), (btc / btc.shift(168) - 1)
    rows, trades = [], {}
    for c in b.COINS:
        df = data["h"][c]
        o, cl, hi, lo, vol = (df[k].to_numpy() for k in ("open", "close", "high", "low", "volume"))
        idx = df.index
        pos = np.asarray(pos_cache[c])
        stress = data["stress_h"][c].reindex(idx).fillna(False).to_numpy()
        ret = pd.Series(cl, index=idx).pct_change()
        rv24 = ret.rolling(24).std() * math.sqrt(8760)
        rv720 = ret.rolling(720).std() * math.sqrt(8760)
        sma720 = pd.Series(cl, index=idx).rolling(720).mean()
        v24, v720 = pd.Series(vol, index=idx).rolling(24).sum(), pd.Series(vol, index=idx).rolling(720).mean() * 24
        N = b.BREAKOUT_PARAMS[c]["N"]
        tr = np.maximum(hi - lo, np.maximum(abs(hi - np.roll(cl, 1)), abs(lo - np.roll(cl, 1))))
        atr = pd.Series(tr, index=idx).rolling(N).mean()
        size = np.minimum(1.0, 0.8 / rv720.where(rv720 > 0).to_numpy())
        cs = b.COST[c]
        t = 1
        while t < len(pos) - 1:
            if pos[t] == 1 and pos[t - 1] == 0:
                e = t
                while e < len(pos) and pos[e] == 1:
                    e += 1
                if e >= len(pos) - 1:
                    break
                exit_fill, entry_fill = e + 1, t + 1     # fills at the next bar opens
                c_in = cs * (b.STRESS_MULT if stress[t] else 1.0)
                c_out = cs * (b.STRESS_MULT if stress[e] else 1.0)
                net = o[exit_fill] / o[entry_fill] * (1 - c_in) * (1 - c_out) - 1
                ts = idx[t]
                rows.append({
                    "coin": c, "t": ts, "t_label": idx[exit_fill], "y": int(net > 0), "net": net,
                    "rv24": rv24.iloc[t], "rv720": rv720.iloc[t], "rv_ratio": rv24.iloc[t] / rv720.iloc[t],
                    "r24": cl[t] / cl[t - 24] - 1 if t >= 24 else np.nan,
                    "r72": cl[t] / cl[t - 72] - 1 if t >= 72 else np.nan,
                    "r168": cl[t] / cl[t - 168] - 1 if t >= 168 else np.nan,
                    "vs_sma720": cl[t] / sma720.iloc[t] - 1, "vol_ratio": v24.iloc[t] / v720.iloc[t],
                    "atr_pct": atr.iloc[t] / cl[t], "hour": ts.hour,
                    "btc_r24": btc24.asof(ts), "btc_r168": btc168.asof(ts), "breadth": breadth.asof(ts),
                    "funding": funding[c].iloc[t] if c in funding else np.nan,
                    "size": 0.0 if np.isnan(size[t]) else size[t], "i": t, "e": e,
                })
                t = e
            t += 1
    return pd.DataFrame(rows), pos_cache


FEAT_META = ["rv24", "rv720", "rv_ratio", "r24", "r72", "r168", "vs_sma720", "vol_ratio", "atr_pct",
             "hour", "btc_r24", "btc_r168", "breadth", "funding"]


def meta_weights(data: dict, S: pd.DataFrame, p: pd.Series, theta: float) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=data["open_h"].index, columns=data["open_h"].columns)
    for c, grp in S.groupby("coin"):
        idx = data["h"][c].index
        col = np.zeros(len(idx))
        for k, row in grp.iterrows():
            if p.get(k, np.nan) >= theta:
                col[int(row["i"]):int(row["e"])] = row["size"] / len(b.COINS)
        w.loc[idx, c] = col
    return w


# --- strategy 6: daily gradient-boosted coin model ----------------------------------------

FEAT_DAILY = ["r1", "r3", "r7", "r14", "r28", "rv7", "rv30", "rv_ratio", "vol_ratio", "vs_high28",
              "vs_sma50", "btc_r7", "btc_r28", "breadth", "funding", "dow"]


def daily_samples(data: dict) -> pd.DataFrame:
    close, opens = data["close_d"], data["open_d"]
    vol = pd.concat({c: data["d"][c]["volume"] for c in b.COINS}, axis=1).reindex(close.index)
    high = pd.concat({c: data["d"][c]["high"] for c in b.COINS}, axis=1).reindex(close.index)
    ret = close.pct_change()
    breadth = (close / close.shift(28) - 1 > 0).sum(axis=1) / close.notna().sum(axis=1).clip(lower=1)
    btc = close["BTC"]
    rows = []
    fwd = opens.shift(-4) / opens.shift(-1) - 1
    for c in b.COINS:
        f = pd.DataFrame({
            "r1": close[c].pct_change(1), "r3": close[c].pct_change(3), "r7": close[c].pct_change(7),
            "r14": close[c].pct_change(14), "r28": close[c].pct_change(28),
            "rv7": ret[c].rolling(7).std() * math.sqrt(365), "rv30": ret[c].rolling(30).std() * math.sqrt(365),
            "vol_ratio": vol[c].rolling(7).mean() / vol[c].rolling(30).mean(),
            "vs_high28": close[c] / high[c].rolling(28).max() - 1, "vs_sma50": close[c] / close[c].rolling(50).mean() - 1,
            "btc_r7": btc.pct_change(7), "btc_r28": btc.pct_change(28), "breadth": breadth,
            "funding": data["fund_d"][c], "dow": close.index.dayofweek,
        })
        f["rv_ratio"] = f["rv7"] / f["rv30"]
        f["coin"], f["t"] = c, close.index
        f["t_label"] = pd.Series(close.index, index=close.index).shift(-4)
        f["fwd"] = fwd[c]
        f = f[close[c].notna() & f["fwd"].notna() & f["r28"].notna()]
        rows.append(f)
    S = pd.concat(rows, ignore_index=True)
    S["y"] = (S["fwd"] > 0).astype(int)
    return S


def daily_weights(data: dict, S: pd.DataFrame, p: pd.Series, theta: float) -> pd.DataFrame:
    vol30 = data["close_d"].pct_change().rolling(30).std() * math.sqrt(365)
    size = (0.6 / vol30).clip(upper=1.0) / len(b.COINS)
    take = S.loc[p.reindex(S.index) >= theta, ["t", "coin"]]
    w = pd.DataFrame(0.0, index=data["close_d"].index, columns=data["close_d"].columns)
    for (t, c) in take.itertuples(index=False):
        w.at[t, c] = size.at[t, c] if pd.notna(size.at[t, c]) else 0.0
    return w


# --- run --------------------------------------------------------------------------------------

def evaluate(name, S, feats, weights_fn, opens, stress, bpy, bh, bh_dd, data) -> dict:
    X, y, ts, tl = S[feats].replace([np.inf, -np.inf], np.nan), S["y"], S["t"], S["t_label"]
    preds, mods = {}, {}
    for d in (2, 3):
        preds[d], mods[d] = walk_predict(X, y, ts, tl, d)
    is_sharpes = []
    for g in GRIDS[name]:
        net, _ = b.simulate(opens, weights_fn(data, S, preds[g["depth"]], g["theta"]), stress)
        is_sharpes.append(sharpe(window(net, *IS_TEST), bpy))
    best = GRIDS[name][int(np.argmax(is_sharpes))]
    p = preds[best["depth"]]
    w = weights_fn(data, S, p, best["theta"])
    net_is, _ = b.simulate(opens, w, stress)
    net_is = window(net_is, *IS_TEST)
    o = b.oos_eval(w, opens, stress, bpy, bh)
    o.pop("_net")
    is_mask, oos_mask = (ts >= IS_TEST[0]) & (ts < IS_END), (ts >= OOS_START) & (ts < OOS_END)
    out = {"grid_is_sharpe": dict(zip([str(g) for g in GRIDS[name]], is_sharpes)), "final": best,
           "in_sample": {"return": float((1 + net_is).prod() - 1), "sharpe": sharpe(net_is, bpy),
                         "model": scores(p, y, is_mask)},
           "oos": o, "oos_model": scores(p, y, oos_mask),
           "taken_share_oos": float((p[oos_mask] >= best["theta"]).mean()) if oos_mask.any() else None,
           "importances_2024Q4": importances(mods[best["depth"]], X, y, ts)}
    tier = b.tier(o, bh_dd)
    if name == "meta_breakout":
        adds = o["sharpe"] > UNFILTERED["sharpe"] and o["max_dd"] > UNFILTERED["max_dd"]
        out["beats_unfiltered"] = adds
        tier = "retire" if not adds else ("watch" if tier == "candidate" else tier)
    out["tier"] = tier
    return out


def main() -> None:
    ch = get_clickhouse_client()
    data = b.load(ch)
    od, oh, sd, sh = data["open_d"], data["open_h"], data["stress_d"], data["stress_h"]
    bh = {}
    for bpy, opens in ((365, od), (8760, oh)):
        r = opens.shift(-1) / opens - 1
        bh[bpy] = window(r.mean(axis=1, skipna=True).fillna(0.0), OOS_START, OOS_END)
    from src.research.intraday_study import max_drawdown
    bh_dd = max_drawdown(bh[365])

    funding = hourly_funding(ch, {c: data["h"][c].index for c in b.COINS})
    S_meta, _ = breakout_samples(data, funding)
    S_meta = S_meta.reset_index(drop=True)
    S_day = daily_samples(data)
    results = {
        "meta_breakout": evaluate("meta_breakout", S_meta, FEAT_META, meta_weights, oh, sh, 8760, bh, bh_dd, data),
        "gbm_daily": evaluate("gbm_daily", S_day, FEAT_DAILY, daily_weights, od, sd, 365, bh, bh_dd, data),
    }
    results["meta_breakout"]["samples"] = {"total": len(S_meta),
                                           "in_sample": int((S_meta["t"] < IS_END).sum()),
                                           "oos": int((S_meta["t"] >= OOS_START).sum()),
                                           "win_rate_all": float(S_meta["y"].mean())}
    out = {"results": results, "unfiltered_breakout_oos": UNFILTERED, "buy_and_hold_max_dd": bh_dd,
           "trials_added": 12, "trial_total": 349}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    for name, r in results.items():
        i, o, m = r["in_sample"], r["oos"], r["oos_model"]
        print(f"\n{name}: final {r['final']} | IS 2023-24 return {i['return']:+.1%} sharpe {i['sharpe']:+.2f} "
              f"| IS model AUC {i['model']['auc'] and round(i['model']['auc'], 3)} n {i['model']['n']}")
        print("  grid IS sharpe:", {k: round(v, 2) for k, v in r["grid_is_sharpe"].items()})
        print(f"  OOS return {o['return']:+.1%} sharpe {o['sharpe']:+.2f} maxDD {o['max_dd']:.1%} "
              f"underwater {o['underwater_days']:.0f}d costs x2 {o['costs_x2_return']:+.1%} exposure {o['exposure']:.0%} "
              f"turnover {o['turnover_per_year']:.1f}/yr")
        print(f"  OOS model AUC {m['auc'] and round(m['auc'], 3)} accuracy {m['accuracy'] and round(m['accuracy'], 3)} "
              f"base rate {m.get('base_rate') and round(m['base_rate'], 3)} n {m['n']} | taken share {r['taken_share_oos']}")
        print("  per coin:", {c: f"{v:+.1%}" for c, v in o["per_coin_contribution"].items()})
        print("  events:", {k: f"{v['strategy']:+.1%} vs {v['buy_and_hold']:+.1%}" for k, v in o["events"].items()})
        print("  top features:", list(r["importances_2024Q4"].items())[:5])
        if "beats_unfiltered" in r:
            print(f"  beats unfiltered breakout (Sharpe > 0.90 and DD < 20.1%): {r['beats_unfiltered']}")
        print(f"  TIER: {r['tier'].upper()}")


if __name__ == "__main__":
    main()
