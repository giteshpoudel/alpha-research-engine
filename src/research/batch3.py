"""Batch 3: does Kraken taker order flow predict BTC/ETH/SOL returns 4-12h ahead?

Pre-registered in docs/prereg/2026-11-03-batch3-order-flow.md (commit 88d678d) before any
result was computed. Run once:

    PYTHONPATH=. python -m src.research.batch3

Conventions (same as batches 1-2): bar b covers [b, b+1h). The decision is made at bar b's
close (time b+1h) using trades with timestamp < b+1h. The target return runs from the
open of bar b+1 to the open of bar b+1+H on Binance.US.

Implementation choices the pre-registration left open, fixed here before running:
- Large trades: the trailing-7-day 95th-percentile trade size is found from per-hour
  histograms of trade notional in 400 log-spaced bins (~3% wide), so the threshold is
  exact to within one bin.
- Trading rule z-score: for the ofi_z_* features the feature already is a z-score and is
  used directly; for the others the feature is z-scored against its own trailing 168 hourly
  values (at least 72), excluding the current hour.
- Placebo entries are drawn uniformly over the test hours where the feature is defined;
  overlapping holds merge (weight never above 1/3 per coin).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research.intraday_study import IS_START, STRESS_MULT, stress_flags

COINS = ["BTC", "ETH", "SOL"]
COST = {"BTC": 5e-4, "ETH": 5e-4, "SOL": 6e-4}
DATA_START = datetime(2026, 8, 6)
SEL_START, TEST_START, TEST_END = datetime(2026, 8, 7), datetime(2026, 10, 1), datetime(2026, 11, 3)
HORIZONS = [4, 12]
FEATURES = ["ofi_1h", "ofi_4h", "ofi_12h", "ofi_z_1h", "ofi_z_4h", "ofi_z_12h", "intensity", "large_ofi_4h"]
WEEK, MIN_HIST = 168, 72
NBINS = 400
MAX_GAP_H = 6
PLACEBO_RUNS = 1000
OUT = Path("data/batch3_results.json")


# --- data -------------------------------------------------------------------------------

def hourly_flow(ch, coin: str, start: datetime, end: datetime) -> dict:
    """Per-hour taker buy/sell notional, trade counts, and per-bin notional histograms."""
    db = database_name()
    q = (f"SELECT toStartOfHour(ts) AS h, price * qty AS n, is_buyer_maker FROM {db}.market_trades "
         f"WHERE exchange = 'kraken:USD' AND symbol = %(s)s AND ts >= %(a)s AND ts < %(b)s")
    df = ch.query_df(q, parameters={"s": coin, "a": start, "b": end})
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    hi = hours.get_indexer(pd.to_datetime(df["h"]))
    n = df["n"].to_numpy(dtype=float)
    # is_buyer_maker = 1 means the seller crossed the spread (taker sell)
    sign = np.where(df["is_buyer_maker"].to_numpy().astype(bool), -1.0, 1.0)
    logn = np.log10(np.clip(n, 1e-2, None))
    edges = np.linspace(-2, 8, NBINS + 1)              # $0.01 .. $100M
    bi = np.clip(np.searchsorted(edges, logn, side="right") - 1, 0, NBINS - 1)
    H = len(hours)
    cnt_bins = np.zeros((H, NBINS)); signed_bins = np.zeros((H, NBINS)); tot_bins = np.zeros((H, NBINS))
    np.add.at(cnt_bins, (hi, bi), 1.0)
    np.add.at(signed_bins, (hi, bi), sign * n)
    np.add.at(tot_bins, (hi, bi), n)
    return {"hours": hours, "cnt_bins": cnt_bins, "signed_bins": signed_bins, "tot_bins": tot_bins}


def features(flow: dict) -> pd.DataFrame:
    """All 8 features, indexed by bar b (uses trades in hours <= b only)."""
    hours, cb, sb, tb = flow["hours"], flow["cnt_bins"], flow["signed_bins"], flow["tot_bins"]
    signed = pd.Series(sb.sum(1), hours)
    total = pd.Series(tb.sum(1), hours)
    count = pd.Series(cb.sum(1), hours)
    f = pd.DataFrame(index=hours)
    for w in (1, 4, 12):
        tot = total.rolling(w, min_periods=w).sum()
        f[f"ofi_{w}h"] = signed.rolling(w, min_periods=w).sum() / tot.where(tot > 0)
    for w in (1, 4, 12):
        x = f[f"ofi_{w}h"]
        prev = x.shift(1).rolling(WEEK, min_periods=MIN_HIST)
        f[f"ofi_z_{w}h"] = (x - prev.mean()) / prev.std()
    med = count.shift(1).rolling(WEEK, min_periods=MIN_HIST).median()
    f["intensity"] = (count / med.where(med > 0)) * np.sign(f["ofi_1h"])
    # large trades: threshold = trailing-7-day 95th percentile of trade size (hours b-167..b)
    cum = np.cumsum(cb, axis=0)
    win = cum - np.vstack([np.zeros((WEEK, NBINS)), cum[:-WEEK]])
    large_signed = np.full(len(hours), np.nan); large_total = np.full(len(hours), np.nan)
    thr_bin = np.full(len(hours), -1)
    for i in range(len(hours)):
        c = win[i]
        if i + 1 < MIN_HIST or c.sum() == 0:
            continue
        thr_bin[i] = int(np.searchsorted(np.cumsum(c) / c.sum(), 0.95))
    for i in range(len(hours)):
        if i < 3 or thr_bin[i] < 0:
            continue
        t = thr_bin[i]
        large_signed[i] = sb[i - 3:i + 1, t:].sum()
        large_total[i] = tb[i - 3:i + 1, t:].sum()
    f["large_ofi_4h"] = large_signed / np.where(large_total > 0, large_total, np.nan)
    f["_count"] = count
    return f


def gap_hours(count: pd.Series) -> pd.Index:
    """Hours inside runs of more than MAX_GAP_H consecutive hours with no trades."""
    zero = (count == 0).astype(int)
    run_id = (zero.diff() != 0).cumsum()
    run_len = zero.groupby(run_id).transform("sum")
    return count.index[(zero == 1) & (run_len > MAX_GAP_H)]


# --- part A -----------------------------------------------------------------------------

def targets(opens: pd.Series, H: int) -> pd.Series:
    """For bar b: open[b+1+H] / open[b+1] - 1."""
    return opens.shift(-(1 + H)) / opens.shift(-1) - 1


def ic_stats(feats: dict, tgts: dict, name: str, H: int, start, end) -> dict:
    """Spearman IC pooled across coins on non-overlapping samples (every H-th hour)."""
    xs, ys = [], []
    for c in COINS:
        x, y = feats[c][name], tgts[(c, H)]
        idx = x.index[(x.index >= start) & (x.index < end)]
        idx = idx[::H]
        d = pd.DataFrame({"x": x.reindex(idx), "y": y.reindex(idx)}).dropna()
        xs.append(d["x"]); ys.append(d["y"])
    x, y = pd.concat(xs), pd.concat(ys)
    n = len(x)
    ic = float(spearmanr(x, y).correlation) if n > 2 else float("nan")
    t = ic * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ic * ic))
    return {"feature": name, "H": H, "ic": ic, "t": float(t), "n": n}


# --- part B -----------------------------------------------------------------------------

def signal_z(x: pd.Series, name: str) -> pd.Series:
    if name.startswith("ofi_z_"):
        return x
    prev = x.shift(1).rolling(WEEK, min_periods=MIN_HIST)
    return (x - prev.mean()) / prev.std()


def entries_from_signal(z: pd.Series, sign: float, H: int, start, end) -> list[int]:
    """Positions (in z's index) of entries: |z| beyond 1 in the frozen direction, no overlap."""
    idx = np.where((z.index >= start) & (z.index < end))[0]
    hit = (z.to_numpy() * sign) > 1.0
    out, busy_until = [], -1
    for i in idx:
        if i > busy_until and hit[i]:
            out.append(i)
            busy_until = i + H - 1
    return out


def weights_from_entries(n: int, entries, H: int) -> np.ndarray:
    w = np.zeros(n)
    for i in entries:
        w[i:i + H] = 1.0 / 3
    return w


def simulate(opens: pd.DataFrame, w: pd.DataFrame, stress: pd.DataFrame, mult: float = 1.0) -> pd.Series:
    """Weight decided at bar b's close is held over bar b+1 (open b+1 -> open b+2)."""
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = np.where(stress.shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0)
    cost = change * pd.Series(COST)[opens.columns].to_numpy() * smult * mult
    return (held * r - cost).sum(axis=1)


def total_return(net: pd.Series, start, end) -> float:
    s = net[(net.index >= start) & (net.index < end)]
    return float((1 + s).prod() - 1)


# --- run --------------------------------------------------------------------------------

def run() -> dict:
    ch = get_clickhouse_client()
    feats, tgts, opens, stress, gaps = {}, {}, {}, {}, {}
    for c in COINS:
        flow = hourly_flow(ch, c, DATA_START, TEST_END)
        f = features(flow)
        g = gap_hours(f["_count"])
        gaps[c] = int(((g >= TEST_START) & (g < TEST_END)).sum())
        f.loc[f.index.isin(g), FEATURES] = np.nan
        feats[c] = f
        px = load_ohlcv(ch, c, "1h", start=IS_START, end=TEST_END)
        st = pd.Series(stress_flags(px, "1h"), index=px.index)
        hours = f.index
        opens[c] = px["open"].reindex(hours)
        stress[c] = st.reindex(hours).fillna(False).astype(bool)
        for H in HORIZONS:
            tgts[(c, H)] = targets(px["open"], H).reindex(hours)
    opens_df, stress_df = pd.DataFrame(opens), pd.DataFrame(stress)

    sel = [ic_stats(feats, tgts, n, H, SEL_START, TEST_START) for n in FEATURES for H in HORIZONS]
    test = [ic_stats(feats, tgts, n, H, TEST_START, TEST_END) for n in FEATURES for H in HORIZONS]
    pick = max(sel, key=lambda r: abs(r["t"]))
    sign = float(np.sign(pick["ic"]))
    pt = next(r for r in test if r["feature"] == pick["feature"] and r["H"] == pick["H"])
    a_pass = bool(np.sign(pt["ic"]) == sign and abs(pt["ic"]) >= 0.03 and pt["t"] * sign >= 2.0)
    out = {"selected": pick, "direction": "momentum" if sign > 0 else "reversal", "test": pt,
           "part_a_pass": a_pass, "selection_all": sel, "test_all": test, "gap_hours_test": gaps,
           "trades_test": {c: int(feats[c]["_count"][(feats[c].index >= TEST_START) &
                                                      (feats[c].index < TEST_END)].sum()) for c in COINS}}

    # Part B is computed only when Part A passes, as pre-registered.
    if a_pass:
        name, H = pick["feature"], pick["H"]
        n = len(opens_df)
        ent, w = {}, {}
        for c in COINS:
            z = signal_z(feats[c][name], name)
            ent[c] = entries_from_signal(z, sign, H, TEST_START, TEST_END)
            w[c] = weights_from_entries(n, ent[c], H)
        W = pd.DataFrame(w, index=opens_df.index)
        net = simulate(opens_df, W, stress_df)
        net2 = simulate(opens_df, W, stress_df, mult=2.0)
        ret, ret2 = total_return(net, TEST_START, TEST_END), total_return(net2, TEST_START, TEST_END)
        rng = np.random.default_rng(0)
        pool = {c: np.where((opens_df.index >= TEST_START) & (opens_df.index < TEST_END - pd.Timedelta(hours=H))
                            & feats[c][name].notna().to_numpy())[0] for c in COINS}
        plac = []
        for _ in range(PLACEBO_RUNS):
            Wp = pd.DataFrame({c: weights_from_entries(n, rng.choice(pool[c], size=len(ent[c]), replace=False), H)
                               for c in COINS}, index=opens_df.index)
            plac.append(total_return(simulate(opens_df, Wp, stress_df), TEST_START, TEST_END))
        p80 = float(np.percentile(plac, 80))
        out["part_b"] = {"entries": {c: len(ent[c]) for c in COINS}, "net_return": ret, "net_return_cost_x2": ret2,
                         "placebo_p80": p80, "placebo_median": float(np.median(plac)),
                         "placebo_rank": float(np.mean(np.array(plac) < ret)),
                         "pass": bool(ret > 0 and ret2 > 0 and ret > p80)}
        out["outcome"] = "watch" if out["part_b"]["pass"] else "retire as a strategy (keep feature)"
    else:
        out["outcome"] = "retire"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    r = run()
    print(f"selected: {r['selected']['feature']} H={r['selected']['H']} ({r['direction']}), "
          f"selection IC {r['selected']['ic']:+.4f} t {r['selected']['t']:+.2f} n {r['selected']['n']}")
    print(f"test: IC {r['test']['ic']:+.4f} t {r['test']['t']:+.2f} n {r['test']['n']} -> Part A "
          f"{'PASS' if r['part_a_pass'] else 'FAIL'}")
    print(f"gap hours in test: {r['gap_hours_test']}")
    print("\nall pairs (selection IC / t  |  test IC / t):")
    for s, t in zip(r["selection_all"], r["test_all"]):
        print(f"  {s['feature']:13} H={s['H']:<3} {s['ic']:+.4f} {s['t']:+6.2f}  |  {t['ic']:+.4f} {t['t']:+6.2f}  (n {s['n']}/{t['n']})")
    if "part_b" in r:
        print("\nPart B:", json.dumps(r["part_b"], indent=2))
    print(f"\nOUTCOME: {r['outcome']}")


if __name__ == "__main__":
    main()
