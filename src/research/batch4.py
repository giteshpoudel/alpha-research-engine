"""Batch 4: dip-buying in uptrends, funding-panic entries, passive market making.

Pre-registered in docs/prereg/2026-11-04-batch4.md (commit 72bd7b3) before any result.

    PYTHONPATH=. python -m src.research.batch4

Strategies 7 and 8 reuse batch 1's loader, simulator, walk-forward and tier rules.
Strategy 9 replays the Binance.US trade tape with a conservative fill model.
"""

from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_funding
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research import batch1 as b1
from src.research.intraday_study import IS_START, OOS_END, OOS_START, max_drawdown, sharpe, window

UTC = timezone.utc
COINS = b1.COINS
VOL_TARGET = 0.80
OUT = Path("data/research/batch4.json")

GRIDS = {
    "dip_buy": [dict(L=L, k=k, Z=Z, M=M) for L, k, Z, M in itertools.product((50, 100), (1, 3), (1.5, 2.0), (3, 7))],
    "funding_fear": [dict(A=A, q=q, M=M) for A, q, M in itertools.product((1, 3), (5, 10), (7, 14))],
}

MM_COINS = ["XRP", "DOGE", "LINK"]
MM_COST = {"XRP": 5e-4, "DOGE": 7e-4, "LINK": 8e-4}
MM_LOT_USD = 10.0
MM_HALF_LIFE_S = 300.0
MM_DELAY_S = 2.0
MM_STRESS_BPS = 2e-4
MM_START, MM_SPLIT, MM_END = (datetime(2026, 9, 29, tzinfo=UTC), datetime(2026, 10, 16, tzinfo=UTC),
                              datetime(2026, 11, 3, tzinfo=UTC))
MM_GRID = [dict(h=h, C=C) for h, C in itertools.product((10e-4, 20e-4, 40e-4), (1, 3))]


# --- strategies 7 and 8: daily, long/flat, fixed hold -------------------------------------

def entry_size(close: pd.Series) -> pd.Series:
    vol = close.pct_change().rolling(30).std() * math.sqrt(365)
    return np.minimum(1.0, VOL_TARGET / vol.where(vol > 0))


def hold_weights(signal: pd.Series, size: pd.Series, M: int) -> pd.Series:
    """Weight decided at day t's close; an entry at t sets w[t..t+M-1] (held t+1..t+M)."""
    sig, sz = signal.fillna(False).to_numpy(bool), size.to_numpy()
    w = np.zeros(len(sig))
    busy_until = -1
    for t in range(len(sig)):
        if t > busy_until and sig[t] and not np.isnan(sz[t]):
            w[t:t + M] = sz[t]
            busy_until = t + M - 1
    return pd.Series(w, index=signal.index)


def dip_signal(close: pd.Series, L: int, k: int, Z: float) -> pd.Series:
    trend = close > close.rolling(L).mean()
    vol_d = close.pct_change().rolling(30).std()
    dip = (close / close.shift(k) - 1) / (vol_d * math.sqrt(k)) <= -Z
    return trend & dip


def w_dip_buy(data: dict, L: int, k: int, Z: float, M: int) -> pd.DataFrame:
    close = data["close_d"]
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for c in COINS:
        s = close[c]
        out[c] = hold_weights(dip_signal(s, L, k, Z), entry_size(s), M) / len(COINS)
    return out


def daily_funding(ch, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean hourly Hyperliquid funding rate per calendar day (UTC), per coin."""
    cols = {}
    for c in COINS:
        f = load_funding(ch, c, start=IS_START, end=OOS_END)
        cols[c] = f.resample("1D").mean()
    return pd.concat(cols, axis=1).reindex(daily_idx)


def funding_signal(fund_daily: pd.Series, A: int, q: int) -> pd.Series:
    avg = fund_daily.rolling(A, min_periods=A).mean()
    thr = avg.shift(1).rolling(90, min_periods=60).quantile(q / 100)
    return avg < thr


def w_funding_fear(data: dict, fund: pd.DataFrame, A: int, q: int, M: int) -> pd.DataFrame:
    close = data["close_d"]
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for c in COINS:
        out[c] = hold_weights(funding_signal(fund[c], A, q), entry_size(close[c]), M) / len(COINS)
    return out


# --- strategy 9: passive market making on the trade tape ----------------------------------

def load_tape(ch, coin: str, start: datetime, end: datetime) -> pd.DataFrame:
    db = database_name()
    q = (f"SELECT ts, price, qty FROM {db}.market_trades WHERE exchange = 'binance_us' AND symbol = %(s)s "
         f"AND ts >= %(a)s AND ts < %(b)s ORDER BY ts, trade_id")
    df = ch.query_df(q, parameters={"s": coin, "a": start.replace(tzinfo=None), "b": end.replace(tzinfo=None)})
    ts = pd.to_datetime(df["ts"])
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    # seconds since epoch, independent of the datetime unit (ClickHouse returns microseconds)
    df["t"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return df[["t", "price", "qty"]].reset_index(drop=True)


def lagged_fair(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Fair price usable at each trade: time-decayed average of trades >= MM_DELAY_S older."""
    fair = np.empty(len(p))
    f = p[0]
    for i in range(len(p)):
        if i:
            a = 1 - 0.5 ** ((t[i] - t[i - 1]) / MM_HALF_LIFE_S)
            f = f + a * (p[i] - f)
        fair[i] = f
    j = np.searchsorted(t, t - MM_DELAY_S, side="right") - 1      # last trade at least 2 s older
    out = np.where(j >= 0, fair[np.clip(j, 0, None)], np.nan)
    return out


def simulate_mm(tape: pd.DataFrame, h: float, C: int, cost: float, stress: float = 0.0) -> dict:
    t, p, q = tape["t"].to_numpy(), tape["price"].to_numpy(), tape["qty"].to_numpy()
    if len(p) < 2:
        return {"pnl": 0.0, "fills": 0}
    fair = lagged_fair(t, p)
    cash, inv, fills, buys, sells = 0.0, 0.0, 0, 0, 0
    for i in range(len(p)):
        if np.isnan(fair[i]):
            continue
        bid, ask = fair[i] * (1 - h), fair[i] * (1 + h)
        pos_usd = inv * fair[i]
        if p[i] < bid and pos_usd < C * MM_LOT_USD - 1e-9:
            x = min(MM_LOT_USD / bid, q[i])
            cash -= x * bid * (1 + stress); inv += x; fills += 1; buys += 1
        elif p[i] > ask and pos_usd > -C * MM_LOT_USD + 1e-9:
            x = min(MM_LOT_USD / ask, q[i])
            cash += x * ask * (1 - stress); inv -= x; fills += 1; sells += 1
    last = p[-1]
    pnl = cash + inv * last - abs(inv) * last * cost
    return {"pnl": float(pnl), "fills": fills, "buys": buys, "sells": sells, "end_inventory_usd": float(inv * last)}


def mm_period(tapes: dict, start: datetime, end: datetime, h: float, C: int, stress: float = 0.0) -> dict:
    per = {}
    for c, tape in tapes.items():
        a, b = start.timestamp(), end.timestamp()
        per[c] = simulate_mm(tape[(tape["t"] >= a) & (tape["t"] < b)].reset_index(drop=True), h, C, MM_COST[c], stress)
    total = sum(v["pnl"] for v in per.values())
    capital = C * MM_LOT_USD * 2 * len(tapes)
    return {"per_coin": per, "pnl": total, "return_on_capital": total / capital}


# --- run ----------------------------------------------------------------------------------

def main() -> None:
    ch = get_clickhouse_client()
    data = b1.load(ch)
    od, sd = data["open_d"], data["stress_d"]
    r = od.shift(-1) / od - 1
    bh = window(r.mean(axis=1, skipna=True).fillna(0.0), OOS_START, OOS_END)
    bh_stats = {"return": float((1 + bh).prod() - 1), "sharpe": sharpe(bh, 365), "max_dd": max_drawdown(bh)}
    bhd = {365: bh}
    fund = daily_funding(ch, od.index)

    specs = {"dip_buy": lambda p: w_dip_buy(data, **p),
             "funding_fear": lambda p: w_funding_fear(data, fund, **p)}
    results = {}
    for name, fn in specs.items():
        wf = b1.walk_forward(fn, od, sd, GRIDS[name], 365)
        o = b1.oos_eval(fn(wf["final"]), od, sd, 365, bhd)
        w = fn(wf["final"])
        entries = int(((w > 0) & (w.shift(1).fillna(0) == 0))[(w.index >= OOS_START) & (w.index < OOS_END)].sum().sum())
        o.pop("_net", None)
        o["entries"] = entries
        results[name] = {"in_sample": wf, "oos": o, "tier": b1.tier(o, bh_stats["max_dd"])}

    tapes = {c: load_tape(ch, c, MM_START, MM_END) for c in MM_COINS}
    sel = [{**g, **mm_period(tapes, MM_START, MM_SPLIT, g["h"], g["C"])} for g in MM_GRID]
    pick = max(MM_GRID, key=lambda g: next(s for s in sel if s["h"] == g["h"] and s["C"] == g["C"])["return_on_capital"])
    test = mm_period(tapes, MM_SPLIT, MM_END, pick["h"], pick["C"])
    stress = mm_period(tapes, MM_SPLIT, MM_END, pick["h"], pick["C"], stress=MM_STRESS_BPS)
    winners = sum(v["pnl"] > 0 for v in test["per_coin"].values())
    mm_pass = test["pnl"] > 0 and stress["pnl"] > 0 and winners >= 2
    results["passive_mm"] = {"selection": sel, "pick": pick, "test": test, "test_stress_2bps": stress,
                             "coins_profitable": winners, "tier": "watch" if mm_pass else "retire",
                             "tape_trades": {c: len(v) for c, v in tapes.items()}}

    out = {"buy_and_hold": bh_stats, "results": results, "trials_added": 30, "trial_total": 395}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"buy-and-hold 6 coins OOS: {bh_stats['return']:+.1%} sharpe {bh_stats['sharpe']:+.2f} maxDD {bh_stats['max_dd']:.1%}")
    for name in ("dip_buy", "funding_fear"):
        x = results[name]
        o, i = x["oos"], x["in_sample"]
        print(f"\n{name}: final {i['final']}  in-sample WF return {i['return']:+.1%} sharpe {i['sharpe']:+.2f} "
              f"folds+ {i['folds_positive']}/8")
        print(f"  OOS return {o['return']:+.1%} sharpe {o['sharpe']:+.2f} maxDD {o['max_dd']:.1%} "
              f"underwater {o['underwater_days']}d exposure {o['exposure']:.2f} entries {o['entries']} "
              f"costs x2 {o['costs_x2_return']:+.1%}  -> {x['tier'].upper()}")
    m = results["passive_mm"]
    print(f"\npassive_mm: tape trades {m['tape_trades']}")
    for s in sel:
        print(f"  sel h={s['h'] * 1e4:.0f}bps C={s['C']}: pnl ${s['pnl']:+.3f} roc {s['return_on_capital']:+.2%} "
              f"fills {sum(v['fills'] for v in s['per_coin'].values())}")
    print(f"  pick {m['pick']}; test pnl ${m['test']['pnl']:+.3f} roc {m['test']['return_on_capital']:+.2%}; "
          f"with +2bps per fill ${m['test_stress_2bps']['pnl']:+.3f}; coins profitable {m['coins_profitable']}/3")
    for c, v in m["test"]["per_coin"].items():
        print(f"    {c}: {v}")
    print(f"  -> {m['tier'].upper()}")


if __name__ == "__main__":
    main()
