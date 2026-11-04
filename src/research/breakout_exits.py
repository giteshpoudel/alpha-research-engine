"""Tighter exits for the vol-sized breakout, entries frozen.

Pre-registered in docs/prereg/2026-11-04-breakout-exits.md.

    PYTHONPATH=. python -m src.research.breakout_exits
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import get_clickhouse_client
from src.research import batch1 as b1
from src.research.intraday_strategies import atr, realized_vol
from src.research.intraday_study import IS_START, OOS_END, OOS_START, STRESS_MULT, max_drawdown, stress_flags, window

# frozen BREAKOUT_VOL_PARAMS (chorus-howl signals.py)
PARAMS = {
    "BTC": dict(N=96, vratio=1.0, k=3.0, T=192),
    "ETH": dict(N=48, vratio=1.3, k=3.0, T=96),
    "SOL": dict(N=24, vratio=1.3, k=3.0, T=48),
    "XRP": dict(N=24, vratio=1.0, k=3.0, T=24),
    "DOGE": dict(N=48, vratio=1.0, k=2.0, T=48),
    "LINK": dict(N=48, vratio=1.0, k=3.0, T=48),
}
COINS = list(PARAMS)
COST = {"BTC": 5e-4, "ETH": 5e-4, "SOL": 6e-4, "XRP": 5e-4, "DOGE": 7e-4, "LINK": 8e-4}
VOL_TARGET, VOL_WINDOW = 0.80, 720
LOCK_TRIGGER_ATR, LOCK_GIVEBACK = 1.0, 0.5
VARIANTS = {"baseline": None, "tight15": 1.5, "tight10": 1.0, "lock": "lock"}
WINDOWS = {"oos_2025_26": (OOS_START, OOS_END), "pre_2022_24": (IS_START, OOS_START)}
OUT = Path("data/research/breakout_exits.json")


def breakout_exit(h, l, c, N: int, vratio: float, k: float, T: int, variant=None) -> np.ndarray:
    """vol_breakout with the same entry; the exit depends on `variant`:
    None = original (k ATR trailing), a number = trailing at that many ATR,
    "lock" = original stop plus a half-giveback profit lock once up 1 ATR."""
    n = len(c)
    prior_high = pd.Series(h).shift(1).rolling(N, min_periods=N).max().to_numpy()
    ratio = realized_vol(c, max(2, N // 4)) / realized_vol(c, 4 * N)
    a = atr(h, l, c, N)
    kk = k if variant in (None, "lock") else float(variant)
    pos = np.zeros(n)
    inpos, peak, held, entry, armed = False, 0.0, 0, 0.0, False
    for t in range(n):
        if inpos:
            held += 1
            peak = max(peak, c[t])
            stop = c[t] < peak - kk * a[t] or held >= T
            if variant == "lock":
                armed = armed or peak >= entry + LOCK_TRIGGER_ATR * a[t]
                if armed and c[t] <= entry + LOCK_GIVEBACK * (peak - entry):
                    stop = True
            if stop:
                inpos = False
        elif (not np.isnan(prior_high[t]) and not np.isnan(ratio[t]) and not np.isnan(a[t])
              and c[t] > prior_high[t] and ratio[t] > vratio):
            inpos, peak, held, entry, armed = True, c[t], 0, c[t], False
        pos[t] = 1.0 if inpos else 0.0
    return pos


def weights(df: pd.DataFrame, coin: str, variant) -> np.ndarray:
    """Vol-sized weight decided at each hourly close (size fixed at entry), 1/6 per coin."""
    pos = breakout_exit(*[df[x].to_numpy() for x in ("high", "low", "close")], **PARAMS[coin], variant=variant)
    vol = (df["close"].pct_change().rolling(VOL_WINDOW).std() * math.sqrt(8760)).to_numpy()
    size = np.minimum(1.0, VOL_TARGET / np.where(vol > 0, vol, np.nan))
    w = np.zeros(len(df))
    for t in range(len(df)):
        if pos[t] == 1:
            w[t] = w[t - 1] if t and pos[t - 1] == 1 else (0.0 if np.isnan(size[t]) else size[t])
    return w / len(COINS)


def simulate(opens: pd.DataFrame, w: pd.DataFrame, stress: pd.DataFrame, mult: float = 1.0):
    """Batch-1 convention: weight decided at bar t's close is held over bar t+1."""
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    held = held.where(opens.notna() & opens.shift(-1).notna(), 0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = pd.DataFrame(np.where(stress.shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0),
                         index=opens.index, columns=opens.columns)
    side_cost = smult * pd.Series(COST)[opens.columns].to_numpy() * mult
    return held * r - change * side_cost, held, r, side_cost


def trades(held: pd.DataFrame, r: pd.DataFrame, side_cost: pd.DataFrame, start, end) -> dict:
    """Per-trade net return (price move over the held hours minus both sides' costs)."""
    rets, hours = [], []
    for c in held.columns:
        h, rr, sc = held[c].to_numpy(), r[c].to_numpy(), side_cost[c].to_numpy()
        idx = held.index
        t = 1
        while t < len(h):
            if h[t] > 0 and h[t - 1] == 0:
                s = t
                while t < len(h) and h[t] > 0:
                    t += 1
                if start <= idx[s] < end:
                    exit_cost = sc[t] if t < len(h) else 0.0
                    rets.append(float(np.prod(1 + rr[s:t]) - 1 - sc[s] - exit_cost))
                    hours.append(t - s)
            else:
                t += 1
    x = np.array(rets)
    wins, losses = x[x > 0], x[x <= 0]
    return {"trades": len(x), "win_rate": float(len(wins) / len(x)) if len(x) else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "avg_trade": float(x.mean()) if len(x) else 0.0,
            "avg_hours": float(np.mean(hours)) if hours else 0.0}


def decide(res: dict) -> tuple[dict, str]:
    base = res["baseline"]
    checks = {}
    for v in ("tight15", "tight10", "lock"):
        o, b = res[v]["oos_2025_26"], base["oos_2025_26"]
        checks[v] = {
            "sharpe_and_return": o["sharpe"] >= b["sharpe"] + 0.10 and o["return"] > b["return"],
            "costs_x2": o["costs_x2_return"] > b["costs_x2_return"],
            "check_window": res[v]["pre_2022_24"]["return"] >= base["pre_2022_24"]["return"],
        }
        checks[v]["pass"] = all(checks[v].values())
    passing = [v for v in checks if checks[v]["pass"]]
    winner = max(passing, key=lambda v: res[v]["oos_2025_26"]["sharpe"]) if passing else "baseline"
    return checks, winner


def main() -> None:
    ch = get_clickhouse_client()
    px = {c: load_ohlcv(ch, c, "1h", start=IS_START, end=OOS_END) for c in COINS}
    opens = pd.concat({c: px[c]["open"] for c in COINS}, axis=1).sort_index()
    stress = pd.concat({c: pd.Series(stress_flags(px[c], "1h"), index=px[c].index) for c in COINS},
                       axis=1).reindex(opens.index).fillna(False).astype(bool)
    res = {}
    for name, variant in VARIANTS.items():
        W = pd.concat({c: pd.Series(weights(px[c], c, variant), index=px[c].index) for c in COINS},
                      axis=1).reindex(opens.index).fillna(0.0)
        pc, held, r, sc = simulate(opens, W, stress)
        pc2, *_ = simulate(opens, W, stress, mult=2.0)
        res[name] = {}
        for label, (a, b) in WINDOWS.items():
            net, net2 = window(pc.sum(axis=1), a, b), window(pc2.sum(axis=1), a, b)
            h = held[(held.index >= a) & (held.index < b)]
            s = b1.stats(net, h, 8760)
            s["costs_x2_return"] = float((1 + net2).prod() - 1)
            s.update(trades(held, r, sc, a, b))
            res[name][label] = s
    checks, winner = decide(res)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"params": PARAMS, "costs": COST, "results": res, "checks": checks,
                               "winner": winner, "trials_added": 3, "trial_total": 402}, indent=2, default=str))
    for name in res:
        for label, s in res[name].items():
            print(f"{name:9s} {label}: return {s['return']:+.1%} sharpe {s['sharpe']:+.2f} maxDD {s['max_dd']:.1%} "
                  f"costs x2 {s['costs_x2_return']:+.1%} | trades {s['trades']} win {s['win_rate']:.0%} "
                  f"avg win {s['avg_win']:+.2%} avg loss {s['avg_loss']:+.2%} avg {s['avg_trade']:+.2%} "
                  f"hours {s['avg_hours']:.0f}")
    print(json.dumps(checks, indent=1))
    print(f"\nEXIT: {winner}")


if __name__ == "__main__":
    main()
