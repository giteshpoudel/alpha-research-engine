"""The frozen hourly breakout rule on Zcash, with a hindsight guard.

Pre-registered in docs/prereg/2026-11-12-zcash-breakout.md (commit 4d2f45c).

    PYTHONPATH=. python -m src.research.zcash_breakout
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import _utc_index, clean_ohlcv
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research import batch1 as b1
from src.research.intraday_strategies import vol_breakout
from src.research.intraday_study import IS_START, OOS_END, OOS_START, STRESS_MULT, max_drawdown, sharpe, stress_flags, window

UTC = timezone.utc
PARAMS = dict(N=48, vratio=1.0, k=3.0, T=48)
COST = 8e-4
VOL_TARGET, VOL_WINDOW = 0.80, 720
PRE = (IS_START, OOS_START)
YEARS = [(datetime(y, 1, 1, tzinfo=UTC), min(datetime(y + 1, 1, 1, tzinfo=UTC), OOS_END)) for y in range(2022, 2027)]
OUT = Path("data/research/zcash_breakout.json")


def load_coinbase(ch, symbol: str = "ZEC") -> pd.DataFrame:
    rows = ch.query(
        f"SELECT ts, open, high, low, close, volume FROM {database_name()}.ohlcv FINAL "
        "WHERE symbol = {s:String} AND interval = '1h' AND exchange = 'coinbase' "
        "AND ts >= {a:DateTime} AND ts < {b:DateTime} ORDER BY ts",
        parameters={"s": symbol, "a": IS_START.replace(tzinfo=None), "b": OOS_END.replace(tzinfo=None)},
    ).result_rows
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = _utc_index(df.pop("ts"))
    return clean_ohlcv(df, "1h")


def weights(df: pd.DataFrame) -> np.ndarray:
    """Vol-sized breakout weight decided at each hourly close (size fixed at entry), full weight."""
    pos = vol_breakout(*[df[x].to_numpy() for x in ("open", "high", "low", "close", "volume")], **PARAMS)
    vol = (df["close"].pct_change().rolling(VOL_WINDOW).std() * math.sqrt(8760)).to_numpy()
    size = np.minimum(1.0, VOL_TARGET / np.where(vol > 0, vol, np.nan))
    w = np.zeros(len(df))
    for t in range(len(df)):
        if pos[t] == 1:
            w[t] = w[t - 1] if t and pos[t - 1] == 1 else (0.0 if np.isnan(size[t]) else size[t])
    return w


def simulate(opens: pd.Series, w: pd.Series, stress: pd.Series, mult: float = 1.0):
    """Batch-1 convention: weight decided at bar t's close is held over bar t+1."""
    r = (opens.shift(-1) / opens - 1).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    smult = np.where(stress.shift(1).fillna(False).astype(bool), STRESS_MULT, 1.0)
    return held * r - change * COST * smult * mult, held, r


def trade_list(net: pd.Series, held: pd.Series, start, end) -> list[float]:
    """Net return of each trade (a run of held > 0) that starts inside [start, end)."""
    out, cur, inside = [], None, False
    for t, h in held.items():
        if h > 0:
            if cur is None:
                cur, inside = 1.0, start <= t < end
            cur *= 1 + net[t]
        elif cur is not None:
            cur *= 1 + net[t]          # exit cost lands on the first flat bar
            if inside:
                out.append(cur - 1)
            cur = None
    if cur is not None and inside:
        out.append(cur - 1)
    return out


def summarize(net, net2, held, r, start, end) -> dict:
    n, n2 = window(net, start, end), window(net2, start, end)
    h = held[(held.index >= start) & (held.index < end)]
    s = b1.stats(n, h.to_frame(), 8760)
    s["costs_x2_return"] = float((1 + n2).prod() - 1)
    tr = trade_list(net, held, start, end)
    s["entries"], s["win_rate"] = len(tr), (sum(x > 0 for x in tr) / len(tr) if tr else 0.0)
    s["avg_trade"] = float(np.mean(tr)) if tr else 0.0
    bh = window(r, start, end)
    s["buy_and_hold"] = {"return": float((1 + bh).prod() - 1), "max_dd": max_drawdown(bh), "sharpe": sharpe(bh, 8760)}
    return s


def main() -> None:
    df = load_coinbase(get_clickhouse_client())
    stress = pd.Series(stress_flags(df, "1h"), index=df.index)
    w = pd.Series(weights(df), index=df.index)
    net, held, r = simulate(df["open"], w, stress)
    net2, _, _ = simulate(df["open"], w, stress, mult=2.0)

    res = {"oos_2025_26": summarize(net, net2, held, r, OOS_START, OOS_END),
           "pre_2022_24": summarize(net, net2, held, r, *PRE)}
    years = {str(a.year): summarize(net, net2, held, r, a, b) for a, b in YEARS}
    o, p = res["oos_2025_26"], res["pre_2022_24"]
    t = b1.tier(o, o["buy_and_hold"]["max_dd"])
    guard = p["return"] > 0 and p["costs_x2_return"] > 0
    if not guard:
        t = "retire"
    out = {"params": PARAMS, "cost": COST, "source": "coinbase ZEC-USD 1h", "results": res, "years": years,
           "hindsight_guard_passed": guard, "tier": t, "trials_added": 1, "trial_total": 407}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    for label, s in {**res, **years}.items():
        bh = s["buy_and_hold"]
        print(f"{label:12s}: return {s['return']:+.1%} sharpe {s['sharpe']:+.2f} maxDD {s['max_dd']:.1%} "
              f"underwater {s['underwater_days']:.0f}d exposure {s['exposure']:.2f} costs x2 {s['costs_x2_return']:+.1%} "
              f"| entries {s['entries']} win {s['win_rate']:.0%} avg {s['avg_trade']:+.2%} "
              f"| buy-and-hold {bh['return']:+.1%} maxDD {bh['max_dd']:.1%}")
    print(f"\nhindsight guard (2022-24 > 0 and > 0 with costs x2): {'passed' if guard else 'FAILED'}")
    print(f"TIER: {t.upper()}")


if __name__ == "__main__":
    main()
