"""Alt-coin 1h volatility breakout test. Implements docs/prereg/2026-11-03-alt-breakout.md.

    PYTHONPATH=. python -m src.research.alt_breakout

Output: data/research/alt_breakout.json and a printed summary. OOS runs once.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import get_clickhouse_client
from src.research.intraday_strategies import GRIDS, vol_breakout
from src.research.intraday_study import (
    FOLD_TESTS, IS_END, IS_START, OOS_END, OOS_START, UTC, max_drawdown, run, sharpe,
    stress_flags, window,
)

COINS = ("XRP", "DOGE", "LINK")
SIDE_COST = {"XRP": 5e-4, "DOGE": 7e-4, "LINK": 8e-4}  # fee 2 + p95 half-spread + slip 2 bps
BPY = 8760
GRID = GRIDS["vol_breakout"]
EVENTS = {"2025-10-10 crash": "2025-10-10", "Binance hack": "2026-10-29"}
OUT = Path("data/research/alt_breakout.json")


def positions(df: pd.DataFrame, params: dict) -> np.ndarray:
    return vol_breakout(*[df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")],
                        **params)


def main() -> None:
    ch = get_clickhouse_client()
    data = {c: load_ohlcv(ch, c, "1h", start=IS_START, end=OOS_END) for c in COINS}
    stress = {c: stress_flags(data[c], "1h") for c in COINS}
    trials = {(c, i): run(data[c], positions(data[c], p), SIDE_COST[c], stress[c])
              for c in COINS for i, p in enumerate(GRID)}
    pos_cache = {(c, i): positions(data[c], p) for c in COINS for i, p in enumerate(GRID)}

    # in-sample walk-forward
    stitched, chosen, is_trades = {}, {}, []
    for c in COINS:
        parts = []
        for test_start in FOLD_TESTS:
            train_start = test_start - pd.DateOffset(months=12)
            if len(window(data[c]["close"], train_start, test_start)) < 0.8 * BPY:
                continue  # not enough history for a 12-month train window
            test_end = test_start + pd.DateOffset(months=3)
            best = max(range(len(GRID)), key=lambda i: sharpe(
                window(trials[(c, i)].net, train_start, test_start), BPY))
            chosen.setdefault(c, []).append(GRID[best])
            parts.append(window(trials[(c, best)].net, test_start, test_end))
            mask = (data[c].index >= test_start) & (data[c].index < test_end)
            is_trades += run(data[c][mask], pos_cache[(c, best)][mask], SIDE_COST[c],
                             stress[c][mask]).trades
        if parts:
            stitched[c] = pd.concat(parts)
    is_port = pd.concat(stitched, axis=1).mean(axis=1).fillna(0)
    is_folds = [float((1 + window(is_port, t, t + pd.DateOffset(months=3))).prod() - 1)
                for t in FOLD_TESTS]

    # final params (best 2024 net Sharpe) and one OOS run
    final = {c: max(range(len(GRID)), key=lambda i: sharpe(
        window(trials[(c, i)].net, datetime(2024, 1, 1, tzinfo=UTC), IS_END), BPY)) for c in COINS}

    def oos(mult: float) -> dict:
        rets, per, trades = {}, {}, []
        for c in COINS:
            df = data[c]
            mask = (df.index >= OOS_START) & (df.index < OOS_END)
            r = run(df[mask], pos_cache[(c, final[c])][mask], SIDE_COST[c] * mult, stress[c][mask])
            rets[c], trades = r.net, trades + r.trades
            per[c] = {"return": float((1 + r.net).prod() - 1), "sharpe": sharpe(r.net, BPY),
                      "max_dd": max_drawdown(r.net), "trades": len(r.trades), "exposure": r.exposure}
        port = pd.concat(rets, axis=1).fillna(0).mean(axis=1)
        return {"return": float((1 + port).prod() - 1), "sharpe": sharpe(port, BPY),
                "max_dd": max_drawdown(port), "trades": len(trades),
                "mean_trade": float(np.mean(trades)) if trades else 0.0,
                "win_rate": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
                "per_coin": per, "_port": port}

    base, x2 = oos(1.0), oos(2.0)
    bh = pd.concat({c: window(data[c]["open"].pct_change().shift(-1).fillna(0), OOS_START, OOS_END)
                    for c in COINS}, axis=1).fillna(0).mean(axis=1)
    exposure = float(np.mean([v["exposure"] for v in base["per_coin"].values()]))
    events = {}
    for name, day in EVENTS.items():
        d = pd.Timestamp(day, tz=UTC)
        w = lambda s: window(s, d - pd.Timedelta(days=1), d + pd.Timedelta(days=3))
        events[name] = {"strategy": float((1 + w(base["_port"])).prod() - 1),
                        "buy_and_hold": float((1 + w(bh)).prod() - 1)}
    retire = base["return"] <= 0 or base["sharpe"] < 0.5 or base["max_dd"] <= -0.20
    out = {
        "coins": COINS, "side_cost": SIDE_COST,
        "in_sample": {"sharpe": sharpe(is_port, BPY), "return": float((1 + is_port).prod() - 1),
                      "max_dd": max_drawdown(is_port), "folds": is_folds,
                      "folds_positive": sum(f > 0 for f in is_folds), "trades": len(is_trades),
                      "mean_trade": float(np.mean(is_trades)) if is_trades else 0.0,
                      "chosen": chosen},
        "final_params": {c: GRID[i] for c, i in final.items()},
        "oos": {k: v for k, v in base.items() if k != "_port"},
        "oos_cost_x2": {k: v for k, v in x2.items() if k != "_port"},
        "buy_and_hold": {"return": float((1 + bh).prod() - 1), "sharpe": sharpe(bh, BPY),
                         "max_dd": max_drawdown(bh)},
        "bh_exposure_matched_return": float((1 + bh * exposure).prod() - 1), "exposure": exposure,
        "events": events,
        "decision": "retire" if retire else "shadow candidates",
        "shadow_candidates": [] if retire else [c for c, v in base["per_coin"].items() if v["return"] > 0],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    i = out["in_sample"]
    print(f"IN-SAMPLE walk-forward: return {i['return']:+.1%} sharpe {i['sharpe']:+.2f} maxDD {i['max_dd']:.1%} "
          f"folds+ {i['folds_positive']}/8 trades {i['trades']} per-trade {i['mean_trade'] * 1e4:+.1f}bps")
    print("final params:", out["final_params"])
    for label, o in (("OOS base", base), ("OOS costs x2", x2)):
        print(f"{label:13} return {o['return']:+.1%} sharpe {o['sharpe']:+.2f} maxDD {o['max_dd']:.1%} "
              f"trades {o['trades']} per-trade {o['mean_trade'] * 1e4:+.1f}bps win {o['win_rate']:.0%}")
    for c, v in base["per_coin"].items():
        print(f"  {c}: return {v['return']:+.1%} sharpe {v['sharpe']:+.2f} maxDD {v['max_dd']:.1%} trades {v['trades']}")
    b = out["buy_and_hold"]
    print(f"buy-and-hold  return {b['return']:+.1%} sharpe {b['sharpe']:+.2f} maxDD {b['max_dd']:.1%} | "
          f"exposure-matched {out['bh_exposure_matched_return']:+.1%} (in market {exposure:.0%})")
    for n, e in events.items():
        print(f"  {n}: strategy {e['strategy']:+.1%} vs buy-and-hold {e['buy_and_hold']:+.1%}")
    print(f"DECISION: {out['decision']} {out['shadow_candidates']}")


if __name__ == "__main__":
    main()
