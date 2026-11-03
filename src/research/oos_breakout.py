"""Exploratory OOS check of vol_breakout @ 1h. Implements
docs/prereg/2026-11-03-phase2-oos-exploratory.md (run once).

    PYTHONPATH=. python -m src.research.oos_breakout

Output: data/research/phase2_oos_breakout.json and a printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.schemas import get_clickhouse_client
from src.research.intraday_strategies import vol_breakout
from src.research.intraday_study import (
    KRAKEN_FEE_BPS, OOS_END, OOS_START, SYMBOLS, UTC, cost_per_side, load_bars,
    max_drawdown, run, sharpe, stress_flags, window,
)

BAR, BPY = "1h", 8760
PARAMS = {  # frozen 2024 selection (Phase 2 final_params)
    "BTC": dict(N=96, vratio=1.0, k=3.0, T=192),
    "ETH": dict(N=48, vratio=1.3, k=3.0, T=96),
    "SOL": dict(N=24, vratio=1.3, k=3.0, T=48),
}
MEASURED_SIDE_COST = 2.5e-4  # canary 2026-11-03: 2 bps fee, 0 slippage, ~0.5 bps half-spread
EVENTS = {"2025-10-10 crash": "2025-10-10", "Binance hack": "2026-10-29"}
OUT = Path("data/research/phase2_oos_breakout.json")


def scenario(data, pos, stress, side_cost) -> dict:
    rets, trades, per_symbol = {}, [], {}
    for s in SYMBOLS:
        df = data[s]
        mask = (df.index >= OOS_START) & (df.index < OOS_END)
        r = run(df[mask], pos[s][mask], side_cost(s), stress[s][mask])
        rets[s] = r.net
        trades += r.trades
        per_symbol[s] = {"return": float((1 + r.net).prod() - 1), "sharpe": sharpe(r.net, BPY),
                         "trades": len(r.trades), "exposure": r.exposure}
    port = pd.concat(rets, axis=1).fillna(0).mean(axis=1)
    quarters = {str(q): float((1 + g).prod() - 1)
                for q, g in port.groupby(port.index.tz_localize(None).to_period("Q"))}
    return {"return": float((1 + port).prod() - 1), "sharpe": sharpe(port, BPY),
            "max_dd": max_drawdown(port), "trades": len(trades),
            "mean_trade": float(np.mean(trades)) if trades else 0.0,
            "win_rate": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
            "per_symbol": per_symbol, "quarters": quarters, "_port": port}


def main() -> None:
    ch = get_clickhouse_client()
    data = {s: load_bars(ch, s)[BAR] for s in SYMBOLS}
    stress = {s: stress_flags(data[s], BAR) for s in SYMBOLS}
    pos = {s: vol_breakout(*[data[s][c].to_numpy() for c in
                             ("open", "high", "low", "close", "volume")], **PARAMS[s])
           for s in SYMBOLS}
    scen = {
        "base": scenario(data, pos, stress, lambda s: cost_per_side(s)),
        "cost_x2": scenario(data, pos, stress, lambda s: cost_per_side(s, mult=2.0)),
        "kraken_fees": scenario(data, pos, stress, lambda s: cost_per_side(s, fee_bps=KRAKEN_FEE_BPS)),
        "measured_cost": scenario(data, pos, stress, lambda s: MEASURED_SIDE_COST),
    }
    bh = pd.concat({s: window(data[s]["open"].pct_change().shift(-1).fillna(0), OOS_START, OOS_END)
                    for s in SYMBOLS}, axis=1).fillna(0).mean(axis=1)
    exposure = float(np.mean([v["exposure"] for v in scen["base"]["per_symbol"].values()]))
    base = scen["base"]
    events = {}
    for name, day in EVENTS.items():
        d = pd.Timestamp(day, tz=UTC)
        w = lambda s: window(s, d - pd.Timedelta(days=1), d + pd.Timedelta(days=3))
        events[name] = {"strategy": float((1 + w(base["_port"])).prod() - 1),
                        "buy_and_hold": float((1 + w(bh)).prod() - 1)}
    out = {
        "window": [OOS_START.isoformat(), OOS_END.isoformat()], "params": PARAMS,
        "scenarios": {k: {kk: vv for kk, vv in v.items() if kk != "_port"} for k, v in scen.items()},
        "buy_and_hold": {"return": float((1 + bh).prod() - 1), "sharpe": sharpe(bh, BPY),
                         "max_dd": max_drawdown(bh)},
        "exposure": exposure,
        "bh_exposure_matched_return": float((1 + bh * exposure).prod() - 1),
        "events": events,
    }
    retire = base["return"] <= 0 or base["sharpe"] < 0.5 or base["max_dd"] <= -0.20
    out["decision"] = "retire" if retire else "live-learning candidate"
    out["beats_bh_sharpe"] = bool(base["sharpe"] > out["buy_and_hold"]["sharpe"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    for k, v in scen.items():
        print(f"{k:14} return {v['return']:+.1%}  sharpe {v['sharpe']:+.2f}  maxDD {v['max_dd']:.1%}  "
              f"trades {v['trades']}  per-trade {v['mean_trade'] * 1e4:+.1f}bps  win {v['win_rate']:.0%}")
    b = out["buy_and_hold"]
    print(f"buy-and-hold   return {b['return']:+.1%}  sharpe {b['sharpe']:+.2f}  maxDD {b['max_dd']:.1%}"
          f" | exposure-matched {out['bh_exposure_matched_return']:+.1%} (time in market {exposure:.0%})")
    for s, v in base["per_symbol"].items():
        print(f"  {s}: return {v['return']:+.1%} sharpe {v['sharpe']:+.2f} trades {v['trades']}")
    print("  quarters:", {q: f"{r:+.1%}" for q, r in base["quarters"].items()})
    for name, e in events.items():
        print(f"  {name}: strategy {e['strategy']:+.1%} vs buy-and-hold {e['buy_and_hold']:+.1%}")
    print(f"DECISION: {out['decision']} | beats buy-and-hold Sharpe: {out['beats_bh_sharpe']}")


if __name__ == "__main__":
    main()
