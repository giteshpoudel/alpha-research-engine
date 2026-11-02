"""Phase 2 intraday study harness. Implements docs/prereg/2026-11-02-phase2-intraday.md.

    python -m src.research.intraday_study              # IS walk-forward, OOS for passers
    python -m src.research.intraday_study --no-oos     # IS only

OOS is evaluated only for (family, bar) pairs that pass the IS bar, exactly once.
Results go to data/research/phase2_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.meta_learning.selection import deflated_sharpe_ratio, per_bar_moments
from src.research.intraday_strategies import FUNCS, GRIDS, realized_vol

SYMBOLS = ("BTC", "ETH", "SOL")
BARS = {"15m": 35040, "1h": 8760}          # bars per year
FEE_BPS = 2.0
SLIP_BPS = 2.0
HALF_SPREAD_BPS = {"BTC": 1.0, "ETH": 1.0, "SOL": 2.0}
KRAKEN_FEE_BPS = 40.0
STRESS_MULT = 3.0
STRESS_QUANTILE = 0.90
N_TRIALS = 240

UTC = timezone.utc
IS_START, IS_END = datetime(2022, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
OOS_START, OOS_END = datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)
FOLD_TESTS = [pd.Timestamp(f"{y}-{m:02d}-01", tz=UTC) for y in (2023, 2024) for m in (1, 4, 7, 10)]
CRASHES = {"LUNA": "2022-05-09", "Celsius/3AC": "2022-06-13", "FTX": "2022-11-08",
           "yen carry": "2024-08-05", "2025-10-10 crash": "2025-10-10",
           "Binance hack": "2026-10-29"}

IS_PASS = dict(sharpe=1.0, folds_positive=5, trades=100, dsr=0.95, trade_cost_mult=2.0)
OOS_PASS = dict(sharpe=0.5, max_dd=0.20)


def cost_per_side(symbol: str, fee_bps: float = FEE_BPS, mult: float = 1.0) -> float:
    return (fee_bps + HALF_SPREAD_BPS[symbol] + SLIP_BPS) * mult / 1e4


# --- data ------------------------------------------------------------------------

def resample(df5: pd.DataFrame, bar: str) -> pd.DataFrame:
    rule = {"15m": "15min", "1h": "1h"}[bar]
    out = df5.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def load_bars(ch, symbol: str) -> dict[str, pd.DataFrame]:
    df5 = load_ohlcv(ch, symbol, "5m", start=IS_START, end=OOS_END)
    return {bar: resample(df5, bar) for bar in BARS}


def stress_flags(df: pd.DataFrame, bar: str) -> np.ndarray:
    """Top-decile trailing-24h realized vol, threshold fixed on IS only (causal)."""
    per_day = {"15m": 96, "1h": 24}[bar]
    rv = pd.Series(realized_vol(df["close"].to_numpy(), per_day), index=df.index)
    threshold = rv[(rv.index >= IS_START) & (rv.index < IS_END)].quantile(STRESS_QUANTILE)
    return (rv > threshold).to_numpy()


# --- backtest core ---------------------------------------------------------------

@dataclass
class Result:
    net: pd.Series          # per-bar net return, indexed by bar time
    trades: list[float]     # per-trade net returns
    exposure: float


def run(df: pd.DataFrame, pos: np.ndarray, side_cost: float, stress: np.ndarray,
        stress_mult: float = STRESS_MULT) -> Result:
    """Position decided at close t is held from open t+1 to open t+2."""
    o = df["open"].to_numpy()
    r_oo = np.append(o[1:] / o[:-1] - 1.0, 0.0)      # open t -> open t+1
    held = np.concatenate([[0.0], pos[:-1]])          # position during bar t
    prev = np.concatenate([[0.0], held[:-1]])
    stress_prev = np.concatenate([[False], stress[:-1]])
    cost = np.abs(held - prev) * side_cost * np.where(stress_prev, stress_mult, 1.0)
    net = held * r_oo - cost
    trades, start = [], None
    for t in range(len(held)):
        if held[t] == 1 and prev[t] == 0:
            start = t
        if held[t] == 0 and prev[t] == 1 and start is not None:
            trades.append(float(np.prod(1 + held[start:t] * r_oo[start:t])
                                * (1 - cost[start]) * (1 - cost[t]) - 1))
            start = None
    return Result(pd.Series(net, index=df.index), trades, float(held.mean()))


def sharpe(ret: pd.Series, bars_per_year: float) -> float:
    sd = ret.std()
    return float(ret.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0


def max_drawdown(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def window(s: pd.Series, start, end) -> pd.Series:
    return s[(s.index >= start) & (s.index < end)]


def trades_in(df: pd.DataFrame, pos: np.ndarray, side_cost: float, stress: np.ndarray,
              start, end) -> list[float]:
    mask = (df.index >= start) & (df.index < end)
    if not mask.any():
        return []
    return run(df[mask], pos[mask], side_cost, stress[mask]).trades


# --- study -----------------------------------------------------------------------

def study(ch, run_oos: bool = True) -> dict:
    data = {s: load_bars(ch, s) for s in SYMBOLS}
    out: dict = {"generated_at": datetime.now(UTC).isoformat(), "families": {}}

    for bar, bpy in BARS.items():
        # 1) every trial over the full history once (signals are causal)
        trials: dict[tuple, Result] = {}
        positions: dict[tuple, np.ndarray] = {}
        stress = {s: stress_flags(data[s][bar], bar) for s in SYMBOLS}
        for fam, grid in GRIDS.items():
            for s in SYMBOLS:
                df = data[s][bar]
                arrs = [df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")]
                for i, params in enumerate(grid):
                    pos = FUNCS[fam](*arrs, **params)
                    positions[(fam, s, i)] = pos
                    trials[(fam, s, i)] = run(df, pos, cost_per_side(s), stress[s])
        # trial dispersion for the Deflated Sharpe (this bar size's trials, full IS)
        trial_srs = [per_bar_moments(window(r.net, IS_START, IS_END))[0]
                     for r in trials.values()]
        sr_std = float(np.std(trial_srs))

        for fam, grid in GRIDS.items():
            key = f"{fam}@{bar}"
            stitched, fold_rets, all_trades, chosen = {}, [], [], {}
            for s in SYMBOLS:
                df = data[s][bar]
                parts = []
                for test_start in FOLD_TESTS:
                    train_start = test_start - pd.DateOffset(months=12)
                    test_end = test_start + pd.DateOffset(months=3)
                    best = max(range(len(grid)), key=lambda i: sharpe(
                        window(trials[(fam, s, i)].net, train_start, test_start), bpy))
                    chosen.setdefault(s, []).append(grid[best])
                    parts.append(window(trials[(fam, s, best)].net, test_start, test_end))
                    all_trades += trades_in(df, positions[(fam, s, best)], cost_per_side(s),
                                            stress[s], test_start, test_end)
                stitched[s] = pd.concat(parts)
            port = pd.concat(stitched, axis=1).fillna(0.0).mean(axis=1)
            for test_start in FOLD_TESTS:
                fold_rets.append(float((1 + window(port, test_start,
                                 test_start + pd.DateOffset(months=3))).prod() - 1))
            sr_bar, skew, kurt = per_bar_moments(port)
            dsr = deflated_sharpe_ratio(sr_bar, N_TRIALS, len(port), skew, kurt, sr_std)
            bh = pd.concat({s: window(data[s][bar]["open"].pct_change().shift(-1).fillna(0),
                                      FOLD_TESTS[0], IS_END) for s in SYMBOLS},
                           axis=1).fillna(0).mean(axis=1)
            exposure = float(np.mean([(stitched[s] != 0).mean() for s in SYMBOLS]))
            avg_rt_cost = 2 * np.mean([cost_per_side(s) for s in SYMBOLS])
            mean_trade = float(np.mean(all_trades)) if all_trades else 0.0
            is_metrics = {
                "sharpe": sharpe(port, bpy), "total_return": float((1 + port).prod() - 1),
                "max_dd": max_drawdown(port), "folds_positive": sum(r > 0 for r in fold_rets),
                "fold_returns": fold_rets, "trades": len(all_trades),
                "mean_trade": mean_trade, "avg_round_trip_cost": avg_rt_cost,
                "dsr": dsr, "exposure": exposure,
                "bh_sharpe": sharpe(bh, bpy),
                "bh_return_exposure_matched": float((1 + bh * exposure).prod() - 1),
            }
            checks = {
                "sharpe>=1.0": is_metrics["sharpe"] >= IS_PASS["sharpe"],
                "folds>=5/8": is_metrics["folds_positive"] >= IS_PASS["folds_positive"],
                "trades>=100": is_metrics["trades"] >= IS_PASS["trades"],
                "dsr>=0.95": dsr >= IS_PASS["dsr"],
                "trade>=2x_cost": mean_trade >= IS_PASS["trade_cost_mult"] * avg_rt_cost,
                "beats_bh_sharpe": is_metrics["sharpe"] > is_metrics["bh_sharpe"],
            }
            checks = {k: bool(v) for k, v in checks.items()}
            fam_out = {"is": is_metrics, "is_checks": checks, "is_pass": all(checks.values())}

            # final params: same rule, trained on the last 12 IS months
            final = {}
            for s in SYMBOLS:
                best = max(range(len(grid)), key=lambda i: sharpe(
                    window(trials[(fam, s, i)].net, datetime(2024, 1, 1, tzinfo=UTC), IS_END),
                    bpy))
                final[s] = best
            fam_out["final_params"] = {s: grid[i] for s, i in final.items()}

            # crash stress (pre-registered for every family)
            stress_out = {}
            for name, day in CRASHES.items():
                d = pd.Timestamp(day, tz=UTC)
                rets = [float((1 + window(trials[(fam, s, final[s])].net,
                                          d - pd.Timedelta(days=1), d + pd.Timedelta(days=3))
                               ).prod() - 1) for s in SYMBOLS]
                stress_out[name] = {"portfolio": float(np.mean(rets)),
                                    "worst_symbol": float(min(rets)),
                                    "flag": float(min(rets)) < -0.05}
            fam_out["crash_stress"] = stress_out

            if run_oos and fam_out["is_pass"]:
                fam_out["oos"] = _oos(data, bar, bpy, fam, grid, final, positions, stress)
            out["families"][key] = fam_out
            print(f"{key:24} IS sharpe {is_metrics['sharpe']:+.2f} "
                  f"(B&H {is_metrics['bh_sharpe']:+.2f}) folds {is_metrics['folds_positive']}/8 "
                  f"trades {is_metrics['trades']} trade {mean_trade * 1e4:+.1f}bps "
                  f"dsr {dsr:.2f} -> {'PASS' if fam_out['is_pass'] else 'fail'}", flush=True)
    return out


def _oos(data, bar, bpy, fam, grid, final, positions, stress) -> dict:
    res = {}
    for label, side_cost in (("base", lambda s: cost_per_side(s)),
                             ("cost_x2", lambda s: cost_per_side(s, mult=2.0)),
                             ("kraken_fees", lambda s: cost_per_side(s, fee_bps=KRAKEN_FEE_BPS))):
        rets = {}
        for s in SYMBOLS:
            df = data[s][bar]
            r = run(df, positions[(fam, s, final[s])], side_cost(s), stress[s])
            rets[s] = window(r.net, OOS_START, OOS_END)
        port = pd.concat(rets, axis=1).fillna(0).mean(axis=1)
        res[label] = {"sharpe": sharpe(port, bpy), "total_return": float((1 + port).prod() - 1),
                      "max_dd": max_drawdown(port)}
    bh = pd.concat({s: window(data[s][bar]["open"].pct_change().shift(-1).fillna(0),
                              OOS_START, OOS_END) for s in SYMBOLS}, axis=1).fillna(0).mean(axis=1)
    res["bh_sharpe"] = sharpe(bh, bpy)
    base = res["base"]
    res["checks"] = {
        "return>0_and_sharpe>=0.5": base["total_return"] > 0 and base["sharpe"] >= OOS_PASS["sharpe"],
        "max_dd<20%": base["max_dd"] > -OOS_PASS["max_dd"],
        "beats_bh_sharpe": base["sharpe"] > res["bh_sharpe"],
        "positive_at_cost_x2": res["cost_x2"]["total_return"] > 0,
    }
    res["checks"] = {k: bool(v) for k, v in res["checks"].items()}
    res["pass"] = all(res["checks"].values())
    return res


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 intraday study (pre-registered)")
    parser.add_argument("--no-oos", action="store_true")
    parser.add_argument("--out", default="data/research/phase2_results.json")
    args = parser.parse_args(argv)
    from src.ingestion.schemas import get_clickhouse_client

    result = study(get_clickhouse_client(), run_oos=not args.no_oos)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
