"""Delta-neutral funding carry: long spot + short perp, harvest perp funding.

Gross carry is the funding rate accrued while positioned; net subtracts
round-trip execution on every flip. ``run_funding_backtest`` charges ``2*fee``
per flip (both legs), so pass the per-leg cost = spot taker + perp taker.

Two positioning rules:

- ``carry_metrics``: fixed threshold with optional hysteresis (enter above
  ``threshold``, exit below ``exit_threshold``).
- ``carry_regime_metrics``: trailing-mean regime guard — enter when the recent
  (e.g. 24h) mean funding is positive, exit when the slower (e.g. 7d) mean turns
  negative. Holds the carry through noise, steps aside in bad regimes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.backtesting.data import load_funding
from src.backtesting.engine import run_carry_regime_backtest, run_funding_backtest
from src.backtesting.fees import fee_for_symbol, perp_fee
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe

_SECONDS_PER_YEAR = 365 * 24 * 3600


def _summarize(result, funding: pd.Series) -> dict:
    years = ((funding.index[-1] - funding.index[0]).total_seconds()
             / _SECONDS_PER_YEAR) if len(funding) > 1 else 0.0
    if years > 0 and result.total_return > -1.0:
        annualized = (1.0 + result.total_return) ** (1.0 / years) - 1.0
    else:
        annualized = -1.0
    return {"total_return": result.total_return, "annualized": annualized,
            "sharpe": result.sharpe, "max_drawdown": result.max_drawdown,
            "trades": result.num_trades, "years": years}


def carry_metrics(funding: pd.Series, threshold: float, fee: float,
                  exit_threshold: float | None = None) -> dict:
    """Net carry stats for a fixed-threshold (optionally hysteretic) rule."""
    return _summarize(run_funding_backtest(funding, threshold=threshold, fee=fee,
                                           exit_threshold=exit_threshold), funding)


def carry_regime_metrics(funding: pd.Series, fee: float,
                         enter_window: int = 72, exit_window: int = 336,
                         enter_level: float = 0.0, exit_level: float = -5e-5) -> dict:
    """Net carry stats under the trailing-mean regime guard."""
    return _summarize(run_carry_regime_backtest(
        funding, enter_window=enter_window, exit_window=exit_window,
        enter_level=enter_level, exit_level=exit_level, fee=fee), funding)


def evaluate(ch_client, symbol: str, threshold: float = 0.0,
             fee: float | None = None, start: datetime | None = None,
             end: datetime | None = None) -> dict:
    """Load funding for a symbol and return net carry metrics."""
    if fee is None:
        fee = fee_for_symbol(ch_client, symbol) + perp_fee()
    funding = load_funding(ch_client, symbol, start=start, end=end)
    return carry_metrics(funding, threshold, fee)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Delta-neutral funding carry evaluation")
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--thresholds", default="0,5e-6,1e-5,2e-5,5e-5")
    parser.add_argument("--fee", type=float, default=None,
                        help="per-leg fee; default spot taker + perp taker")
    parser.add_argument("--regime", action="store_true",
                        help="show unguarded vs regime-guarded carry")
    parser.add_argument("--enter-window", type=int, default=72)
    parser.add_argument("--exit-window", type=int, default=336)
    parser.add_argument("--enter-level", type=float, default=0.0)
    parser.add_argument("--exit-level", type=float, default=-5e-5)
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    if args.regime:
        print(f"{'symbol':6}{'fee_bps':>8}{'hold':>10}{'regime':>10}{'maxDD':>10}{'flips':>7}")
        for symbol in symbols:
            fee = args.fee if args.fee is not None else fee_for_symbol(ch, symbol) + perp_fee()
            try:
                funding = load_funding(ch, symbol)
                hold = carry_metrics(funding, -1e-9, fee, -1.0)
                reg = carry_regime_metrics(funding, fee, args.enter_window,
                                           args.exit_window,
                                           enter_level=args.enter_level,
                                           exit_level=args.exit_level)
                print(f"{symbol:6}{fee * 1e4:>8.1f}{hold['annualized'] * 100:>9.2f}%"
                      f"{reg['annualized'] * 100:>9.2f}%{reg['max_drawdown'] * 100:>9.1f}%"
                      f"{reg['trades']:>7d}")
            except Exception as exc:
                print(f"{symbol:6} FAILED: {exc}")
        return

    thresholds = [float(t) for t in args.thresholds.split(",")]
    header = "".join(f"{('thr=' + format(t, '.0e')):>13}" for t in thresholds)
    print(f"{'symbol':6}{'fee_bps':>8}{header}")
    for symbol in symbols:
        fee = args.fee if args.fee is not None else fee_for_symbol(ch, symbol) + perp_fee()
        try:
            funding = load_funding(ch, symbol)
            cells = "".join(
                f"{carry_metrics(funding, t, fee)['annualized'] * 100:>12.2f}%"
                for t in thresholds)
            print(f"{symbol:6}{fee * 1e4:>8.1f}{cells}")
        except Exception as exc:
            print(f"{symbol:6} FAILED: {exc}")


if __name__ == "__main__":
    main()
