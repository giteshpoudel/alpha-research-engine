"""Delta-neutral funding carry: long spot + short perp, harvest perp funding.

Gross carry is the funding rate accrued while positioned; net subtracts
round-trip execution on every flip. ``run_funding_backtest`` charges ``2*fee``
per flip (both legs), so pass the per-leg cost = spot taker + perp taker.
A positive ``threshold`` only holds the position when funding is large enough
to be worth it, reducing flip churn during funding sign changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.backtesting.data import load_funding
from src.backtesting.engine import run_funding_backtest
from src.backtesting.fees import fee_for_symbol, perp_fee
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe

_SECONDS_PER_YEAR = 365 * 24 * 3600


def carry_metrics(funding: pd.Series, threshold: float, fee: float,
                  exit_threshold: float | None = None) -> dict:
    """Net carry stats for one funding series at a given threshold/fee."""
    res = run_funding_backtest(funding, threshold=threshold, fee=fee,
                               exit_threshold=exit_threshold)
    years = ((funding.index[-1] - funding.index[0]).total_seconds()
             / _SECONDS_PER_YEAR) if len(funding) > 1 else 0.0
    if years > 0 and res.total_return > -1.0:
        annualized = (1.0 + res.total_return) ** (1.0 / years) - 1.0
    else:
        annualized = -1.0
    return {"total_return": res.total_return, "annualized": annualized,
            "sharpe": res.sharpe, "max_drawdown": res.max_drawdown,
            "trades": res.num_trades, "years": years}


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
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
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
