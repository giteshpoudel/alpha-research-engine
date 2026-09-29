"""Paper-trade the delta-neutral funding carry (Hyperliquid funding).

Replays the regime-guarded carry equity into the shared ``paper_equity`` table
under the strategy name ``funding_carry`` — one additive-equity row per funding
bar (equity = 1 + cumsum(funding - costs)). The table is a ReplacingMergeTree
keyed on (strategy, symbol, ts), so replay is idempotent: re-running rewrites the
same rows instead of duplicating them.

This is a *model* of the carry P&L (funding accrual minus per-leg costs), not a
perp-execution simulator; it exists to track realised net carry forward. Live
perp execution is out of scope and, per policy, never pushed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from src.backtesting.data import load_funding
from src.backtesting.engine import run_carry_regime_backtest
from src.backtesting.fees import fee_for_symbol, perp_fee
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe
from src.paper.executor import EquityRow
from src.paper import store

CARRY_STRATEGY = "funding_carry"
DEFAULTS = {"enter_window": 72, "exit_window": 336,
            "enter_level": 0.0, "exit_level": -5e-5}


def replay(client, symbol: str, fee: float | None = None,
           since: datetime | None = None) -> int:
    """Write the regime-guarded carry equity curve for one symbol.

    ``since`` writes only bars at/after that timestamp (idempotent top-up).
    """
    funding = load_funding(client, symbol)
    if fee is None:
        fee = fee_for_symbol(client, symbol) + perp_fee()
    result = run_carry_regime_backtest(funding, fee=fee, **DEFAULTS)
    start = funding.index[0]
    rows = [EquityRow(ts.to_pydatetime(), float(eq), float(eq), 0.0, 0.0, "carry")
            for ts, eq in result.equity_curve.items()
            if ts >= start and (since is None or ts >= since)]
    return store.insert_equity(client, CARRY_STRATEGY, symbol, rows)


def step(client, symbol: str, fee: float | None = None) -> int:
    """Append only carry bars newer than what's already stored."""
    last = store.last_paper_ts(client, CARRY_STRATEGY, symbol)
    if last is None:
        return replay(client, symbol, fee=fee)
    return replay(client, symbol, fee=fee, since=last + timedelta(hours=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper-trade funding carry")
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--step", action="store_true",
                        help="only append bars newer than stored (default: full replay)")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    for symbol in symbols:
        try:
            n = (step if args.step else replay)(ch, symbol, fee=args.fee)
            print(f"funding_carry {symbol}: {n} bars")
        except Exception as exc:
            print(f"funding_carry {symbol}: FAILED: {exc}")


if __name__ == "__main__":
    main()
