"""Fee model: per-exchange maker/taker fees, env-overridable.

Real retail base-tier fees:

- Binance.US: maker 0.00%, taker 0.02%
- Coinbase:  maker 0.40%, taker 0.60%

A symbol's fee is resolved from the exchange its market data comes from.
``FEE_MODE`` selects maker/taker (default ``taker`` — signal strategies
enter at market). Any rate can be overridden with
``FEE_<EXCHANGE>_<MODE>`` (e.g. ``FEE_COINBASE_TAKER=0.005``).
"""

from __future__ import annotations

import os

EXCHANGE_FEES = {
    "binance_us": {"maker": 0.0000, "taker": 0.0002},
    "coinbase": {"maker": 0.0040, "taker": 0.0060},
}
_FALLBACK = {"maker": 0.0000, "taker": 0.0010}


def execution_mode() -> str:
    """``taker`` (market fills) or ``maker`` (passive limit fills)."""
    return os.environ.get("EXECUTION_MODE") or os.environ.get("FEE_MODE", "taker")


def fee_mode() -> str:
    # Backwards-compatible alias for the fee side of the execution mode.
    return execution_mode()


def maker_limit_bps() -> float:
    return float(os.environ.get("MAKER_LIMIT_BPS", "5"))


def maker_max_wait() -> int:
    return int(os.environ.get("MAKER_MAX_WAIT", "1"))


def fee_for_exchange(exchange: str, mode: str | None = None) -> float:
    mode = mode or execution_mode()
    env = os.environ.get(f"FEE_{exchange.upper()}_{mode.upper()}")
    if env is not None:
        return float(env)
    return EXCHANGE_FEES.get(exchange, _FALLBACK).get(mode, _FALLBACK[mode])


def fee_for_interval(interval: str) -> float:
    env = os.environ.get(f"FEE_{interval.upper()}")
    if env is not None:
        return float(env)
    return float(os.environ.get("DEFAULT_FEE", "0.001"))


def fee_for_symbol(ch_client, symbol: str, mode: str | None = None) -> float:
    from src.backtesting.data import exchange_for_symbol

    return fee_for_exchange(exchange_for_symbol(ch_client, symbol), mode)
