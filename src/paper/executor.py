"""Paper-trading executor: a pure, deterministic per-sleeve state machine.

No I/O and no clock: given one closed bar and its signals, ``step`` returns the
next sleeve state, an optional trade, and the resulting equity row. Keeping
this pure makes the forward simulation and replay trivially testable and
idempotent (the same inputs always yield the same outputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SleeveState:
    """Snapshot of one independent $1 sleeve."""

    status: str = "flat"  # "flat" | "long"
    entry_price: float = 0.0
    entry_ts: datetime | None = None
    size: float = 0.0
    cash: float = 1.0
    cost_basis: float = 0.0  # cash deployed at entry, for realized-PnL math


@dataclass
class Trade:
    side: str  # "entry" | "exit"
    ts: datetime
    price: float
    size: float
    notional: float
    fee: float
    realized_pnl: float
    reason: str


@dataclass
class EquityRow:
    ts: datetime
    equity: float
    cash: float
    position_value: float
    mark_price: float
    status: str


def step(state: SleeveState, bar_ts: datetime, close: float,
         entry_sig: bool, exit_sig: bool, fee: float) -> tuple[SleeveState, Trade | None, EquityRow]:
    """Advance one sleeve over one closed bar.

    Long-only, all-in: entry deploys the entire cash balance; exit liquidates
    the entire position. A signal that contradicts the current state is
    ignored (matching VectorBT ``from_signals`` semantics). Fills are assumed
    at this bar's close.
    """
    trade: Trade | None = None

    if state.status == "flat":
        if entry_sig:
            cost = state.cash
            fee_amt = cost * fee
            size = (cost - fee_amt) / close
            state = SleeveState("long", close, bar_ts, size, 0.0, cost)
            trade = Trade("entry", bar_ts, close, size, cost, fee_amt, 0.0, "entry")
    elif exit_sig:
        gross = state.size * close
        fee_amt = gross * fee
        proceeds = gross - fee_amt
        trade = Trade("exit", bar_ts, close, state.size, gross, fee_amt,
                      proceeds - state.cost_basis, "exit")
        state = SleeveState("flat", 0.0, None, 0.0, proceeds, 0.0)

    position_value = state.size * close if state.status == "long" else 0.0
    equity = state.cash + position_value
    return state, trade, EquityRow(bar_ts, equity, state.cash, position_value, close, state.status)
