from datetime import datetime, timedelta, timezone

import pytest

from src.paper.executor import SleeveState, step

TS = datetime(2025, 1, 1, tzinfo=timezone.utc)
FEE = 0.001


def test_entry_math():
    state, trade, eq = step(SleeveState(cash=1.0), TS, 100.0, True, False, FEE)
    assert trade.side == "entry"
    assert state.status == "long"
    assert state.size == pytest.approx((1.0 * (1 - FEE)) / 100.0)
    assert state.cash == 0.0
    assert state.cost_basis == 1.0
    assert eq.equity == pytest.approx(1.0 * (1 - FEE))
    assert eq.position_value == pytest.approx(1.0 * (1 - FEE))
    assert eq.status == "long"


def test_exit_math_net_of_both_fees():
    state, _, _ = step(SleeveState(cash=1.0), TS, 100.0, True, False, FEE)
    state2, trade, eq = step(state, TS + timedelta(hours=1), 120.0, False, True, FEE)
    proceeds = state.size * 120.0 * (1 - FEE)
    assert trade.side == "exit"
    assert state2.status == "flat"
    assert state2.cash == pytest.approx(proceeds)
    assert trade.realized_pnl == pytest.approx(proceeds - 1.0)
    assert eq.equity == pytest.approx(proceeds)


def test_flat_without_signal_is_noop():
    state, trade, eq = step(SleeveState(), TS, 100.0, False, False, FEE)
    assert trade is None
    assert state.status == "flat" and state.cash == 1.0
    assert eq.equity == 1.0


def test_contradictory_signals_are_ignored():
    long_state = SleeveState("long", 100.0, TS, 0.01, 0.0, 1.0)
    state, trade, eq = step(long_state, TS, 90.0, True, False, FEE)
    assert trade is None and state.status == "long"
    assert eq.equity == pytest.approx(state.size * 90.0)

    flat_state = SleeveState()
    state2, trade2, eq2 = step(flat_state, TS, 90.0, False, True, FEE)
    assert trade2 is None and state2.status == "flat" and eq2.equity == 1.0


def test_disabled_blocks_entries_and_force_closes():
    flat = step(SleeveState(), TS, 100.0, True, False, FEE, enabled=False)
    state, trade, eq = flat
    assert trade is None and state.status == "flat" and eq.equity == 1.0

    long_state = SleeveState("long", 100.0, TS, 0.01, 0.0, 1.0)
    state2, trade2, _ = step(long_state, TS, 110.0, False, False, FEE, enabled=False)
    assert trade2.side == "exit" and trade2.reason == "halted"
    assert state2.status == "flat"
