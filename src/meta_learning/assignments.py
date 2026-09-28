"""Per-symbol strategy assignment: which strategy trades each symbol.

Seeded to ``mean_reversion`` for every universe symbol. The optimizer can switch
a symbol's strategy via a champion/challenger-evaluated proposal; paper trading
and backtests then dispatch per symbol.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.backtesting.strategies.registry import STRATEGY_NAMES
from src.ingestion.schemas import database_name
from src.ingestion.universe import universe

DEFAULT_STRATEGY = "mean_reversion"
_COLUMNS = ("symbol", "strategy", "valid_from", "updated_at")


def _write(ch, symbol: str, strategy: str) -> None:
    now = datetime.now(timezone.utc)
    ch.insert(
        f"{database_name()}.strategy_assignments",
        [[symbol, strategy, now, now]],
        column_names=list(_COLUMNS),
    )


def ensure_assignments(ch) -> int:
    count = ch.query(
        f"SELECT count() FROM {database_name()}.strategy_assignments"
    ).result_rows[0][0]
    if count:
        return count
    symbols = universe(ch)
    for symbol in symbols:
        _write(ch, symbol, DEFAULT_STRATEGY)
    return len(symbols)


def assigned_strategy(ch, symbol: str) -> str:
    ensure_assignments(ch)
    rows = ch.query(
        f"SELECT strategy FROM {database_name()}.strategy_assignments FINAL "
        "WHERE symbol = {s:String}",
        parameters={"s": symbol},
    ).result_rows
    return rows[0][0] if rows else DEFAULT_STRATEGY


def set_assignment(ch, symbol: str, strategy: str) -> None:
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"unknown strategy {strategy!r}; expected {STRATEGY_NAMES}")
    _write(ch, symbol, strategy)


def list_assignments(ch) -> dict[str, str]:
    ensure_assignments(ch)
    rows = ch.query(
        f"SELECT symbol, strategy FROM {database_name()}.strategy_assignments FINAL "
        "ORDER BY symbol"
    ).result_rows
    return {symbol: strategy for symbol, strategy in rows}
