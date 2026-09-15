"""ClickHouse persistence for paper trading.

Every write is deterministic (content-addressed keys, ReplacingMergeTree), so
re-running replay or a forward step collapses to the same rows instead of
duplicating them.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from src.ingestion.schemas import database_name
from src.paper.executor import EquityRow, SleeveState, Trade

_EQUITY_COLUMNS = (
    "strategy", "symbol", "ts", "equity", "cash", "position_value",
    "mark_price", "status", "created_at",
)
_TRADE_COLUMNS = (
    "trade_id", "strategy", "symbol", "side", "ts", "price", "size",
    "notional", "fee", "realized_pnl", "reason", "created_at",
)
_POSITION_COLUMNS = (
    "strategy", "symbol", "status", "entry_price", "entry_ts", "size",
    "cash", "cost_basis", "last_bar_ts", "updated_at",
)


def make_trade_id(strategy: str, symbol: str, ts: datetime, side: str) -> str:
    payload = f"{strategy}|{symbol}|{ts.isoformat()}|{side}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def load_state(client, strategy: str, symbol: str) -> SleeveState:
    """Current sleeve snapshot, or a fresh $1 flat sleeve when none exists."""
    rows = client.query(
        f"SELECT status, entry_price, entry_ts, size, cash, cost_basis "
        f"FROM {database_name()}.paper_positions FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} LIMIT 1",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    if not rows:
        return SleeveState()
    status, entry_price, entry_ts, size, cash, cost_basis = rows[0]
    return SleeveState(status, float(entry_price), _utc(entry_ts), float(size),
                       float(cash), float(cost_basis))


def upsert_position(client, strategy: str, symbol: str, state: SleeveState,
                    last_bar_ts: datetime) -> None:
    now = datetime.now(timezone.utc)
    client.insert(
        f"{database_name()}.paper_positions",
        [[strategy, symbol, state.status, state.entry_price, _utc(state.entry_ts),
          state.size, state.cash, state.cost_basis, last_bar_ts, now]],
        column_names=list(_POSITION_COLUMNS),
    )


def insert_equity(client, strategy: str, symbol: str, rows: list[EquityRow]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [[strategy, symbol, r.ts, r.equity, r.cash, r.position_value,
             r.mark_price, r.status, now] for r in rows]
    client.insert(f"{database_name()}.paper_equity", data,
                  column_names=list(_EQUITY_COLUMNS))
    return len(rows)


def insert_trades(client, strategy: str, symbol: str, trades: list[Trade]) -> int:
    if not trades:
        return 0
    now = datetime.now(timezone.utc)
    data = [[make_trade_id(strategy, symbol, t.ts, t.side), strategy, symbol,
             t.side, t.ts, t.price, t.size, t.notional, t.fee, t.realized_pnl,
             t.reason, now] for t in trades]
    client.insert(f"{database_name()}.paper_trades", data,
                  column_names=list(_TRADE_COLUMNS))
    return len(trades)


def last_paper_ts(client, strategy: str, symbol: str) -> datetime | None:
    """Latest processed bar for a sleeve, or None when it has no history."""
    rows = client.query(
        f"SELECT maxOrNull(ts) FROM {database_name()}.paper_equity FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String}",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    return _utc(rows[0][0]) if rows else None


def latest_closed_bar_ts(client, symbol: str, interval: str = "1h") -> datetime | None:
    """Latest closed OHLCV bar for a symbol (any exchange), or None."""
    rows = client.query(
        f"SELECT maxOrNull(ts) FROM {database_name()}.ohlcv FINAL "
        "WHERE symbol = {y:String} AND interval = {i:String}",
        parameters={"y": symbol, "i": interval},
    ).result_rows
    return _utc(rows[0][0]) if rows else None
