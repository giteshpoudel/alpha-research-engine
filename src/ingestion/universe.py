"""Dynamic trading universe: membership plus add/remove with data backfill.

Seeded from ``TICKER_ALIASES`` (the sentiment alias map) on first use, then
owned by the ``universe`` table so the optimizer can add/remove symbols without
code changes. Adding a symbol backfills its OHLCV (Binance.US) and funding
(Hyperliquid) history.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES

_COLUMNS = ("symbol", "enabled", "aliases", "added_at", "notes")
_OHLCV_START = datetime(2017, 1, 1, tzinfo=timezone.utc)
_FUNDING_START = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _write(ch, symbol: str, enabled: bool, aliases: list[str], notes: str) -> None:
    ch.insert(
        f"{database_name()}.universe",
        [[symbol, 1 if enabled else 0, list(aliases), datetime.now(timezone.utc), notes]],
        column_names=list(_COLUMNS),
    )


def ensure_universe(ch) -> int:
    """Seed the universe from the static alias map if the table is empty."""
    count = ch.query(f"SELECT count() FROM {database_name()}.universe").result_rows[0][0]
    if count:
        return count
    for symbol, aliases in TICKER_ALIASES.items():
        _write(ch, symbol, True, list(aliases), "seed")
    return len(TICKER_ALIASES)


def universe(ch, enabled_only: bool = True) -> tuple[str, ...]:
    """The tradable symbol universe (falls back to the alias map if empty)."""
    ensure_universe(ch)
    where = "WHERE enabled = 1" if enabled_only else ""
    rows = ch.query(
        f"SELECT symbol FROM {database_name()}.universe FINAL {where} ORDER BY symbol"
    ).result_rows
    symbols = tuple(r[0] for r in rows)
    return symbols or tuple(TICKER_ALIASES)


def get_symbol(ch, symbol: str) -> dict | None:
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.universe FINAL "
        "WHERE symbol = {s:String}",
        parameters={"s": symbol.upper()},
    ).result_rows
    return dict(zip(_COLUMNS, rows[0])) if rows else None


def list_universe(ch) -> list[dict]:
    ensure_universe(ch)
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.universe FINAL ORDER BY symbol"
    ).result_rows
    return [dict(zip(_COLUMNS, row)) for row in rows]


def add_symbol(ch, symbol: str, aliases: list[str] | None = None,
               backfill: bool = True, notes: str = "agent") -> dict:
    """Register a symbol (backfilling data) and enable it in the universe."""
    symbol = symbol.upper()
    aliases = aliases or [symbol.lower()]
    stats: dict = {"symbol": symbol, "ohlcv": 0, "funding": 0}
    enabled = True
    if backfill:
        from src.ingestion.binance_us import backfill_ohlcv
        from src.ingestion.hyperliquid import backfill_funding
        from src.ingestion.market_data import max_ts
        try:
            stats["ohlcv"] = backfill_ohlcv(ch, symbols=(symbol,),
                                            intervals=("1h", "1d"), start=_OHLCV_START)
        except Exception as exc:
            stats["ohlcv_error"] = str(exc)
        try:
            stats["funding"] = backfill_funding(ch, coins=(symbol,), start=_FUNDING_START)
        except Exception as exc:
            stats["funding_error"] = str(exc)
        if max_ts(ch, "ohlcv", "binance_us", symbol, "1h") is None:
            enabled = False  # not tradable on the US provider we backfill from
            stats["note"] = "no ohlcv found; registered disabled"
    stats["enabled"] = enabled
    _write(ch, symbol, enabled, aliases, notes)
    return stats


def remove_symbol(ch, symbol: str, notes: str = "agent") -> dict:
    symbol = symbol.upper()
    existing = get_symbol(ch, symbol)
    aliases = list(existing["aliases"]) if existing else [symbol.lower()]
    _write(ch, symbol, False, aliases, notes)
    return {"symbol": symbol, "enabled": False}
