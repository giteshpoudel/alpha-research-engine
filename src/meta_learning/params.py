"""Shared tuned-parameter lookup (strategy x symbol, versioned by valid_from).

Tuning publishes a version valid from a deploy time; paper trading uses the
version that was valid at each bar, while comparison/agents use the latest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.ingestion.schemas import database_name


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def get_tuned_params(client, strategy: str, symbol: str,
                     as_of: datetime | None = None) -> dict | None:
    """Latest params valid at ``as_of`` (or the newest overall when None)."""
    if as_of is None:
        rows = client.query(
            f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = {s:String} AND symbol = {y:String} "
            "ORDER BY valid_from DESC LIMIT 1",
            parameters={"s": strategy, "y": symbol},
        ).result_rows
    else:
        rows = client.query(
            f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = {s:String} AND symbol = {y:String} "
            "AND valid_from <= {t:DateTime64(3)} ORDER BY valid_from DESC LIMIT 1",
            parameters={"s": strategy, "y": symbol, "t": _utc(as_of)},
        ).result_rows
    return json.loads(rows[0][0]) if rows else None


def get_tuned_param_history(client, strategy: str, symbol: str) -> list[tuple[datetime, dict]]:
    """All param versions for (strategy, symbol), ascending by valid_from."""
    rows = client.query(
        f"SELECT valid_from, params_json FROM {database_name()}.tuned_params FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} ORDER BY valid_from",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    return [(_utc(vf), json.loads(pj)) for vf, pj in rows]
