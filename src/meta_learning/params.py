"""Shared tuned-parameter lookup (strategy x symbol).

Both the OOS comparison and paper trading consume tuned params, so the read
lives here rather than in either caller.
"""

from __future__ import annotations

import json

from src.ingestion.schemas import database_name


def get_tuned_params(client, strategy: str, symbol: str) -> dict | None:
    """Latest tuned params for (strategy, symbol), or None when never tuned."""
    rows = client.query(
        f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} LIMIT 1",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    return json.loads(rows[0][0]) if rows else None
