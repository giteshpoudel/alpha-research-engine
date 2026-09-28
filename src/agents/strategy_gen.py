"""Register agent-generated strategies after sandbox validation.

A strategy is only written to ``strategies/generated/<name>.py`` and added to
the live registry if it passes the AST allowlist and the subprocess smoke test.
Generated strategies are paper-only by policy.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from src.agents.change_log import log as log_change
from src.agents.strategy_sandbox import smoke_test, validate_source
from src.backtesting.strategies import registry
from src.ingestion.schemas import database_name

_COLUMNS = ("name", "code", "description", "status", "validation_sharpe",
            "model", "created_at", "updated_at")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def register_generated_strategy(ch, name: str, code: str, description: str = "",
                                model: str = "") -> dict:
    name = (name or "").strip().lower()
    if not _NAME_RE.match(name):
        return {"registered": False, "errors": ["invalid name: use [a-z0-9_], 3-41 chars"]}
    errors = validate_source(code)
    if errors:
        return {"registered": False, "errors": errors}
    smoke = smoke_test(code)
    if not smoke.get("ok"):
        return {"registered": False, "errors": [smoke.get("error") or "smoke test failed"]}

    registry.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (registry.GENERATED_DIR / f"{name}.py").write_text(code)
    now = datetime.now(timezone.utc)
    ch.insert(
        f"{database_name()}.generated_strategies",
        [[name, code, description, "active", 0.0, model, now, now]],
        column_names=list(_COLUMNS),
    )
    registry.load_generated()
    log_change(ch, "strategy_registered", subject=name,
               detail={"description": description, "model": model})
    return {"registered": True, "name": name}


def get_generated_strategy(ch, name: str) -> dict | None:
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.generated_strategies FINAL "
        "WHERE name = {n:String}",
        parameters={"n": name},
    ).result_rows
    return dict(zip(_COLUMNS, rows[0])) if rows else None


def list_generated_strategies(ch) -> list[dict]:
    rows = ch.query(
        f"SELECT name, description, status, validation_sharpe, model, created_at "
        f"FROM {database_name()}.generated_strategies FINAL ORDER BY created_at DESC"
    ).result_rows
    cols = ("name", "description", "status", "validation_sharpe", "model", "created_at")
    return [dict(zip(cols, row)) for row in rows]
