"""Requests from the optimizer to the human.

Backtest/paper experimentation is autonomous, but adding data/APIs or making
major changes needs a human: the agent files a request with justification and
expected profit impact. Requests are reviewed via the CLI or dashboard.

Usage:
    python -m src.agents.requests --list
    python -m src.agents.requests --approve <id> [--note "..."]
    python -m src.agents.requests --reject <id> [--note "..."]
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

from src.agents.change_log import log as log_change
from src.ingestion.schemas import database_name, get_clickhouse_client

_COLUMNS = ("request_id", "run_id", "kind", "title", "justification",
            "expected_impact", "status", "model", "created_at", "decided_at",
            "decision_note")
DECISIONS = ("approved", "rejected")


def _write(ch, *, request_id, run_id, kind, title, justification, expected_impact,
           status, model, created_at, decided_at, decision_note) -> None:
    ch.insert(
        f"{database_name()}.agent_requests",
        [[request_id, run_id, kind, title, justification, expected_impact,
          status, model, created_at, decided_at, decision_note]],
        column_names=list(_COLUMNS),
    )


def create_request(ch, title: str, justification: str, kind: str = "data",
                   expected_impact: str = "", model: str = "", run_id: str = "") -> str:
    request_id = uuid.uuid4().hex[:16]
    _write(ch, request_id=request_id, run_id=run_id, kind=kind, title=title,
           justification=justification, expected_impact=expected_impact,
           status="open", model=model, created_at=datetime.now(timezone.utc),
           decided_at=None, decision_note="")
    log_change(ch, "request_created", subject=kind,
               detail={"title": title, "expected_impact": expected_impact}, run_id=run_id)
    return request_id


def get_request(ch, request_id: str) -> dict | None:
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.agent_requests FINAL "
        "WHERE request_id = {r:String}",
        parameters={"r": request_id},
    ).result_rows
    return dict(zip(_COLUMNS, rows[0])) if rows else None


def list_requests(ch, status: str | None = None, limit: int = 50) -> list[dict]:
    where, params = "", {"n": limit}
    if status:
        where, params["s"] = "WHERE status = {s:String}", status
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.agent_requests FINAL "
        f"{where} ORDER BY created_at DESC LIMIT {{n:UInt32}}",
        parameters=params,
    ).result_rows
    return [dict(zip(_COLUMNS, row)) for row in rows]


def decide_request(ch, request_id: str, status: str, note: str = "") -> dict:
    if status not in DECISIONS:
        raise ValueError(f"status must be one of {DECISIONS}")
    row = get_request(ch, request_id)
    if row is None:
        raise ValueError(f"unknown request {request_id}")
    _write(ch, **{**{k: row[k] for k in _COLUMNS},
                  "status": status, "decision_note": note,
                  "decided_at": datetime.now(timezone.utc)})
    return get_request(ch, request_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review optimizer requests")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--approve", metavar="ID")
    group.add_argument("--reject", metavar="ID")
    parser.add_argument("--status", default=None, choices=["open", "approved", "rejected"])
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    if args.list:
        rows = list_requests(ch, status=args.status)
        if not rows:
            print("no requests")
        for r in rows:
            print(f"[{r['status']}] {r['request_id']} ({r['kind']}) {r['title']}\n"
                  f"    why: {r['justification']}\n"
                  f"    impact: {r['expected_impact']}")
        return
    request_id = args.approve or args.reject
    status = "approved" if args.approve else "rejected"
    row = decide_request(ch, request_id, status, note=args.note)
    print(f"{request_id}: {row['status']}")


if __name__ == "__main__":
    main()
