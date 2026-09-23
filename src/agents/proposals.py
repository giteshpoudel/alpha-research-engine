"""Strategy proposals: record an agent's idea, evaluate it, and decide.

Auto-evaluation covers ``params`` proposals: the proposed params must beat the
incumbent on the SAME validation folds (the champion/challenger gate) before
being published. Other kinds are marked for manual review. Every proposal and
its evidence are persisted for audit.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.ingestion.schemas import database_name
from src.meta_learning.params import get_tuned_params
from src.meta_learning.tuner import (
    adopt_candidate,
    publish,
    validate_params,
    walk_forward_folds,
)

_COLUMNS = ("proposal_id", "run_id", "kind", "symbol", "hypothesis",
            "proposed_change", "baseline", "evidence", "decision", "reason",
            "model", "created_at", "decided_at")


class ParamsChange(BaseModel):
    strategy: str = "mean_reversion"
    symbol: str
    params: dict


class Proposal(BaseModel):
    hypothesis: str
    kind: str = "params"
    symbol: str = ""
    model: str = ""
    run_id: str = ""
    proposed_change: dict = Field(default_factory=dict)


def _write(ch, *, proposal_id, run_id, kind, symbol, hypothesis, proposed_change,
           baseline, evidence, decision, reason, model, created_at, decided_at) -> None:
    ch.insert(
        f"{database_name()}.strategy_proposals",
        [[proposal_id, run_id, kind, symbol, hypothesis,
          json.dumps(proposed_change, default=str),
          json.dumps(baseline, default=str) if baseline is not None else "",
          json.dumps(evidence, default=str) if evidence is not None else "",
          decision, reason, model, created_at, decided_at]],
        column_names=list(_COLUMNS),
    )


def create_proposal(ch, hypothesis: str, proposed_change: dict, kind: str = "params",
                    symbol: str = "", model: str = "", run_id: str = "") -> str:
    proposal_id = uuid.uuid4().hex[:16]
    _write(ch, proposal_id=proposal_id, run_id=run_id, kind=kind, symbol=symbol,
           hypothesis=hypothesis, proposed_change=proposed_change, baseline=None,
           evidence=None, decision="pending", reason="", model=model,
           created_at=datetime.now(timezone.utc), decided_at=None)
    return proposal_id


def get_proposal(ch, proposal_id: str) -> dict | None:
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.strategy_proposals FINAL "
        "WHERE proposal_id = {p:String}",
        parameters={"p": proposal_id},
    ).result_rows
    return dict(zip(_COLUMNS, rows[0])) if rows else None


def list_proposals(ch, decision: str | None = None, limit: int = 50) -> list[dict]:
    where, params = "", {"n": limit}
    if decision:
        where, params["d"] = "WHERE decision = {d:String}", decision
    rows = ch.query(
        f"SELECT {', '.join(_COLUMNS)} FROM {database_name()}.strategy_proposals FINAL "
        f"{where} ORDER BY created_at DESC LIMIT {{n:UInt32}}",
        parameters=params,
    ).result_rows
    return [dict(zip(_COLUMNS, row)) for row in rows]


def _row_objects(row: dict) -> dict:
    return {
        "proposal_id": row["proposal_id"], "run_id": row["run_id"],
        "kind": row["kind"], "symbol": row["symbol"], "hypothesis": row["hypothesis"],
        "proposed_change": json.loads(row["proposed_change"]) if row["proposed_change"] else {},
        "baseline": json.loads(row["baseline"]) if row["baseline"] else None,
        "evidence": json.loads(row["evidence"]) if row["evidence"] else None,
        "decision": row["decision"], "reason": row["reason"], "model": row["model"],
        "created_at": row["created_at"], "decided_at": row["decided_at"],
    }


def evaluate_proposal(ch, proposal_id: str, publish_adopted: bool = True,
                      run_id: str = "") -> dict:
    """Champion/challenger evaluation; updates and returns the proposal row."""
    row = get_proposal(ch, proposal_id)
    if row is None:
        raise ValueError(f"unknown proposal {proposal_id}")
    if row["kind"] != "params":
        _write(ch, **{**_row_objects(row), "decision": "manual",
                      "reason": "auto-evaluation only for params",
                      "decided_at": datetime.now(timezone.utc)})
        return get_proposal(ch, proposal_id)

    change = ParamsChange(**json.loads(row["proposed_change"]))
    folds = walk_forward_folds()
    candidate_val = validate_params(ch, change.strategy, change.symbol, change.params, folds)
    incumbent = get_tuned_params(ch, change.strategy, change.symbol)
    incumbent_val = (validate_params(ch, change.strategy, change.symbol, incumbent, folds)
                     if incumbent else None)
    adopted = adopt_candidate(candidate_val, incumbent_val)
    if adopted and publish_adopted and candidate_val is not None:
        publish(ch, change.strategy, change.symbol,
                {"params": change.params, "train_sharpe": 0.0,
                 "validation_sharpe": candidate_val, "folds": len(folds)},
                datetime.now(timezone.utc))

    _write(ch, **{**_row_objects(row), "run_id": run_id or row["run_id"],
                  "baseline": incumbent,
                  "evidence": {"candidate_val": candidate_val,
                               "incumbent_val": incumbent_val, "folds": len(folds)},
                  "decision": "adopted" if adopted else "rejected",
                  "reason": f"candidate {candidate_val} vs incumbent {incumbent_val}",
                  "decided_at": datetime.now(timezone.utc)})
    return get_proposal(ch, proposal_id)
