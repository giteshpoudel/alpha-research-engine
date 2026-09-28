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

from src.agents.change_log import log as log_change
from src.backtesting.strategies import registry
from src.ingestion.schemas import database_name
from src.meta_learning.assignments import assigned_strategy, set_assignment
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


def _record(params: dict, validation: float | None, folds: int) -> dict:
    return {"params": params, "train_sharpe": 0.0,
            "validation_sharpe": validation, "folds": folds}


def evaluate_proposal(ch, proposal_id: str, publish_adopted: bool = True,
                      run_id: str = "") -> dict:
    """Champion/challenger evaluation for params and strategy proposals."""
    row = get_proposal(ch, proposal_id)
    if row is None:
        raise ValueError(f"unknown proposal {proposal_id}")
    change = json.loads(row["proposed_change"])
    folds = walk_forward_folds()
    baseline = None

    if row["kind"] == "params":
        strategy, symbol, params = change["strategy"], change["symbol"], change["params"]
        candidate_val = validate_params(ch, strategy, symbol, params, folds)
        baseline = get_tuned_params(ch, strategy, symbol)
        incumbent_val = (validate_params(ch, strategy, symbol, baseline, folds)
                         if baseline else None)
        adopted = candidate_val is not None and adopt_candidate(candidate_val, incumbent_val)
        if adopted and publish_adopted and candidate_val is not None:
            publish(ch, strategy, symbol, _record(params, candidate_val, len(folds)),
                    datetime.now(timezone.utc))
        evidence = {"candidate_val": candidate_val, "incumbent_val": incumbent_val,
                    "folds": len(folds)}
        reason = f"candidate {candidate_val} vs incumbent {incumbent_val}"

    elif row["kind"] == "strategy":
        candidate, symbol = change["strategy"], change["symbol"]
        candidate_params = change.get("params") or registry.defaults(candidate)
        candidate_val = validate_params(ch, candidate, symbol, candidate_params, folds)
        incumbent_strategy = assigned_strategy(ch, symbol)
        incumbent_params = (get_tuned_params(ch, incumbent_strategy, symbol)
                            or registry.defaults(incumbent_strategy))
        incumbent_val = validate_params(ch, incumbent_strategy, symbol, incumbent_params, folds)
        baseline = {"strategy": incumbent_strategy, "params": incumbent_params}
        adopted = candidate == incumbent_strategy or (
            candidate_val is not None and adopt_candidate(candidate_val, incumbent_val))
        if adopted and publish_adopted and candidate_val is not None:
            publish(ch, candidate, symbol, _record(candidate_params, candidate_val, len(folds)),
                    datetime.now(timezone.utc))
            set_assignment(ch, symbol, candidate)
        evidence = {"candidate_strategy": candidate, "candidate_val": candidate_val,
                    "incumbent_strategy": incumbent_strategy, "incumbent_val": incumbent_val,
                    "folds": len(folds)}
        reason = f"{candidate} {candidate_val} vs {incumbent_strategy} {incumbent_val}"

    else:
        _write(ch, **{**_row_objects(row), "decision": "manual",
                      "reason": "auto-evaluation only for params/strategy",
                      "decided_at": datetime.now(timezone.utc)})
        return get_proposal(ch, proposal_id)

    _write(ch, **{**_row_objects(row), "run_id": run_id or row["run_id"],
                  "baseline": baseline, "evidence": evidence,
                  "decision": "adopted" if adopted else "rejected",
                  "reason": reason, "decided_at": datetime.now(timezone.utc)})
    if adopted:
        subject = row["symbol"] or change.get("symbol", "")
        event = "strategy_switch" if row["kind"] == "strategy" else "params_adopted"
        log_change(ch, event, subject=subject,
                   detail={"kind": row["kind"], "change": change,
                           "evidence": evidence, "reason": reason},
                   run_id=run_id or row["run_id"])
    return get_proposal(ch, proposal_id)
