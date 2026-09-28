"""Per-run quality scoring and regression detection (the eval loop).

Each optimizer run is scored from its outcomes; a sustained drop in scores is
flagged as a regression in ``change_log`` so a degraded loop is noticed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.change_log import log as log_change
from src.ingestion.schemas import database_name

_COLUMNS = ("run_id", "kind", "score", "flag", "adopted", "proposals", "requests",
            "tokens_in", "tokens_out", "cost_usd", "duration_ms", "status", "created_at")


def compute_score(*, adopted: int, proposals: int, requests: int, status: str) -> float:
    score = adopted * 2.0 + proposals * 0.5 + requests * 0.3
    if status != "done":
        score -= 1.0
    return round(score, 4)


def score_run(ch, run_id: str, *, kind: str = "optimizer", adopted: int = 0,
              proposals: int = 0, requests: int = 0, tokens_in: int = 0,
              tokens_out: int = 0, cost_usd: float = 0.0, duration_ms: float = 0.0,
              status: str = "done") -> float:
    score = compute_score(adopted=adopted, proposals=proposals, requests=requests,
                          status=status)
    flag = "low" if score <= 0 else "ok"
    ch.insert(
        f"{database_name()}.agent_run_scores",
        [[run_id, kind, score, flag, adopted, proposals, requests, tokens_in,
          tokens_out, cost_usd, duration_ms, status, datetime.now(timezone.utc)]],
        column_names=list(_COLUMNS),
    )
    return score


def detect_regression(ch, window: int = 10, factor: float = 0.5) -> dict | None:
    """Compare the newest half of recent run scores to the older half."""
    rows = ch.query(
        f"SELECT score FROM {database_name()}.agent_run_scores FINAL "
        "ORDER BY created_at DESC LIMIT {n:UInt32}",
        parameters={"n": window},
    ).result_rows
    scores = [float(r[0]) for r in rows]
    if len(scores) < window:
        return None
    half = window // 2
    recent = sum(scores[:half]) / half
    older = sum(scores[half:]) / (len(scores) - half)
    if older > 0 and recent < older * factor:
        log_change(ch, "run_score_regression",
                   detail={"recent": recent, "older": older})
        return {"recent": recent, "older": older}
    return None
