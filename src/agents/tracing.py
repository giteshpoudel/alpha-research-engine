"""Run/step tracing plus cost and budget accounting for the optimizer.

Every LLM call and tool call should pass through a ``Tracer`` run so the system
is auditable ("why did it do that?") and bounded (cost/step caps). DB-backed by
design — no external tracing stack.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from src.ingestion.schemas import database_name, get_clickhouse_client

# model id -> (usd per 1k input tokens, usd per 1k output tokens). Left empty by
# default: tokens are always tracked, $ is computed only once real prices are
# configured (so we never bake in wrong numbers).
MODEL_COSTS: dict[str, tuple[float, float]] = {}

_RUN_COLUMNS = ("run_id", "kind", "goal_json", "status", "started_at", "ended_at",
                "loops", "tokens_in", "tokens_out", "cost_usd", "model_planner",
                "model_executor", "summary")
_STEP_COLUMNS = ("run_id", "step_idx", "tool", "args_json", "result_json",
                 "latency_ms", "error", "tokens_in", "tokens_out", "cost_usd",
                 "created_at")


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in MODEL_COSTS:
        return 0.0
    cin, cout = MODEL_COSTS[model]
    return tokens_in / 1000 * cin + tokens_out / 1000 * cout


class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its step or cost budget."""


class Budget:
    """Simple runaway guard for autonomous runs (steps and/or cost)."""

    def __init__(self, max_cost_usd: float = 0.0, max_steps: int = 0):
        self.max_cost_usd = max_cost_usd
        self.max_steps = max_steps
        self.cost_usd = 0.0
        self.steps = 0

    def check(self) -> None:
        if self.max_steps and self.steps >= self.max_steps:
            raise BudgetExceeded(f"max steps reached ({self.max_steps})")
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"max cost reached (${self.max_cost_usd})")


class Run:
    """Handle returned by ``Tracer.run()``; records steps into the active run."""

    def __init__(self, run_id: str, ch):
        self.run_id = run_id
        self.ch = ch
        self.step_idx = 0
        self.loops = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0

    def step(self, tool: str, args, result, tokens_in: int = 0, tokens_out: int = 0,
             latency_ms: float = 0.0, error: str = "", model: str = "") -> None:
        now = datetime.now(timezone.utc)
        cost = cost_usd(model, tokens_in, tokens_out)
        self.ch.insert(
            f"{database_name()}.agent_steps",
            [[self.run_id, self.step_idx, tool, json.dumps(args, default=str),
              json.dumps(result, default=str), float(latency_ms), str(error),
              int(tokens_in), int(tokens_out), cost, now]],
            column_names=list(_STEP_COLUMNS),
        )
        self.step_idx += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost


class _RunContext:
    def __init__(self, tracer: "Tracer", kind: str, goal, model_planner: str,
                 model_executor: str):
        self.tracer = tracer
        self.kind = kind
        self.goal = goal or {}
        self.model_planner = model_planner
        self.model_executor = model_executor

    def _write(self, status: str, ended_at, run: Run | None, summary: str) -> None:
        now = datetime.now(timezone.utc)
        self.ch.insert(
            f"{database_name()}.agent_runs",
            [[self.run_id, self.kind, json.dumps(self.goal, default=str), status,
              self.started_at, ended_at, run.loops if run else 0,
              run.tokens_in if run else 0, run.tokens_out if run else 0,
              run.cost_usd if run else 0.0, self.model_planner, self.model_executor,
              summary]],
            column_names=list(_RUN_COLUMNS),
        )

    def __enter__(self) -> Run:
        self.ch = self.tracer.ch or get_clickhouse_client()
        self.run_id = uuid.uuid4().hex[:16]
        self.started_at = datetime.now(timezone.utc)
        self._write("running", None, None, "")
        self.run = Run(self.run_id, self.ch)
        return self.run

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "failed" if exc_type else "done"
        summary = str(exc) if exc else ""
        self._write(status, datetime.now(timezone.utc), self.run, summary)
        return False


class Tracer:
    def __init__(self, ch=None):
        self.ch = ch

    def run(self, kind: str = "optimizer", goal=None, model_planner: str = "",
            model_executor: str = "") -> _RunContext:
        return _RunContext(self, kind, goal, model_planner, model_executor)
