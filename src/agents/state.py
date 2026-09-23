"""Pydantic-typed state and tool I/O for the autonomous optimizer.

Types are the contract between the planner (LLM), the executor, and the
deterministic tools: validated input, validated state transitions, and easy
JSON persistence for the trace tables.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class OptimizerState(BaseModel):
    """Mutable state threaded through an optimizer run."""

    goal_profit_pct: float = 0.0
    loops: int = 0
    max_loops: int = 3
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    universe: list[str] = Field(default_factory=list)
    pending_proposals: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---- tool argument models ----


class BacktestArgs(BaseModel):
    strategy: str = "mean_reversion"
    symbols: list[str] = Field(default_factory=lambda: ["all"])
    window: str = "OOS"


class FitArgs(BaseModel):
    strategy: str = "mean_reversion"
    symbol: str
    n_trials: int = 60


class ValidateArgs(BaseModel):
    strategy: str = "mean_reversion"
    symbol: str
    params: dict


class CompareArgs(BaseModel):
    strategy: str = "mean_reversion"
    symbol: str


class SignalArgs(BaseModel):
    buckets: list[str] = Field(default_factory=lambda: ["1h", "24h"])
    horizons: list[int] = Field(default_factory=lambda: [1, 4, 24])
    symbols: list[str] = Field(default_factory=list)


class ClassifyArgs(BaseModel):
    symbols: list[str] = Field(default_factory=list)


class EmptyArgs(BaseModel):
    pass
