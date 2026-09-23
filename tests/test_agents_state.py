import pytest
from pydantic import ValidationError

from src.agents.state import FitArgs, OptimizerState, ToolResult


def test_state_roundtrip():
    state = OptimizerState(goal_profit_pct=0.5, universe=["BTC", "DOGE"])
    again = OptimizerState.model_validate_json(state.model_dump_json())
    assert again.goal_profit_pct == 0.5
    assert again.universe == ["BTC", "DOGE"]
    assert again.max_loops == 3


def test_required_arg_validation():
    with pytest.raises(ValidationError):
        FitArgs(strategy="mean_reversion")  # symbol is required


def test_tool_result_defaults():
    r = ToolResult(ok=True)
    assert r.error is None and r.data is None and r.latency_ms == 0.0
