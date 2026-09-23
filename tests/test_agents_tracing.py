import pytest

from src.agents import tracing
from src.agents.tracing import Budget, BudgetExceeded, Tracer, cost_usd
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch():
    return get_clickhouse_client()


def _cleanup(ch, run_id):
    ch.command(f"ALTER TABLE {database_name()}.agent_steps DELETE WHERE run_id = '{run_id}'",
               settings={"mutations_sync": 1})
    ch.command(f"ALTER TABLE {database_name()}.agent_runs DELETE WHERE run_id = '{run_id}'",
               settings={"mutations_sync": 1})


def test_tracer_writes_run_and_steps(ch):
    run_id = None
    try:
        with Tracer(ch=ch).run(kind="TEST", goal={"profit": 0.5},
                               model_planner="kimi-k3", model_executor="deepseek-chat") as run:
            run_id = run.run_id
            run.step("tool_a", {"x": 1}, {"ok": True}, latency_ms=5.0)
            run.step("llm", {"role": "planner"}, {"text": "hi"},
                     tokens_in=100, tokens_out=50, model="kimi-k3")

        row = ch.query(
            f"SELECT status, tokens_in, tokens_out, model_planner "
            f"FROM {database_name()}.agent_runs FINAL WHERE run_id = {{r:String}}",
            parameters={"r": run_id},
        ).result_rows
        assert len(row) == 1
        status, tokens_in, tokens_out, planner = row[0]
        assert status == "done" and tokens_in == 100 and tokens_out == 50
        assert planner == "kimi-k3"

        steps = ch.query(
            f"SELECT count() FROM {database_name()}.agent_steps FINAL WHERE run_id = {{r:String}}",
            parameters={"r": run_id},
        ).result_rows[0][0]
        assert steps == 2
    finally:
        if run_id:
            _cleanup(ch, run_id)


def test_tracer_marks_failure(ch):
    run_id = None
    try:
        with pytest.raises(ValueError):
            with Tracer(ch=ch).run(kind="TEST") as run:
                run_id = run.run_id
                raise ValueError("boom")
        row = ch.query(
            f"SELECT status, summary FROM {database_name()}.agent_runs FINAL "
            "WHERE run_id = {r:String}", parameters={"r": run_id},
        ).result_rows[0]
        assert row[0] == "failed" and "boom" in row[1]
    finally:
        if run_id:
            _cleanup(ch, run_id)


def test_cost_usd(monkeypatch):
    tracing.MODEL_COSTS["test-model"] = (0.01, 0.03)
    try:
        assert cost_usd("test-model", 1000, 1000) == pytest.approx(0.04)
        assert cost_usd("unpriced", 1000, 1000) == 0.0
    finally:
        tracing.MODEL_COSTS.pop("test-model", None)


def test_budget_caps():
    steps = Budget(max_steps=2)
    steps.steps = 2
    with pytest.raises(BudgetExceeded, match="steps"):
        steps.check()

    cost = Budget(max_cost_usd=1.0)
    cost.cost_usd = 1.0
    with pytest.raises(BudgetExceeded, match="cost"):
        cost.check()

    assert Budget().check() is None  # no caps -> never raises
