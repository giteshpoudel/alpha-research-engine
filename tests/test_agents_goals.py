from datetime import date

import pytest

from src.agents import goals
from src.ingestion.schemas import database_name, get_clickhouse_client

DAY = date(2020, 1, 15)


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(
        f"ALTER TABLE {database_name()}.agent_goals DELETE WHERE goal_date = {{d:Date}}",
        parameters={"d": DAY}, settings={"mutations_sync": 1},
    )


def test_ensure_creates_default(ch):
    row = goals.ensure_goal(ch, DAY)
    assert row["target_profit_pct"] == pytest.approx(goals.DEFAULT_DAILY_GOAL_PCT)
    assert row["status"] == "pending"


def test_update_met_then_raise(ch):
    goals.ensure_goal(ch, DAY, target=0.1)
    met = goals.update_goal(ch, DAY, 0.5)
    assert met["status"] == "met"
    raised = goals.raise_goal(ch, DAY, factor=2.0)
    assert raised["target_profit_pct"] == pytest.approx(0.2)
    assert raised["status"] == "raised"


def test_update_pending_when_below_target(ch):
    goals.ensure_goal(ch, DAY, target=1.0)
    row = goals.update_goal(ch, DAY, 0.2)
    assert row["status"] == "pending"


def test_daily_profit_pct_runs(ch):
    value = goals.daily_profit_pct(ch)
    assert value is None or isinstance(value, float)
