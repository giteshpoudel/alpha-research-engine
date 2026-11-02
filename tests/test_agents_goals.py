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
        f"ALTER TABLE {database_name()}.agent_goals DELETE "
        "WHERE goal_date >= '2020-01-01' AND goal_date <= '2020-01-31'",
        settings={"mutations_sync": 1},
    )


def test_ensure_creates_default(ch):
    row = goals.ensure_goal(ch, DAY)
    assert row["target_profit_pct"] == pytest.approx(goals.DEFAULT_DAILY_GOAL_PCT)
    assert row["status"] == "pending"


def test_update_met_then_raise(ch):
    goals.ensure_goal(ch, DAY, target=0.04)
    met = goals.update_goal(ch, DAY, 0.5)
    assert met["status"] == "met"
    raised = goals.raise_goal(ch, DAY, factor=2.0)
    assert raised["target_profit_pct"] == pytest.approx(0.08)
    assert raised["status"] == "raised"


def test_update_pending_when_below_target(ch):
    goals.ensure_goal(ch, DAY, target=0.08)
    row = goals.update_goal(ch, DAY, 0.02)
    assert row["status"] == "pending"


def test_daily_profit_pct_runs(ch):
    value = goals.daily_profit_pct(ch)
    assert value is None or isinstance(value, float)


def test_raise_once_per_day(ch):
    goals.ensure_goal(ch, DAY, target=0.04)
    goals.update_goal(ch, DAY, 0.5)
    first = goals.raise_goal(ch, DAY, factor=2.0)
    second = goals.raise_goal(ch, DAY, factor=2.0)
    assert first["target_profit_pct"] == pytest.approx(0.08)
    assert second["target_profit_pct"] == pytest.approx(0.08)  # no double raise


def test_rolling_target_uses_recent_achieved(ch):
    yesterday = date(2020, 1, 14)
    goals.ensure_goal(ch, yesterday, target=0.1)
    goals.update_goal(ch, yesterday, 4.0)  # achieved +4%
    row = goals.ensure_goal(ch, DAY)       # 0.5 * 4.0 = 2.0, capped
    assert row["target_profit_pct"] == pytest.approx(goals.MAX_DAILY_GOAL_PCT)


def test_rolling_target_below_cap(ch):
    yesterday = date(2020, 1, 14)
    goals.ensure_goal(ch, yesterday, target=0.02)
    goals.update_goal(ch, yesterday, 0.12)
    row = goals.ensure_goal(ch, DAY)       # 0.5 * 0.12
    assert row["target_profit_pct"] == pytest.approx(0.06)


def test_targets_match_annual_band():
    assert goals.annualized_pct(goals.DEFAULT_DAILY_GOAL_PCT) == pytest.approx(10.0, abs=0.1)
    assert goals.annualized_pct(goals.MAX_DAILY_GOAL_PCT) < 50.0


def test_old_uncapped_goal_is_clamped(ch):
    goals.ensure_goal(ch, DAY, target=10.0)
    row = goals.ensure_goal(ch, DAY)
    assert row["target_profit_pct"] == pytest.approx(goals.MAX_DAILY_GOAL_PCT)


def test_daily_profit_compares_each_series_to_itself(ch):
    from datetime import datetime
    db = database_name()
    t1, t0 = datetime(2020, 1, 20, 12), datetime(2020, 1, 19, 12)
    rows = [  # A: 1.00 -> 1.01 (+1%); B: 2.00 -> 1.98 (-1%); C only at t1 (no baseline)
        ["TESTGOAL_a", "X", t0, 1.00], ["TESTGOAL_a", "X", t1, 1.01],
        ["TESTGOAL_b", "X", t0, 2.00], ["TESTGOAL_b", "X", t1, 1.98],
        ["TESTGOAL_c", "X", t1, 5.00],
    ]
    cols = ["strategy", "symbol", "ts", "equity", "cash", "position_value", "mark_price",
            "status", "created_at"]
    ch.insert(f"{db}.paper_equity", [r + [r[3], 0.0, 1.0, "ok", t1] for r in rows],
              column_names=cols)
    try:
        # only look at the test series: pretend everything else is TEST_-prefixed
        q = ch.query
        def scoped(sql, parameters=None):
            sql = sql.replace("NOT startsWith(strategy, 'TEST_')",
                              "startsWith(strategy, 'TESTGOAL_')")
            return q(sql, parameters=parameters)
        ch.query = scoped
        assert goals.daily_profit_pct(ch) == pytest.approx(0.0, abs=1e-9)
    finally:
        ch.query = q
        ch.command(f"ALTER TABLE {db}.paper_equity DELETE WHERE startsWith(strategy, 'TESTGOAL_')",
                   settings={"mutations_sync": 1})
