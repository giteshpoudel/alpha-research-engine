from datetime import date

import pytest

from src.agents import optimizer
from src.ingestion.schemas import database_name, get_clickhouse_client

DAY = date(2020, 2, 1)
STRATEGY = "TEST_mean_reversion"
PROPOSAL = ('{"action":"propose_params","symbol":"BTC",'
            '"params":{"window":24,"z_entry":-2.0,"z_exit":0.0},'
            '"hypothesis":"h","analysis":"a"}')
NONE = '{"action":"none"}'
REQUEST = ('{"action":"request","kind":"data","title":"TESTREQ social chatter",'
           '"justification":"social signal may help","expected_impact":"+0.3%"}')


class _Res:
    def __init__(self, text):
        self.text = text
        self.model = "mock"
        self.tokens_in = 5
        self.tokens_out = 3
        self.cost_usd = 0.0
        self.degraded = False


def test_extract_json():
    assert optimizer._extract_json("prefix " + PROPOSAL + " suffix")["action"] == "propose_params"
    assert optimizer._extract_json("no json here") is None


def test_validate_params():
    assert optimizer.validate_params({"window": 24, "z_entry": -2.0, "z_exit": 0.0})
    assert not optimizer.validate_params({"window": 100, "z_entry": -2.0, "z_exit": 0.0})
    assert not optimizer.validate_params({})


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.strategy_proposals DELETE "
              "WHERE proposed_change LIKE '%TEST_mean_reversion%'", settings={"mutations_sync": 1})
    c.command(f"ALTER TABLE {database_name()}.tuned_params DELETE "
              "WHERE strategy = 'TEST_mean_reversion'", settings={"mutations_sync": 1})
    c.command(f"ALTER TABLE {database_name()}.agent_goals DELETE WHERE goal_date = {{d:Date}}",
              parameters={"d": DAY}, settings={"mutations_sync": 1})


def _cleanup_run(ch, run_id):
    if run_id:
        ch.command(f"ALTER TABLE {database_name()}.agent_steps DELETE WHERE run_id = '{run_id}'",
                   settings={"mutations_sync": 1})
        ch.command(f"ALTER TABLE {database_name()}.agent_runs DELETE WHERE run_id = '{run_id}'",
                   settings={"mutations_sync": 1})


def test_run_optimizer_creates_and_evaluates_proposal(ch):
    def stub(role, system, user):
        return _Res(PROPOSAL)

    result = optimizer.run_optimizer(ch=ch, loops=1, strategy=STRATEGY,
                                     chat_fn=stub, day=DAY)
    try:
        assert result["loops"] == 1
        assert result["proposals"] and result["proposals"][0]["action"] == "proposal"
        assert result["proposals"][0]["decision"] in {"adopted", "rejected"}
        runs = ch.query(
            f"SELECT count() FROM {database_name()}.agent_runs FINAL WHERE run_id = {{r:String}}",
            parameters={"r": result["run_id"]}).result_rows[0][0]
        assert runs == 1
        steps = ch.query(
            f"SELECT count() FROM {database_name()}.agent_steps FINAL WHERE run_id = {{r:String}}",
            parameters={"r": result["run_id"]}).result_rows[0][0]
        assert steps >= 3  # observe + planner + executor + proposal
    finally:
        _cleanup_run(ch, result["run_id"])


def test_focus_symbol_prioritizes_halted(ch):
    observed = {
        "assignments": {"BTC": "mean_reversion", "ETH": "mean_reversion"},
        "paper": {"sleeves": [{"symbol": "BTC", "enabled": False},
                              {"symbol": "ETH", "enabled": True}]},
    }
    focus, strategy = optimizer._focus_symbol(ch, observed, set())
    assert focus == "BTC" and strategy == "mean_reversion"


def test_focus_symbol_avoids_recent(ch):
    observed = {"assignments": {"BTC": "mean_reversion", "ETH": "mean_reversion"},
                "paper": {"sleeves": []}}
    focus, _ = optimizer._focus_symbol(ch, observed, {"BTC"})
    assert focus == "ETH"


def test_run_optimizer_tune_iteration(ch, monkeypatch):
    monkeypatch.setattr(optimizer, "_observe", lambda c: {
        "health": {}, "symbols": [], "strategies": ["mean_reversion"],
        "assignments": {"BTC": "TEST_mean_reversion"}, "paper": {"sleeves": []},
    })
    result = optimizer.run_optimizer(ch=ch, loops=1, strategy="TEST_mean_reversion",
                                     focus_symbol="BTC", tune_trials=3, day=DAY)
    try:
        assert result["mode"] == "tune"
        assert result["proposals"] and result["proposals"][0].get("kind") == "params"
        assert result["proposals"][0]["decision"] in {"adopted", "rejected"}
    finally:
        _cleanup_run(ch, result["run_id"])


def test_run_optimizer_request_action(ch):
    def stub(role, system, user):
        return _Res(REQUEST)

    result = optimizer.run_optimizer(ch=ch, loops=1, strategy=STRATEGY,
                                     chat_fn=stub, day=DAY)
    try:
        assert result["proposals"] == []
        assert result["requests"] and result["requests"][0]["action"] == "request"
    finally:
        _cleanup_run(ch, result["run_id"])
        ch.command(f"ALTER TABLE {database_name()}.agent_requests DELETE "
                   "WHERE title LIKE 'TESTREQ%'", settings={"mutations_sync": 1})


SWITCH = ('{"action":"switch_strategy","symbol":"ZZZTESTSYM","strategy":"momentum",'
          '"hypothesis":"h","analysis":"a"}')


def test_run_optimizer_switch_strategy(ch):
    def stub(role, system, user):
        return _Res(SWITCH)

    result = optimizer.run_optimizer(ch=ch, loops=1, strategy=STRATEGY,
                                     chat_fn=stub, day=DAY)
    try:
        assert result["proposals"] and result["proposals"][0].get("kind") == "strategy"
        assert result["proposals"][0]["decision"] in {"adopted", "rejected"}
    finally:
        _cleanup_run(ch, result["run_id"])
        ch.command(f"ALTER TABLE {database_name()}.strategy_proposals DELETE "
                   "WHERE symbol = 'ZZZTESTSYM'", settings={"mutations_sync": 1})
        ch.command(f"ALTER TABLE {database_name()}.strategy_assignments DELETE "
                   "WHERE symbol = 'ZZZTESTSYM'", settings={"mutations_sync": 1})


def test_run_optimizer_no_action(ch):
    def stub(role, system, user):
        return _Res(NONE)

    result = optimizer.run_optimizer(ch=ch, loops=1, strategy=STRATEGY,
                                     chat_fn=stub, day=DAY)
    try:
        assert result["proposals"] == []
    finally:
        _cleanup_run(ch, result["run_id"])
