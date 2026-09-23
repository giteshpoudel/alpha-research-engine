import json

import pytest

from src.agents import proposals
from src.ingestion.schemas import database_name, get_clickhouse_client

CHANGE = {"strategy": "TEST_prop", "symbol": "BTC",
          "params": {"window": 24, "z_entry": -2.0, "z_exit": 0.0}}


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    c.command(
        f"ALTER TABLE {database_name()}.strategy_proposals DELETE "
        "WHERE proposed_change LIKE '%TEST_prop%'",
        settings={"mutations_sync": 1},
    )


def test_create_pending_and_list(ch):
    pid = proposals.create_proposal(ch, "try tighter z_entry", CHANGE, model="kimi-k3")
    row = proposals.get_proposal(ch, pid)
    assert row["decision"] == "pending"
    assert row["hypothesis"] == "try tighter z_entry"
    assert row["model"] == "kimi-k3"
    assert any(r["proposal_id"] == pid for r in proposals.list_proposals(ch))


def test_evaluate_adopts_better_and_publishes(ch, monkeypatch):
    incumbent = {"window": 48, "z_entry": -3.0, "z_exit": 0.0}
    monkeypatch.setattr(proposals, "get_tuned_params", lambda *a, **k: incumbent)
    monkeypatch.setattr(proposals, "walk_forward_folds", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(
        proposals, "validate_params",
        lambda ch_, s, sym, params, folds: 0.5 if params == CHANGE["params"] else 0.1)
    published = []
    monkeypatch.setattr(proposals, "publish", lambda *a, **k: published.append(a))

    pid = proposals.create_proposal(ch, "better", CHANGE)
    row = proposals.evaluate_proposal(ch, pid)
    assert row["decision"] == "adopted"
    assert published, "adopted proposal should publish params"
    evidence = json.loads(row["evidence"])
    assert evidence["candidate_val"] == 0.5 and evidence["incumbent_val"] == 0.1


def test_evaluate_rejects_worse(ch, monkeypatch):
    incumbent = {"window": 48, "z_entry": -3.0, "z_exit": 0.0}
    monkeypatch.setattr(proposals, "get_tuned_params", lambda *a, **k: incumbent)
    monkeypatch.setattr(proposals, "walk_forward_folds", lambda *a, **k: [1])
    monkeypatch.setattr(
        proposals, "validate_params",
        lambda ch_, s, sym, params, folds: 0.1 if params == CHANGE["params"] else 0.5)
    published = []
    monkeypatch.setattr(proposals, "publish", lambda *a, **k: published.append(a))

    pid = proposals.create_proposal(ch, "worse", CHANGE)
    row = proposals.evaluate_proposal(ch, pid)
    assert row["decision"] == "rejected"
    assert not published


def test_non_params_marked_manual(ch):
    pid = proposals.create_proposal(ch, "add symbol X", {"symbol": "PEPE"}, kind="symbol")
    row = proposals.evaluate_proposal(ch, pid)
    assert row["decision"] == "manual"
