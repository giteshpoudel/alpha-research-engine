import pytest

from src.agents import requests as req
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.agent_requests DELETE "
              "WHERE title LIKE 'TESTREQ%'", settings={"mutations_sync": 1})


def test_create_list_decide(ch):
    rid = req.create_request(ch, "TESTREQ add X data", "would help profit",
                             kind="data", expected_impact="+0.2%")
    row = req.get_request(ch, rid)
    assert row["status"] == "open" and row["kind"] == "data"
    assert any(r["request_id"] == rid for r in req.list_requests(ch, status="open"))

    decided = req.decide_request(ch, rid, "approved", note="ok")
    assert decided["status"] == "approved" and decided["decision_note"] == "ok"


def test_decide_rejects_bad_status(ch):
    rid = req.create_request(ch, "TESTREQ bad", "j")
    with pytest.raises(ValueError, match="approved"):
        req.decide_request(ch, rid, "bogus")
