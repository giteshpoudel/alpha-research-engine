import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.paper import carry_runner

TEST_STRATEGY = "TEST_funding_carry"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Never touch real 'funding_carry' production rows from tests.
    monkeypatch.setattr(carry_runner, "CARRY_STRATEGY", TEST_STRATEGY)


@pytest.fixture
def ch_client():
    c = get_clickhouse_client()
    yield c
    c.command(
        f"ALTER TABLE {database_name()}.paper_equity DELETE "
        f"WHERE strategy = '{TEST_STRATEGY}'",
        settings={"mutations_sync": 1},
    )


def test_carry_replay_is_idempotent(ch_client):
    n1 = carry_runner.replay(ch_client, "BTC")
    assert n1 > 0
    n2 = carry_runner.replay(ch_client, "BTC")
    assert n2 == n1
    rows = ch_client.query(
        f"SELECT count(), min(equity), max(equity) "
        f"FROM {database_name()}.paper_equity FINAL "
        f"WHERE strategy = '{TEST_STRATEGY}' AND symbol = 'BTC'"
    ).result_rows[0]
    assert rows[0] == n1
    assert rows[1] >= 0.0


def test_carry_step_adds_nothing_when_caught_up(ch_client):
    assert carry_runner.replay(ch_client, "ETH") > 0
    assert carry_runner.step(ch_client, "ETH") == 0


def test_carry_equity_accumulates_for_positive_funding(ch_client):
    carry_runner.replay(ch_client, "BTC")
    first, last = ch_client.query(
        f"SELECT argMin(equity, ts), argMax(equity, ts) "
        f"FROM {database_name()}.paper_equity FINAL "
        f"WHERE strategy = '{TEST_STRATEGY}' AND symbol = 'BTC'"
    ).result_rows[0]
    assert last > first  # net-positive funding over the window
