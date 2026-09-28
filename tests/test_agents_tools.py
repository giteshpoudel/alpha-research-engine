import pytest
from pydantic import BaseModel

from src.agents import tools
from src.ingestion.schemas import get_clickhouse_client


class _Args(BaseModel):
    x: int


def test_register_and_invoke():
    tools.register("test_echo", "echo", _Args)(lambda a, ch: {"x": a.x})
    try:
        result = tools.invoke("test_echo", {"x": 3})
        assert result.ok and result.data == {"x": 3}
    finally:
        tools.TOOL_REGISTRY.pop("test_echo", None)


def test_invoke_unknown_tool():
    result = tools.invoke("does_not_exist")
    assert result.ok is False and "unknown tool" in result.error


def test_invoke_invalid_args():
    tools.register("test_req", "d", _Args)(lambda a, ch: a.x)
    try:
        result = tools.invoke("test_req", {})  # missing required x
        assert result.ok is False and "invalid args" in result.error
    finally:
        tools.TOOL_REGISTRY.pop("test_req", None)


def test_invoke_handler_error_captured():
    def boom(a, ch):
        raise ValueError("kaboom")

    tools.register("test_boom", "d", _Args)(boom)
    try:
        result = tools.invoke("test_boom", {"x": 1})
        assert result.ok is False and "kaboom" in result.error
    finally:
        tools.TOOL_REGISTRY.pop("test_boom", None)


@pytest.fixture
def ch_client():
    return get_clickhouse_client()


def test_tool_scopes_enforced(ch_client):
    denied = tools.invoke("run_backtest", {"symbols": ["BTC"]}, ch=ch_client,
                          allow=("read",))
    assert denied.ok is False and "scope denied" in denied.error

    allowed = tools.invoke("classify_universe", {"symbols": ["BTC"]}, ch=ch_client,
                           allow=("read",))
    assert allowed.ok is True

    scopes = {t["name"]: t["scope"] for t in tools.describe()}
    assert scopes["register_strategy"] == "admin"
    assert scopes["run_backtest"] == "write"
    assert scopes["classify_universe"] == "read"


def test_classify_universe_tool(ch_client):
    result = tools.invoke("classify_universe", {"symbols": ["BTC", "DOGE"]}, ch=ch_client)
    assert result.ok
    cats = {row["symbol"]: row["category"] for row in result.data}
    assert cats["DOGE"] == "meme" and cats["BTC"] == "major"


def test_system_health_tool(ch_client):
    result = tools.invoke("system_health", {}, ch=ch_client)
    assert result.ok and "ohlcv_latest" in result.data


def test_request_tools(ch_client):
    from src.ingestion.schemas import database_name

    created = tools.invoke("create_request",
                           {"title": "TESTREQ tool", "justification": "j",
                            "kind": "data", "expected_impact": "+"}, ch=ch_client)
    try:
        assert created.ok and created.data["request_id"]
        assert tools.invoke("list_requests", {}, ch=ch_client).ok
    finally:
        ch_client.command(
            f"ALTER TABLE {database_name()}.agent_requests DELETE WHERE title LIKE 'TESTREQ%'",
            settings={"mutations_sync": 1})


def test_universe_tools(ch_client):
    from src.ingestion.schemas import database_name

    added = tools.invoke("add_symbol",
                         {"symbol": "TESTCOIN", "aliases": ["testcoin"], "backfill": False},
                         ch=ch_client)
    try:
        assert added.ok and added.data["enabled"]
        assert tools.invoke("list_universe", {}, ch=ch_client).ok
        removed = tools.invoke("remove_symbol", {"symbol": "TESTCOIN"}, ch=ch_client)
        assert removed.ok and removed.data["enabled"] is False
    finally:
        ch_client.command(
            f"ALTER TABLE {database_name()}.universe DELETE WHERE symbol = 'TESTCOIN'",
            settings={"mutations_sync": 1},
        )
