from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from src.agents import tools
from src.agents.breakers import BreakerStore
from src.agents.multi_llm import Cooldown
from src.ingestion.schemas import database_name, get_clickhouse_client


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.circuit_breakers "
              "DELETE WHERE breaker LIKE '%test%'", settings={"mutations_sync": 1})


def test_breaker_store_roundtrip(ch):
    store = BreakerStore(ch)
    assert store.get("test_provider") == (0, None)
    until = datetime.now(timezone.utc) + timedelta(minutes=5)
    store.save("test_provider", 3, until)
    failures, opened = store.get("test_provider")
    assert failures == 3 and opened is not None


def test_cooldown_persists_across_instances(ch):
    store = BreakerStore(ch)
    first = Cooldown(threshold=2, seconds=300.0, store=store)
    first.record_failure("test_llm")
    first.record_failure("test_llm")   # threshold reached -> opened
    # a fresh Cooldown (new process) must still see it open
    second = Cooldown(threshold=2, seconds=300.0, store=store)
    assert second.should_skip("test_llm") is True


def test_tool_breaker_opens_after_failures(ch):
    class _Args(BaseModel):
        pass

    def boom(a, ch_):
        raise ValueError("kaboom")

    tools.register("test_breaker", "d", _Args)(boom)
    try:
        for _ in range(5):
            result = tools.invoke("test_breaker", {}, ch=ch)
            assert result.ok is False and "kaboom" in result.error
        opened = tools.invoke("test_breaker", {}, ch=ch)
        assert opened.ok is False and "circuit open" in opened.error
    finally:
        tools.TOOL_REGISTRY.pop("test_breaker", None)
        ch.command(f"ALTER TABLE {database_name()}.circuit_breakers "
                   "DELETE WHERE breaker = 'tool:test_breaker'",
                   settings={"mutations_sync": 1})
