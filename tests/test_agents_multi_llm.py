import httpx
import pytest

from src.agents import multi_llm
from src.agents.multi_llm import Cooldown, chat


def _set_keys(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "mk")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk")
    monkeypatch.setenv("KIMI_PLANNER_MODEL", "kimi-k3")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr("time.sleep", lambda s: None)  # no backoff delays in tests


def test_endpoint_order(monkeypatch):
    _set_keys(monkeypatch)
    assert [e.name for e in multi_llm.endpoints("planner")] == ["moonshot", "deepseek", "ollama"]
    assert [e.name for e in multi_llm.endpoints("executor")] == ["deepseek", "moonshot", "ollama"]


def test_provider_defaults_and_temperature(monkeypatch):
    _set_keys(monkeypatch)
    planner, executor = multi_llm.endpoints("planner")[0], multi_llm.endpoints("executor")[0]
    assert planner.name == "moonshot" and planner.model == "kimi-k3"
    assert planner.temperature == 1.0  # kimi-k3 requires 1.0
    assert executor.name == "deepseek" and executor.model == "deepseek-chat"
    assert executor.temperature == 0.3


def test_endpoints_ollama_only_without_keys(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert [e.name for e in multi_llm.endpoints("planner")] == ["ollama"]


def test_chat_falls_back_on_failure(monkeypatch):
    _set_keys(monkeypatch)
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if "moonshot" in str(req.url):
            return httpx.Response(500)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = chat("planner", "sys", "usr", http_client=client, cooldown=Cooldown())
    assert result.text == "ok"
    assert result.model == "deepseek-chat"
    assert result.degraded is True
    assert result.tokens_in == 10 and result.tokens_out == 5
    assert any("moonshot" in u for u in calls) and any("deepseek" in u for u in calls)


def test_cooldown_skips_failing_provider(monkeypatch):
    _set_keys(monkeypatch)
    cd = Cooldown(threshold=1, seconds=300)
    cd.record_failure("moonshot")
    assert cd.should_skip("moonshot")

    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = chat("planner", "s", "u", http_client=client, cooldown=cd)
    assert result.model == "deepseek-chat"  # moonshot skipped
