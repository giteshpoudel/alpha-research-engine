import httpx
import pytest

from src.research import llm


def test_active_model_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    assert llm.active_model() == llm.OLLAMA_MODEL


def test_active_model_kimi_when_key_set(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    assert llm.active_model() == llm.KIMI_MODEL


def test_chat_uses_kimi_endpoint_when_key_set(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "analysis text"}}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert llm.chat("sys", "user", http_client=client) == "analysis text"
    assert "api.moonshot.ai" in seen["url"]
    assert seen["auth"] == "Bearer test-key"


def test_chat_falls_back_to_ollama(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "local text"}}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert llm.chat("sys", "user", http_client=client) == "local text"
    assert "11434" in seen["url"]


def test_chat_retries_once_then_raises(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="LLM request failed"):
        llm.chat("sys", "user", http_client=client)
    assert len(calls) == 2  # exactly one retry
