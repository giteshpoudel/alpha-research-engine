import httpx
import pytest

from src.ingestion.http import get_with_backoff


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_success_first_try():
    client = _client(lambda req: httpx.Response(200, json={"ok": True}))
    resp = get_with_backoff("https://example.com/x", client=client)
    assert resp.status_code == 200


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    resp = get_with_backoff("https://example.com/x", client=_client(handler))
    assert resp.status_code == 200
    assert len(calls) == 3


def test_non_retryable_status_raises_immediately():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(403)

    with pytest.raises(RuntimeError, match="403"):
        get_with_backoff("https://example.com/x", client=_client(handler))
    assert len(calls) == 1


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="429"):
        get_with_backoff("https://example.com/x",
                         client=_client(lambda req: httpx.Response(429)))
