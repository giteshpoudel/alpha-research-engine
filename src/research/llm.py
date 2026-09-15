"""LLM client: Kimi (Moonshot) primary, local Ollama fallback.

Both expose OpenAI-compatible chat endpoints. The choice is pure config:
MOONSHOT_API_KEY set -> Kimi; otherwise local Ollama. Report provenance is
recorded via active_model().
"""

from __future__ import annotations

import os

import httpx

KIMI_MODEL = "kimi-k2-0905-preview"
OLLAMA_MODEL = "deepseek-r1:8b"
_KIMI_BASE_URL = "https://api.moonshot.ai/v1"


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "localhost")


def active_model() -> str:
    return KIMI_MODEL if os.environ.get("MOONSHOT_API_KEY") else OLLAMA_MODEL


def _endpoint() -> tuple[str, str, str]:
    """(base_url, api_key, default_model) for the active backend."""
    key = os.environ.get("MOONSHOT_API_KEY")
    if key:
        return _KIMI_BASE_URL, key, KIMI_MODEL
    return f"http://{_ollama_host()}:11434/v1", "ollama", OLLAMA_MODEL


def chat(system: str, user: str, model: str | None = None,
         http_client: httpx.Client | None = None) -> str:
    """One chat completion with a single retry. Returns the assistant text."""
    base_url, api_key, default_model = _endpoint()
    payload = {
        "model": model or default_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    try:
        last_error: Exception | None = None
        for _attempt in range(2):  # initial try + one retry
            try:
                resp = http_client.post(
                    f"{base_url}/chat/completions",
                    json=payload, headers=headers, timeout=120.0,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM request failed after retry: {last_error}")
    finally:
        if owns_client:
            http_client.close()
