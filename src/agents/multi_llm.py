"""Planner/executor LLM clients with a fallback chain and usage capture.

- planner  → Moonshot kimi-k3 → DeepSeek → local Ollama
- executor → DeepSeek → Moonshot kimi-k3 → local Ollama

OpenAI-compatible chat endpoints. A provider that fails repeatedly is put on a
short cooldown (circuit breaker) so a rate-limited API isn't hammered. Every
result carries token usage and cost (cost is $0 until prices are configured in
``tracing.MODEL_COSTS``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from src.agents.tracing import cost_usd
from src.ingestion.http import ollama_base_url, post_with_backoff

OLLAMA_MODEL = "deepseek-r1:8b"


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    degraded: bool
    errors: list[str] | None = None  # provider failures encountered before success


@dataclass
class _Endpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float | None = 0.3  # kimi-k3 requires 1.0; None omits the field


class Cooldown:
    """Skip a provider after N consecutive failures for a cooldown window."""

    def __init__(self, threshold: int = 3, seconds: float = 300.0):
        self.threshold = threshold
        self.seconds = seconds
        self.failures: dict[str, int] = {}
        self.until: dict[str, float] = {}

    def should_skip(self, name: str) -> bool:
        return self.until.get(name, 0.0) > time.monotonic()

    def record_failure(self, name: str) -> None:
        count = self.failures.get(name, 0) + 1
        self.failures[name] = count
        if count >= self.threshold:
            self.until[name] = time.monotonic() + self.seconds

    def record_success(self, name: str) -> None:
        self.failures.pop(name, None)
        self.until.pop(name, None)


_GLOBAL_COOLDOWN = Cooldown()


def _moonshot() -> _Endpoint | None:
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        return None
    # kimi-k3 rejects any temperature other than 1.0.
    return _Endpoint("moonshot", "https://api.moonshot.ai/v1", key,
                     os.environ.get("KIMI_PLANNER_MODEL", "kimi-k3"), temperature=1.0)


def _deepseek() -> _Endpoint | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    return _Endpoint("deepseek",
                     os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                     key, os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))


def _ollama() -> _Endpoint:
    return _Endpoint("ollama", f"{ollama_base_url()}/v1", "ollama", OLLAMA_MODEL)


def endpoints(role: str) -> list[_Endpoint]:
    """Ordered provider candidates for a role."""
    moon, deep, oll = _moonshot(), _deepseek(), _ollama()
    order = [moon, deep, oll] if role == "planner" else [deep, moon, oll]
    return [ep for ep in order if ep is not None]


def _call(ep: _Endpoint, system: str, user: str, http_client: httpx.Client) -> LLMResult:
    payload: dict = {
        "model": ep.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if ep.temperature is not None:
        payload["temperature"] = ep.temperature
    resp = post_with_backoff(
        f"{ep.base_url}/chat/completions",
        json=payload,
        client=http_client,
        headers={"Authorization": f"Bearer {ep.api_key}"},
        max_attempts=2,
        timeout=60.0,
    )
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    tokens_in = int(usage.get("prompt_tokens", 0))
    tokens_out = int(usage.get("completion_tokens", 0))
    return LLMResult(text, ep.model, tokens_in, tokens_out,
                     cost_usd(ep.model, tokens_in, tokens_out), degraded=False)


def chat(role: str, system: str, user: str, http_client: httpx.Client | None = None,
         cooldown: Cooldown | None = None) -> LLMResult:
    """Chat as ``role`` with provider fallback. Raises if every provider fails."""
    candidates = endpoints(role)
    if not candidates:
        raise RuntimeError(f"no LLM providers configured for role '{role}'")
    preferred = candidates[0].name
    cooldown = cooldown or _GLOBAL_COOLDOWN
    owns_client = http_client is None
    http_client = http_client or httpx.Client()
    errors: list[str] = []
    try:
        for ep in candidates:
            if cooldown.should_skip(ep.name):
                continue
            try:
                result = _call(ep, system, user, http_client)
                cooldown.record_success(ep.name)
                result.degraded = ep.name != preferred
                result.errors = errors or None
                return result
            except Exception as exc:
                cooldown.record_failure(ep.name)
                errors.append(f"{ep.name}: {exc}")
        raise RuntimeError(f"all LLM providers failed for role '{role}': {errors}")
    finally:
        if owns_client:
            http_client.close()
