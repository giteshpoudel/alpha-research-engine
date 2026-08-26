"""Shared HTTP helper: GET with timeouts and exponential backoff.

Retries 429, 5xx, and transport errors (1s, 2s, 4s ... capped at 10s,
max 4 attempts). Any other >=400 status raises immediately.
"""

from __future__ import annotations

import time

import httpx

USER_AGENT = "alpha-research-engine/0.1 (local quant research)"


def get_with_backoff(
    url: str,
    *,
    client: httpx.Client | None = None,
    headers: dict | None = None,
    params: dict | None = None,
    max_attempts: int = 4,
    timeout: float = 15.0,
) -> httpx.Response:
    merged_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    owns_client = client is None
    client = client or httpx.Client()
    delay = 1.0
    try:
        for attempt in range(max_attempts):
            try:
                resp = client.get(
                    url, headers=merged_headers, params=params,
                    timeout=timeout, follow_redirects=True,
                )
            except httpx.TransportError:
                resp = None
            if resp is not None and resp.status_code < 400:
                return resp
            status = resp.status_code if resp is not None else "connection error"
            retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == max_attempts - 1:
                raise RuntimeError(f"GET {url} failed with status {status}")
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
    finally:
        if owns_client:
            client.close()
    raise AssertionError("unreachable")
