# Phase 4 Research Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LangGraph multi-agent research engine producing a daily markdown research report (risk + macro/sentiment + synthesis), stored in ClickHouse and on disk, with Kimi/Ollama LLM backends behind one interface.

**Architecture:** Deterministic collectors (ClickHouse/Qdrant queries, fully tested) → thin LLM agent nodes (mockable `chat()` interface) → report synthesis → storage. LangGraph wires the 5 nodes; `python -m src.research.daily_report` runs it.

**Tech Stack:** Python 3.11, LangGraph (new), openai (new), clickhouse-connect, qdrant-client, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-phase4-research-agents-design.md`

## Global Constraints

- "Portfolio" is strategy-implied, never presented as real holdings — the report template includes the disclaimer verbatim.
- LLM selection: `MOONSHOT_API_KEY` set → Moonshot endpoint + `kimi-k2-0905-preview`; else Ollama `/v1` + `deepseek-r1:8b`. Tests must never require a live LLM (mock `chat()` or the endpoint-selection logic only).
- No LLM call in any test. The real report run is a post-merge controller step.
- Collectors are deterministic and read-only; agent prompts embed collector output.
- Report storage idempotent: re-running a date replaces rows (`ORDER BY (report_date, section)`) and overwrites the markdown file.
- LLM failure after one retry → sections written with `model='unavailable'` containing the deterministic data summary — never an empty/silent report.
- Test hygiene: report tests use `report_date = date(2026, 1, 15)` (a date no real report will use), teardown deletes `WHERE report_date = ...` with mutations_sync and removes the TEST output file.
- The existing 115 tests must stay green. New dependencies: `langgraph`, `openai`.

---

### Task 1: Dependencies + `research_reports` schema + LLM client

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ingestion/schemas.py` (one DDL template + register)
- Create: `src/research/__init__.py` (empty)
- Create: `src/research/llm.py`
- Test: `tests/test_schemas.py` (append), `tests/test_llm.py` (new)

**Interfaces:**
- Consumes: env keys `MOONSHOT_API_KEY`, `OLLAMA_HOST`; `create_clickhouse_schema`, `database_name`.
- Produces:
  - table `research_reports` (columns per spec: report_date Date, section LowCardinality(String), content String, model LowCardinality(String), created_at DateTime64(3); ReplacingMergeTree, ORDER BY (report_date, section))
  - `llm.KIMI_MODEL = "kimi-k2-0905-preview"`, `llm.OLLAMA_MODEL = "deepseek-r1:8b"`
  - `llm.active_model() -> str` — `KIMI_MODEL` if key set else `OLLAMA_MODEL`
  - `llm.chat(system: str, user: str, model: str | None = None, http_client=None) -> str` — one retry on failure, returns assistant content string

- [ ] **Step 1: Dependencies**

Append to `requirements.txt`:

```
langgraph>=0.2
openai>=1.50
```

Run: `/Users/giteshpoudel/Documents/GitHub/alpha-research-engine/.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_schemas.py`:

```python
def test_research_reports_table(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.research_reports").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["report_date"] == "Date"
    assert cols["section"] == "LowCardinality(String)"
    assert cols["content"] == "String"
    assert cols["model"] == "LowCardinality(String)"
    assert cols["created_at"] == "DateTime64(3)"
```

Create `tests/test_llm.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v -k research_reports; python -m pytest tests/test_llm.py -v`
Expected: schema test FAILs — `Code: 60 ... Table alpha.research_reports does not exist`. llm tests FAIL at collection — `ModuleNotFoundError: No module named 'src.research.llm'`.

- [ ] **Step 4: Add the DDL to `src/ingestion/schemas.py`**

Add after `_TUNED_PARAMS_DDL`:

```python
# One row per (report_date, section): re-running a day's report replaces it.
_RESEARCH_REPORTS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.research_reports
(
    report_date Date,
    section LowCardinality(String),
    content String,
    model LowCardinality(String),
    created_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (report_date, section)
"""
```

Extend the DDL loop:

```python
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL, _SENTIMENT_METRICS_DDL,
                _BACKTEST_RUNS_DDL, _BACKTEST_EQUITY_DDL, _TUNED_PARAMS_DDL, _RESEARCH_REPORTS_DDL):
        client.command(ddl.format(db=db))
```

- [ ] **Step 5: Implement `src/research/llm.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py tests/test_llm.py -v`
Expected: all pass (22 schema + 5 llm = 27... trust pytest's count: 1 new schema test + 5 llm tests pass, existing stay green).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/ingestion/schemas.py src/research/__init__.py src/research/llm.py tests/test_schemas.py tests/test_llm.py
git commit -m "feat: add research_reports schema and Kimi/Ollama LLM client"
```

---

### Task 2: Deterministic collectors (`collectors.py`)

**Files:**
- Create: `src/research/collectors.py`
- Test: `tests/test_collectors.py`

**Interfaces:**
- Consumes: `ingestion.schemas.database_name`; `backtesting.data.load_ohlcv` (30d windows); live ClickHouse.
- Produces (Task 3 passes these dicts into prompts):
  - `collectors.spike_flags(velocity: float, engagement_ratio: float) -> list[str]` — `["mention_spike"]` when velocity > 2, `["engagement_spike"]` when engagement_ratio > 2
  - `collectors.collect_risk_data(ch_client) -> dict` — keys: `top_correlated` (list of {pair, corr}, top 5), `top_volatile` (list of {symbol, ann_vol}, top 5), `signal_states` (list of {symbol, state, z}), `tuned_drawdowns` (list of {symbol, current_dd}, from tuned-variant OOS equity curves)
  - `collectors.collect_macro_data(ch_client) -> dict` — keys: `sentiment` (per ticker: weighted_score, velocity, engagement_ratio, flags), `price_changes_24h` (per ticker pct), `funding_extremes` (top 3 by |rate|), `top_posts` (3 posts: {post_id, source, tickers, likes, score, text_excerpt})

- [ ] **Step 1: Write the failing tests**

`tests/test_collectors.py`:

```python
import pytest

from src.research import collectors
from src.ingestion.schemas import get_clickhouse_client


@pytest.fixture(scope="module")
def ch_client():
    return get_clickhouse_client()


def test_spike_flags():
    assert collectors.spike_flags(2.5, 1.0) == ["mention_spike"]
    assert collectors.spike_flags(1.0, 3.0) == ["engagement_spike"]
    assert collectors.spike_flags(2.5, 3.0) == ["mention_spike", "engagement_spike"]
    assert collectors.spike_flags(1.0, 1.0) == []


def test_collect_risk_data(ch_client):
    data = collectors.collect_risk_data(ch_client)
    assert set(data) == {"top_correlated", "top_volatile", "signal_states", "tuned_drawdowns"}
    for item in data["top_correlated"]:
        assert -1.0 <= item["corr"] <= 1.0
        assert " vs " in item["pair"]
    assert len(data["top_correlated"]) <= 5
    assert len(data["top_volatile"]) <= 5
    assert all(v["ann_vol"] > 0 for v in data["top_volatile"])
    states = {s["state"] for s in data["signal_states"]}
    assert states <= {"in", "out"}
    assert len(data["signal_states"]) == 12


def test_collect_macro_data(ch_client):
    data = collectors.collect_macro_data(ch_client)
    assert set(data) == {"sentiment", "price_changes_24h", "funding_extremes", "top_posts"}
    assert len(data["funding_extremes"]) <= 3
    assert len(data["top_posts"]) <= 3
    for item in data["sentiment"]:
        assert isinstance(item["flags"], list)
    if data["top_posts"]:
        post = data["top_posts"][0]
        assert post["likes"] >= data["top_posts"][-1]["likes"]  # sorted desc
        assert len(post["text_excerpt"]) <= 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collectors.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.research.collectors'`.

- [ ] **Step 3: Implement `src/research/collectors.py`**

```python
"""Deterministic data collectors for the research agents (no LLM).

Everything the agents reason about is computed here, so the LLM surface
stays small and every number is testable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.backtesting.strategies.mean_reversion import MR_DEFAULTS
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES

_LOOKBACK_DAYS = 30


def spike_flags(velocity: float, engagement_ratio: float) -> list[str]:
    flags = []
    if velocity > 2.0:
        flags.append("mention_spike")
    if engagement_ratio > 2.0:
        flags.append("engagement_spike")
    return flags


def _returns_30d(ch_client, symbol: str) -> pd.Series:
    start = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
    close = load_ohlcv(ch_client, symbol, start=start)["close"]
    return close.pct_change().dropna()


def _z_state(close: pd.Series) -> tuple[str, float]:
    window = MR_DEFAULTS["window"]
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = float((close.iloc[-1] - ma.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] else 0.0
    return ("in" if z <= MR_DEFAULTS["z_entry"] else "out"), z


def collect_risk_data(ch_client) -> dict:
    returns = {}
    for symbol in TICKER_ALIASES:
        try:
            returns[symbol] = _returns_30d(ch_client, symbol)
        except ValueError:
            continue
    frame = pd.DataFrame(returns).dropna()
    corr = frame.corr()
    pairs = []
    symbols = list(corr.columns)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pairs.append({"pair": f"{symbols[i]} vs {symbols[j]}",
                          "corr": float(corr.iloc[i, j])})
    pairs.sort(key=lambda p: -p["corr"])

    vols = [{"symbol": s, "ann_vol": float(frame[s].std() * np.sqrt(24 * 365))}
            for s in symbols]
    vols.sort(key=lambda v: -v["ann_vol"])

    signal_states = []
    for symbol in TICKER_ALIASES:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=7)
            close = load_ohlcv(ch_client, symbol, start=start)["close"]
            state, z = _z_state(close)
            signal_states.append({"symbol": symbol, "state": state, "z": round(z, 3)})
        except ValueError:
            signal_states.append({"symbol": symbol, "state": "out", "z": 0.0})

    rows = ch_client.query(
        f"""
        SELECT symbol, equity FROM (
            SELECT symbol, ts, equity,
                   row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
            FROM {database_name()}.backtest_equity FINAL
            WHERE run_id IN (
                SELECT run_id FROM {database_name()}.backtest_runs FINAL
                WHERE strategy = 'mean_reversion' AND window = 'OOS'
                  AND params_json LIKE '%"tuned"%' AND NOT startsWith(symbol, 'TEST')
            )
        ) WHERE rn = 1
        """
    ).result_rows
    tuned_drawdowns = []
    for symbol, latest_equity in rows:
        dd_rows = ch_client.query(
            f"""
            SELECT symbol, min(dd) AS max_dd FROM (
                SELECT e.symbol, e.ts, e.equity,
                       max(e.equity) OVER (PARTITION BY e.symbol ORDER BY e.ts
                                           ROWS UNBOUNDED PRECEDING) AS peak,
                       e.equity / peak - 1.0 AS dd
                FROM {database_name()}.backtest_equity e FINAL
                WHERE e.run_id IN (
                    SELECT run_id FROM {database_name()}.backtest_runs FINAL
                    WHERE strategy = 'mean_reversion' AND window = 'OOS'
                      AND params_json LIKE '%"tuned"%' AND NOT startsWith(symbol, 'TEST')
                )
            ) GROUP BY symbol
            """
        ).result_rows
        current_dd = {s: float(d) for s, d in dd_rows}.get(symbol, 0.0)
        tuned_drawdowns.append({"symbol": symbol, "current_dd": round(current_dd, 4)})

    return {
        "top_correlated": pairs[:5],
        "top_volatile": vols[:5],
        "signal_states": signal_states,
        "tuned_drawdowns": tuned_drawdowns,
    }


def collect_macro_data(ch_client) -> dict:
    rows = ch_client.query(
        f"""
        SELECT ticker, weighted_score, velocity, engagement_ratio FROM (
            SELECT ticker, bucket_start, weighted_score, velocity, engagement_ratio,
                   row_number() OVER (PARTITION BY ticker ORDER BY bucket_start DESC) AS rn
            FROM {database_name()}.sentiment_metrics FINAL
            WHERE bucket_size = '1h'
        ) WHERE rn = 1
        """
    ).result_rows
    sentiment = [
        {"ticker": t, "weighted_score": round(float(s), 3),
         "velocity": round(float(v), 2), "engagement_ratio": round(float(e), 2),
         "flags": spike_flags(float(v), float(e))}
        for t, s, v, e in rows
    ]

    price_changes = []
    for symbol in TICKER_ALIASES:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=2)
            close = load_ohlcv(ch_client, symbol, start=start)["close"]
            day_ago = close.index[-1] - pd.Timedelta(hours=24)
            past = close.loc[:day_ago]
            if len(past) > 0:
                change = float(close.iloc[-1] / past.iloc[-1] - 1.0)
                price_changes.append({"symbol": symbol, "pct_24h": round(change * 100, 2)})
        except ValueError:
            continue

    funding_rows = ch_client.query(
        f"""
        SELECT symbol, funding_rate FROM (
            SELECT symbol, ts, funding_rate,
                   row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
            FROM {database_name()}.funding_rates FINAL
        ) WHERE rn = 1 ORDER BY abs(funding_rate) DESC LIMIT 3
        """
    ).result_rows
    funding_extremes = [{"symbol": s, "rate": float(r)} for s, r in funding_rows]

    post_rows = ch_client.query(
        f"""
        SELECT post_id, source, tickers, likes, sentiment_score, text
        FROM {database_name()}.sentiment_posts FINAL
        WHERE published_at >= now64(3) - INTERVAL 2 DAY
          AND sentiment_score IS NOT NULL
        ORDER BY likes DESC LIMIT 3
        """
    ).result_rows
    top_posts = [
        {"post_id": pid, "source": src, "tickers": list(tks), "likes": int(lks),
         "score": round(float(sc), 3),
         "text_excerpt": (txt[:197] + "...") if len(txt) > 200 else txt}
        for pid, src, tks, lks, sc, txt in post_rows
    ]

    return {
        "sentiment": sentiment,
        "price_changes_24h": price_changes,
        "funding_extremes": funding_extremes,
        "top_posts": top_posts,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collectors.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/research/collectors.py tests/test_collectors.py
git commit -m "feat: add deterministic research data collectors"
```

---

### Task 3: Agents, graph, daily report CLI

**Files:**
- Create: `src/research/agents.py`
- Create: `src/research/graph.py`
- Create: `src/research/daily_report.py`
- Test: `tests/test_research_report.py`

**Interfaces:**
- Consumes: `collectors.collect_risk_data/collect_macro_data`, `llm.chat/active_model`, `ingestion.schemas` clients, LangGraph.
- Produces:
  - `agents.risk_agent(state: dict) -> dict` — returns `{"risk_section": str}`
  - `agents.macro_agent(state) -> dict` — returns `{"macro_section": str}`
  - `agents.report_agent(state) -> dict` — returns `{"report_md": str}`
  - `graph.build_graph() -> CompiledGraph`
  - `graph.run_report(ch_client, report_date, out_dir: Path | None = None, chat_fn=None) -> str` — full run: collectors → agents → store 3 sections + write `data/reports/{date}.md` (or out_dir); returns markdown
  - CLI: `python -m src.research.daily_report [--date YYYY-MM-DD]`

- [ ] **Step 1: Write the failing tests**

`tests/test_research_report.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research.graph import run_report

TEST_DATE = date(2026, 1, 15)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.research_reports DELETE WHERE report_date = {{d:Date}}",
        parameters={"d": TEST_DATE}, settings={"mutations_sync": 1},
    )


def _stub_chat(system: str, user: str, **kwargs) -> str:
    if "risk" in system:
        return "RISK: BTC vol elevated; BTC-ETH correlation 0.9."
    if "macro" in system:
        return "MACRO: sentiment bullish on SOL; funding neutral."
    return "# Daily Research Report\n\nSynthesis of risk and macro."


def test_run_report_stores_sections_and_file(ch_client, tmp_path: Path):
    md = run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=_stub_chat)
    assert "Daily Research Report" in md

    out_file = tmp_path / f"{TEST_DATE.isoformat()}.md"
    assert out_file.exists()
    assert "Synthesis" in out_file.read_text()

    rows = ch_client.query(
        f"SELECT section, model, length(content) > 0 FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date} ORDER BY section",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert [r[0] for r in rows] == ["macro", "report", "risk"]
    assert all(r[1] == "mock" for r in rows)
    assert all(r[2] for r in rows)

    # idempotent: re-run replaces, never duplicates
    run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=_stub_chat)
    count = ch_client.query(
        f"SELECT count() FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date}",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert count[0][0] == 3


def test_run_report_llm_failure_falls_back_to_data(ch_client, tmp_path: Path):
    def boom(system, user, **kwargs):
        raise RuntimeError("LLM request failed after retry: down")

    md = run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=boom)
    rows = ch_client.query(
        f"SELECT DISTINCT model FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date}",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert [r[0] for r in rows] == ["unavailable"]
    assert md  # deterministic data summary, never empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_research_report.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.research.graph'`.

- [ ] **Step 3: Implement `src/research/agents.py`**

```python
"""LLM agent nodes: thin prompt-in/text-out wrappers over llm.chat."""

from __future__ import annotations

import json

from src.research.llm import chat

_RISK_SYSTEM = (
    "You are a portfolio risk analyst for a crypto research desk. Analyze the "
    "provided data: signal states (strategy-implied book, not real holdings), "
    "30d correlations, realized volatility, and strategy drawdowns. Write a "
    "risk section (max 300 words): exposure summary, concentration/correlation "
    "risks, volatility outliers, and what to watch. Be specific with numbers."
)

_MACRO_SYSTEM = (
    "You are a macro & sentiment analyst for a crypto research desk. Analyze "
    "the provided data: sentiment metrics with spike flags, 24h price changes, "
    "funding rate extremes, and top social posts. Write a macro section (max "
    "300 words): market trend, sentiment divergences (price vs sentiment), and "
    "likely catalysts. Be specific with numbers."
)

_REPORT_SYSTEM = (
    "You are the editor of a daily crypto research report. Synthesize the risk "
    "and macro sections plus the raw data into a markdown daily report with: "
    "title with date, a one-paragraph executive summary, the risk section, the "
    "macro section, and a closing note. Include this disclaimer verbatim at "
    "the end: 'Portfolio figures are strategy-implied (research/backtest "
    "scope), not real holdings. Not financial advice.'"
)


def risk_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps(state["risk_data"], default=str)
    return {"risk_section": chat_fn(_RISK_SYSTEM, user)}


def macro_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps(state["macro_data"], default=str)
    return {"macro_section": chat_fn(_MACRO_SYSTEM, user)}


def report_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps({
        "date": str(state["report_date"]),
        "risk_section": state["risk_section"],
        "macro_section": state["macro_section"],
        "risk_data": state["risk_data"],
        "macro_data": state["macro_data"],
    }, default=str)
    return {"report_md": chat_fn(_REPORT_SYSTEM, user)}
```

- [ ] **Step 4: Implement `src/research/graph.py`**

```python
"""LangGraph wiring for the daily research report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research import agents, collectors
from src.research.llm import active_model, chat

_SECTION_COLUMNS = ("report_date", "section", "content", "model", "created_at")


class ReportState(TypedDict, total=False):
    report_date: date
    risk_data: dict
    macro_data: dict
    risk_section: str
    macro_section: str
    report_md: str


def build_graph(chat_fn=chat) -> object:
    graph = StateGraph(ReportState)

    def collect_risk(state: ReportState) -> dict:
        return {"risk_data": collectors.collect_risk_data(get_clickhouse_client())}

    def collect_macro(state: ReportState) -> dict:
        return {"macro_data": collectors.collect_macro_data(get_clickhouse_client())}

    graph.add_node("collect_risk", collect_risk)
    graph.add_node("collect_macro", collect_macro)
    graph.add_node("risk_agent", lambda s: agents.risk_agent(s, chat_fn))
    graph.add_node("macro_agent", lambda s: agents.macro_agent(s, chat_fn))
    graph.add_node("report_agent", lambda s: agents.report_agent(s, chat_fn))

    graph.add_edge(START, "collect_risk")
    graph.add_edge(START, "collect_macro")
    graph.add_edge("collect_risk", "risk_agent")
    graph.add_edge("collect_macro", "macro_agent")
    graph.add_edge("risk_agent", "report_agent")
    graph.add_edge("macro_agent", "report_agent")
    graph.add_edge("report_agent", END)
    return graph.compile()


def _data_summary(state: ReportState) -> str:
    lines = ["# Daily Research Report (data summary — LLM unavailable)", ""]
    lines.append("## Signal states (strategy-implied book)")
    for s in state["risk_data"]["signal_states"]:
        lines.append(f"- {s['symbol']}: {s['state']} (z={s['z']})")
    lines.append("")
    lines.append("## Sentiment")
    for item in state["macro_data"]["sentiment"][:5]:
        lines.append(f"- {item['ticker']}: score {item['weighted_score']}, "
                     f"velocity {item['velocity']}, flags {item['flags']}")
    lines.append("")
    lines.append("Portfolio figures are strategy-implied (research/backtest "
                 "scope), not real holdings. Not financial advice.")
    return "\n".join(lines)


def run_report(ch_client, report_date: date, out_dir: Path | None = None,
               chat_fn=chat) -> str:
    """Run the full report graph, store sections, write markdown. Returns markdown."""
    out_dir = out_dir or Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    state: ReportState = {"report_date": report_date}
    # Identity check: tests inject a stub chat_fn; production uses llm.chat.
    model = "mock" if chat_fn is not chat else active_model()
    try:
        result = build_graph(chat_fn).invoke(state)
        state.update(result)
        risk_section = state["risk_section"]
        macro_section = state["macro_section"]
        report_md = state["report_md"]
    except RuntimeError:
        model = "unavailable"
        # LLM failed: collectors may not have run yet in this process — collect directly.
        if "risk_data" not in state:
            state["risk_data"] = collectors.collect_risk_data(ch_client)
        if "macro_data" not in state:
            state["macro_data"] = collectors.collect_macro_data(ch_client)
        risk_section = _data_summary(state)
        macro_section = _data_summary(state)
        report_md = _data_summary(state)

    now = datetime.now(timezone.utc)
    rows = [
        [report_date, "risk", risk_section, model, now],
        [report_date, "macro", macro_section, model, now],
        [report_date, "report", report_md, model, now],
    ]
    ch_client.insert(
        f"{database_name()}.research_reports", rows,
        column_names=list(_SECTION_COLUMNS),
    )
    (out_dir / f"{report_date.isoformat()}.md").write_text(report_md)
    return report_md
```

- [ ] **Step 5: Implement `src/research/daily_report.py`**

```python
"""CLI: python -m src.research.daily_report [--date YYYY-MM-DD]"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from src.ingestion.schemas import get_clickhouse_client
from src.research.graph import run_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the daily research report")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="report date (default: today UTC)")
    args = parser.parse_args(argv)
    report_date = (date.fromisoformat(args.date) if args.date
                   else datetime.now(timezone.utc).date())
    md = run_report(get_clickhouse_client(), report_date)
    print(f"report for {report_date}: {len(md)} chars written")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_research_report.py -v`
Expected: 2 tests PASS. (The `model='mock'` assertion relies on `chat_fn is not chat` identity detection in run_report — tests inject stubs, production defaults to the real `llm.chat`.)

- [ ] **Step 7: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all tests PASS (115 + 1 schema + 5 llm + 3 collectors + 2 report = 126; trust pytest's actual count).

- [ ] **Step 8: Commit**

```bash
git add src/research/agents.py src/research/graph.py src/research/daily_report.py tests/test_research_report.py
git commit -m "feat: add LangGraph research agents and daily report CLI"
```

---

## Self-Review Notes

- **Spec coverage:** research_reports schema (Task 1), LLM client with endpoint selection + retry (Task 1), both collectors with exact keys (Task 2), three agents (Task 3), LangGraph wiring (Task 3), run_report idempotent storage + file (Task 3), LLM-failure fallback to data summary (Task 3 Step 4 `_data_summary` + test), disclaimer verbatim (Task 3 prompt + `_data_summary`), no live LLM in tests (Global Constraints + mocked chat_fn), new deps langgraph+openai (Task 1 Step 1). All spec sections covered.
- **Type consistency:** `chat(system, user, model=None, http_client=None) -> str`, `active_model() -> str`, `collect_risk_data(ch_client) -> dict`, `collect_macro_data(ch_client) -> dict`, `risk_agent/macro_agent/report_agent(state, chat_fn) -> dict`, `run_report(ch_client, date, out_dir=None, chat_fn=chat) -> str` — spelled identically in interfaces, code, and tests.
- **Test count:** 115 + 1 + 5 + 3 + 2 = 126 (Task 3 Step 7 states this).
- **Ordering dependency:** Task 3 consumes Tasks 1–2. Sequential execution required.
- **Known soft spot handled in design:** the `model='mock'` detection uses `chat_fn is not chat` identity (tests inject stubs; production defaults to `llm.chat`).
- **Real report run is NOT a plan step** — the controller runs `python -m src.research.daily_report` post-merge (uses Kimi if MOONSHOT_API_KEY is set, else Ollama fallback).
