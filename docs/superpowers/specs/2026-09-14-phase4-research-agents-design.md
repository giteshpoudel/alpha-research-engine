# Phase 4 Design — Multi-Agent Research Engine (LangGraph)

Date: 2026-09-14
Scope: Blueprint Phase 4 — three agents (Portfolio Risk, Macro & Sentiment, Report Generator) producing a daily research report, built on LangGraph with deterministic data collectors feeding small, mockable LLM analyst calls. Live portfolio tracking, real-time alerts, and dashboard rendering of reports are out of scope.

## Decisions (made under auto mode; user may veto)

- **"Portfolio" = strategy-implied book.** There are no real positions (research/paper scope). The Risk Agent reports: current tuned mean-reversion signal state per ticker (in/out + latest z-score), 30-day correlation matrix and realized volatility across the 12-asset universe, and drawdown status of the tuned OOS equity curves. Reports label this explicitly — never presented as a live portfolio.
- **LLM:** Kimi (Moonshot, OpenAI-compatible endpoint, model `kimi-k2-0905-preview`) primary; Ollama `deepseek-r1:8b` fallback when `MOONSHOT_API_KEY` is unset. One client interface (`src/research/llm.py`), so the choice is config, not code. Tests use the Ollama fallback or mocks — never require the key.
- **Graph shape:** `collect_risk_data` ∥ `collect_macro_data` → `risk_agent` ∥ `macro_agent` → `report_agent` → `store_report`. Collectors are deterministic and fully tested; LLM nodes are thin (prompt in, text out).
- **Output:** markdown report → new `research_reports` table (date, sections JSON, report_md) AND `data/reports/YYYY-MM-DD.md`.
- **New deps:** `langgraph`, `openai`.

## New schema (in `src/ingestion/schemas.py`)

### `research_reports`
| Column | Type |
| --- | --- |
| report_date | Date |
| section | LowCardinality(String) — `risk` / `macro` / `report` |
| content | String — agent's text (report section = full markdown) |
| model | LowCardinality(String) — LLM used (`kimi-k2-0905-preview` / `deepseek-r1:8b` / `mock`) |
| created_at | DateTime64(3) |

Engine: `ReplacingMergeTree`, `ORDER BY (report_date, section)` — re-running a day replaces its sections.

## New modules in `src/research/`

### `llm.py`
- `chat(system: str, user: str, model: str | None = None) -> str` — OpenAI-compatible chat call. Endpoint selection: `MOONSHOT_API_KEY` set → `https://api.moonshot.ai/v1` with that key, default model `kimi-k2-0905-preview`; else `http://{OLLAMA_HOST}:11434/v1` with dummy key, model `deepseek-r1:8b`. Timeout 120s, one retry. Returns the assistant text; also exposes `active_model() -> str` for report provenance.

### `collectors.py` (deterministic, no LLM)
- `collect_risk_data(ch_client) -> dict` — for the 12 tickers: 30-day hourly-return correlation matrix (top-5 correlated pairs), 30-day annualized volatility per ticker (top-5), current tuned-MR signal state per ticker (latest close vs 24-bar z-score → in/out + z), tuned OOS equity latest drawdown per symbol (from `backtest_equity` of the tuned runs).
- `collect_macro_data(ch_client, qd_client) -> dict` — latest 24h sentiment_metrics per ticker (weighted_score, velocity, engagement_ratio; flagged spikes: velocity > 2 or engagement_ratio > 2), 24h price change per ticker, funding rate extremes (top-3 by |latest rate|), and 3 representative recent high-engagement posts (top by likes from sentiment_posts, last 48h) with their scores.

### `agents.py` (LLM nodes)
- `risk_agent(state) -> str` — prompt with risk data → exposure/vol/correlation analysis section (≤ 300 words).
- `macro_agent(state) -> str` — prompt with macro data → trend/sentiment-divergence/catalyst section (≤ 300 words).
- `report_agent(state) -> str` — synthesizes both sections + data into the final daily markdown report (title, summary table, both sections, "strategy-implied book" disclaimer).

### `graph.py`
- `build_graph() -> CompiledGraph` — LangGraph StateGraph with the 5 nodes above; `run_report(ch_client, qd_client, report_date: date) -> str` executes the graph, stores all three sections into `research_reports`, writes `data/reports/{date}.md`, returns the markdown.
- CLI: `python -m src.research.daily_report [--date YYYY-MM-DD]` (default: today UTC). (`daily_report.py` is a thin wrapper calling `run_report`.)

## Error handling

- LLM endpoint unreachable after retry → report marked `model='unavailable'` with sections containing the deterministic data summary only (never a silent empty report).
- A collector that fails for one ticker logs and continues with the rest.
- Report generation is idempotent: re-running a date replaces its rows (ReplacingMergeTree) and overwrites the markdown file.

## Testing

- Collectors: integration tests against live ClickHouse — expected keys, sane values (correlation in [−1,1], vols > 0, signal states ∈ {in, out}), spike flagging on seeded thresholds (no seeding — assert flag logic on real rows where present, else unit-test the flag function).
- `llm.py`: endpoint selection logic (env var set/unset → right base_url/model), chat call via mocked HTTP (MockTransport).
- Agents: mocked `chat()` — assert prompts contain the collected data and sections come back through the graph state.
- `run_report` e2e with `chat` monkeypatched to a stub: markdown written to a TEST path, `research_reports` rows for the 3 sections with `model='mock'`, teardown deletes by report_date and removes the file.
- No live LLM call in the test suite (the real report run is a controller step).

## Known limitations (documented)

- "Portfolio" is strategy-implied, not real holdings; reports carry the disclaimer.
- Correlation/vol windows are fixed at 30 days (not configurable this iteration).
- Report quality with the Ollama fallback (8B reasoning model) is weaker than Kimi; provenance is recorded in `model`.
