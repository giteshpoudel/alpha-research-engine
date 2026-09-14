# Dashboard Design — Monitoring App (Backtesting View First)

Date: 2026-09-14
Scope: Web dashboard for the alpha-research-engine, iteration 1: Backtesting view over the stored `backtest_runs`/`backtest_equity` data. Paper-trading and live-trading views are designed-for later iterations (nav slots only, no stubs with fake data). No authentication (binds to 127.0.0.1).

## Decisions (made under auto mode; user may veto)

- **Stack:** FastAPI + Jinja2 templates + Chart.js from CDN. No JavaScript build tooling, no npm. Streamlit rejected (clunky multi-view growth, weaker showcase); SPA rejected (build tooling overkill). FastAPI is API-first: the same JSON endpoints serve later views and Phase 4 agents.
- **New dependencies:** `fastapi`, `uvicorn` only (Jinja2 is a FastAPI transitive dep).
- **Env toggle:** `APP_ENV` (`dev` default, `prod` reserved) in `.env`, read by `src/dashboard/config.py`. dev → local ClickHouse/Qdrant via existing env keys; prod is a named, empty slot for a future remote deployment.
- **Run:** `python -m src.dashboard` → uvicorn on `127.0.0.1:8000`.

## Structure: `src/dashboard/`

- `config.py` — `Settings` dataclass: `app_env`, `clickhouse_host/port/user/password/db` (from existing env keys), `DASHBOARD_PORT` (default 8000). `get_settings()` reads `.env` via python-dotenv.
- `queries.py` — read-only ClickHouse access:
  - `overview(client) -> list[dict]` — strategy, window, run count, avg(sharpe/sortino/max_dd/total_return/win_rate), per strategy×window.
  - `strategy_symbols(client, strategy) -> list[dict]` — per symbol: metrics for each window (IS/OOS/PRELIM) side by side.
  - `list_runs(client, strategy, symbol) -> list[dict]` — runs for picker (window, dates, metrics, run_id).
  - `run_detail(client, run_id) -> dict` — single run row.
  - `equity_curve(client, run_id) -> list[dict]` — ts/equity points, downsampled to ≤ 500 points (every Nth row) for chart performance.
- `app.py` — FastAPI app:
  - `GET /` → overview page (HTML)
  - `GET /strategies/{strategy}` → per-symbol comparison page
  - `GET /runs/{run_id}` → run detail page with equity chart
  - `GET /api/overview`, `/api/strategies/{strategy}`, `/api/runs/{run_id}`, `/api/runs/{run_id}/equity` → JSON twins of the pages (future views/agents consume these)
- `templates/` — `base.html` (nav: Backtesting active; Paper Trading / Live Trading as disabled slots), `overview.html`, `strategy.html`, `run.html`.
- `static/` — `style.css` (single small hand-written sheet, dark quant-terminal look). Chart.js from CDN with a local-fallback note in the template comment.
- `__main__.py` — `python -m src.dashboard` boots uvicorn with settings from config.

## Pages

1. **Overview (`/`)** — table: strategy | window | runs | avg Sharpe | avg Sortino | avg MaxDD | avg Return | avg Win Rate. Strategy names link to detail. PRELIM rows visually tagged (amber) as preliminary.
2. **Strategy detail (`/strategies/{strategy}`)** — table: symbol | IS metrics | OOS metrics (Sharpe, Return, MaxDD, Win Rate each), each row linking to its runs. Clear visual split between IS and OOS columns (guardrail #1 is a UI concern too: never average across windows).
3. **Run detail (`/runs/{run_id}`)** — metric cards + equity curve line chart (Chart.js, time axis). Window tag (IS/OOS/PRELIM) prominent.

## Error handling

- ClickHouse unreachable → friendly 503 page (no stack trace), JSON endpoints return `{"error": "clickhouse unavailable"}` with 503.
- Unknown run_id/strategy → 404 page.
- All queries FINAL reads, parameterized; read-only (no INSERT/ALTER anywhere in this package).

## Testing

- `tests/test_dashboard.py` — FastAPI TestClient against the live dev DB:
  - Pages render 200 and contain expected markers (strategy names, run links).
  - JSON endpoints return the documented shapes (keys per endpoint).
  - Unknown run_id → 404.
  - `equity_curve` downsampling: ≤ 500 points for the longest run, order preserved.
- No new test infrastructure; integration style matches the existing suite (live ClickHouse required).

## Non-goals this iteration

- No paper/live views, no auth, no websockets/live refresh (manual refresh is fine), no write actions (can't trigger backtests from the UI yet), no deployment work (prod slot is config structure only).
