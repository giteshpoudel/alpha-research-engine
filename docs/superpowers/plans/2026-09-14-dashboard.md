# Dashboard Implementation Plan (Iteration 1: Backtesting View)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web dashboard for monitoring backtest results — overview, per-symbol strategy comparison, and run detail with equity curves — served by FastAPI with server-rendered pages and JSON twins of every page.

**Architecture:** `src/dashboard/` package: `config.py` (env incl. `APP_ENV` toggle), `queries.py` (read-only ClickHouse access), `app.py` (FastAPI routes), Jinja2 templates + one static stylesheet, Chart.js from CDN for equity curves. `python -m src.dashboard` boots uvicorn on 127.0.0.1:8000.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Jinja2 (transitive), clickhouse-connect, pytest + FastAPI TestClient (httpx already installed).

**Spec:** `docs/superpowers/specs/2026-09-14-dashboard-design.md`

## Global Constraints

- Read-only package: no INSERT/ALTER/DELETE anywhere in `src/dashboard/`. All queries FINAL reads, parameterized.
- Bind to `127.0.0.1` only (no auth this iteration).
- Never average across IS/OOS windows in any aggregation — windows are always separate rows/columns (guardrail #1 is a UI concern).
- PRELIM rows/runs are visually tagged preliminary everywhere they appear.
- Equity charts downsample to ≤ 500 points (every Nth row), order preserved.
- JSON endpoints mirror every HTML page (same data, same shapes).
- New dependencies: `fastapi`, `uvicorn` only. The existing 96 tests must stay green.
- Test style matches the existing suite: live ClickHouse required, no mocks for DB tests. Dashboard tests must not write to any table.

---

### Task 1: Config, queries, dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `src/dashboard/__init__.py` (empty)
- Create: `src/dashboard/config.py`
- Create: `src/dashboard/queries.py`
- Test: `tests/test_dashboard_queries.py`

**Interfaces:**
- Consumes: `src/ingestion/schemas.get_clickhouse_client`, `database_name`.
- Produces (Task 2 uses):
  - `config.Settings` dataclass with fields `app_env: str`, `clickhouse_host: str`, `clickhouse_port: int`, `clickhouse_user: str`, `clickhouse_password: str`, `clickhouse_db: str`, `dashboard_port: int`; `config.get_settings() -> Settings`
  - `queries.overview(client) -> list[dict]` — keys: `strategy, window, runs, avg_sharpe, avg_sortino, avg_max_drawdown, avg_total_return, avg_win_rate`
  - `queries.strategy_symbols(client, strategy) -> list[dict]` — keys: `symbol, window, runs, avg_sharpe, avg_total_return, avg_max_drawdown, avg_win_rate`
  - `queries.list_runs(client, strategy, symbol) -> list[dict]` — keys: `run_id, window, start_ts, end_ts, sharpe, total_return`
  - `queries.run_detail(client, run_id) -> dict | None` — all backtest_runs columns
  - `queries.equity_curve(client, run_id, max_points=500) -> list[dict]` — keys `ts, equity`, ≤ max_points, chronological

- [ ] **Step 1: Dependencies**

Append to `requirements.txt`:

```
fastapi>=0.115
uvicorn>=0.30
```

Run: `/Users/giteshpoudel/Documents/GitHub/alpha-research-engine/.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests**

`tests/test_dashboard_queries.py`:

```python
import pytest

from src.dashboard import queries
from src.ingestion.schemas import get_clickhouse_client


@pytest.fixture(scope="module")
def client():
    return get_clickhouse_client()


def test_overview_shape_and_window_separation(client):
    rows = queries.overview(client)
    assert rows, "expected backtest runs in the dev DB"
    keys = set(rows[0])
    assert keys == {"strategy", "window", "runs", "avg_sharpe", "avg_sortino",
                    "avg_max_drawdown", "avg_total_return", "avg_win_rate"}
    windows = {(r["strategy"], r["window"]) for r in rows}
    # IS and OOS are never merged into one row
    assert ("mean_reversion", "IS") in windows
    assert ("mean_reversion", "OOS") in windows


def test_strategy_symbols_shape(client):
    rows = queries.strategy_symbols(client, "mean_reversion")
    assert rows
    assert set(rows[0]) == {"symbol", "window", "runs", "avg_sharpe",
                            "avg_total_return", "avg_max_drawdown", "avg_win_rate"}
    symbols = {r["symbol"] for r in rows}
    assert "BTC" in symbols


def test_list_runs_and_detail(client):
    runs = queries.list_runs(client, "mean_reversion", "BTC")
    assert runs
    assert set(runs[0]) >= {"run_id", "window", "start_ts", "end_ts", "sharpe", "total_return"}
    detail = queries.run_detail(client, runs[0]["run_id"])
    assert detail is not None
    assert detail["run_id"] == runs[0]["run_id"]
    assert detail["strategy"] == "mean_reversion"
    assert queries.run_detail(client, "nonexistent-run-id") is None


def test_equity_curve_downsampled(client):
    run_id = queries.list_runs(client, "mean_reversion", "BTC")[0]["run_id"]
    points = queries.equity_curve(client, run_id, max_points=500)
    assert 0 < len(points) <= 500
    assert set(points[0]) == {"ts", "equity"}
    timestamps = [p["ts"] for p in points]
    assert timestamps == sorted(timestamps)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_queries.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.dashboard'`.

- [ ] **Step 4: Implement `src/dashboard/config.py` and `src/dashboard/queries.py`**

`src/dashboard/config.py`:

```python
"""Dashboard configuration with dev/prod environment toggle.

APP_ENV=dev (default) uses the local containers from the shared .env.
APP_ENV=prod is a reserved slot for a future remote deployment — it reads
the same keys with a PROD_ prefix so a prod .env can diverge without code
changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_env: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_db: str
    dashboard_port: int


def get_settings() -> Settings:
    load_dotenv()
    app_env = os.environ.get("APP_ENV", "dev")
    prefix = "PROD_" if app_env == "prod" else ""

    def env(name: str, default: str) -> str:
        return os.environ.get(prefix + name, os.environ.get(name, default))

    return Settings(
        app_env=app_env,
        clickhouse_host=env("CLICKHOUSE_HOST", "localhost"),
        clickhouse_port=int(env("CLICKHOUSE_PORT", "8123")),
        clickhouse_user=env("CLICKHOUSE_USER", "alpha"),
        clickhouse_password=env("CLICKHOUSE_PASSWORD", ""),
        clickhouse_db=env("CLICKHOUSE_DB", "alpha"),
        dashboard_port=int(os.environ.get("DASHBOARD_PORT", "8000")),
    )
```

`src/dashboard/queries.py`:

```python
"""Read-only ClickHouse queries for the dashboard.

Windows are always separate rows/columns — nothing here ever averages
across IS/OOS (guardrail #1 applies to the UI too).
"""

from __future__ import annotations

from src.ingestion.schemas import database_name


def overview(client) -> list[dict]:
    rows = client.query(
        f"""
        SELECT
            strategy,
            window,
            count() AS runs,
            avg(sharpe) AS avg_sharpe,
            avg(sortino) AS avg_sortino,
            avg(max_drawdown) AS avg_max_drawdown,
            avg(total_return) AS avg_total_return,
            avg(win_rate) AS avg_win_rate
        FROM {database_name()}.backtest_runs FINAL
        WHERE NOT startsWith(strategy, 'TEST_')
        GROUP BY strategy, window
        ORDER BY strategy, window
        """
    ).result_rows
    return [
        dict(zip(
            ("strategy", "window", "runs", "avg_sharpe", "avg_sortino",
             "avg_max_drawdown", "avg_total_return", "avg_win_rate"),
            row,
        ))
        for row in rows
    ]


def strategy_symbols(client, strategy: str) -> list[dict]:
    rows = client.query(
        f"""
        SELECT
            symbol,
            window,
            count() AS runs,
            avg(sharpe) AS avg_sharpe,
            avg(total_return) AS avg_total_return,
            avg(max_drawdown) AS avg_max_drawdown,
            avg(win_rate) AS avg_win_rate
        FROM {database_name()}.backtest_runs FINAL
        WHERE strategy = {{strategy:String}} AND NOT startsWith(strategy, 'TEST_')
        GROUP BY symbol, window
        ORDER BY symbol, window
        """,
        parameters={"strategy": strategy},
    ).result_rows
    return [
        dict(zip(
            ("symbol", "window", "runs", "avg_sharpe",
             "avg_total_return", "avg_max_drawdown", "avg_win_rate"),
            row,
        ))
        for row in rows
    ]


def list_runs(client, strategy: str, symbol: str) -> list[dict]:
    rows = client.query(
        f"""
        SELECT run_id, window, start_ts, end_ts, sharpe, total_return
        FROM {database_name()}.backtest_runs FINAL
        WHERE strategy = {{strategy:String}} AND symbol = {{symbol:String}}
        ORDER BY window, start_ts
        """,
        parameters={"strategy": strategy, "symbol": symbol},
    ).result_rows
    return [
        dict(zip(("run_id", "window", "start_ts", "end_ts", "sharpe", "total_return"), row))
        for row in rows
    ]


def run_detail(client, run_id: str) -> dict | None:
    rows = client.query(
        f"""
        SELECT run_id, strategy, symbol, interval, params_json, window,
               start_ts, end_ts, total_return, sharpe, sortino,
               max_drawdown, calmar, win_rate, num_trades, created_at
        FROM {database_name()}.backtest_runs FINAL
        WHERE run_id = {{run_id:String}}
        LIMIT 1
        """,
        parameters={"run_id": run_id},
    ).result_rows
    if not rows:
        return None
    return dict(zip(
        ("run_id", "strategy", "symbol", "interval", "params_json", "window",
         "start_ts", "end_ts", "total_return", "sharpe", "sortino",
         "max_drawdown", "calmar", "win_rate", "num_trades", "created_at"),
        rows[0],
    ))


def equity_curve(client, run_id: str, max_points: int = 500) -> list[dict]:
    count = client.query(
        f"SELECT count() FROM {database_name()}.backtest_equity FINAL "
        "WHERE run_id = {run_id:String}",
        parameters={"run_id": run_id},
    ).result_rows[0][0]
    if count == 0:
        return []
    stride = max(1, (count + max_points - 1) // max_points)
    rows = client.query(
        f"""
        SELECT ts, equity FROM (
            SELECT ts, equity, row_number() OVER (ORDER BY ts) - 1 AS rn
            FROM {database_name()}.backtest_equity FINAL
            WHERE run_id = {{run_id:String}}
        )
        WHERE rn % {{stride:UInt32}} = 0
        ORDER BY ts
        """,
        parameters={"run_id": run_id, "stride": stride},
    ).result_rows
    return [{"ts": row[0], "equity": float(row[1])} for row in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_queries.py -v`
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/dashboard/__init__.py src/dashboard/config.py src/dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat: add dashboard config and read-only query layer"
```

---

### Task 2: FastAPI app, templates, static, tests

**Files:**
- Create: `src/dashboard/app.py`
- Create: `src/dashboard/__main__.py`
- Create: `src/dashboard/templates/base.html`
- Create: `src/dashboard/templates/overview.html`
- Create: `src/dashboard/templates/strategy.html`
- Create: `src/dashboard/templates/run.html`
- Create: `src/dashboard/templates/error.html`
- Create: `src/dashboard/static/style.css`
- Test: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: `config.get_settings()`, all of `queries`, `get_clickhouse_client`.
- Produces:
  - `app.create_app() -> FastAPI` (testable without booting a server)
  - Routes: `GET /`, `GET /strategies/{strategy}`, `GET /runs/{run_id}` (HTML); `GET /api/overview`, `GET /api/strategies/{strategy}`, `GET /api/runs/{run_id}`, `GET /api/runs/{run_id}/equity` (JSON)
  - `python -m src.dashboard` boots uvicorn on `127.0.0.1:{DASHBOARD_PORT}` (default 8000)

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard_app.py`:

```python
import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_overview_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "mean_reversion" in resp.text
    assert "funding_arb" in resp.text
    assert "/strategies/mean_reversion" in resp.text


def test_strategy_page_renders(client):
    resp = client.get("/strategies/mean_reversion")
    assert resp.status_code == 200
    assert "BTC" in resp.text
    assert "IS" in resp.text and "OOS" in resp.text


def test_run_page_renders_with_chart(client):
    runs = client.get("/api/strategies/mean_reversion").json()
    assert runs
    run_id = client.get("/api/overview").json()
    # pick any run id from the list-runs API shape via strategy page data
    from src.dashboard import queries
    from src.ingestion.schemas import get_clickhouse_client
    rid = queries.list_runs(get_clickhouse_client(), "mean_reversion", "BTC")[0]["run_id"]
    resp = client.get(f"/runs/{rid}")
    assert resp.status_code == 200
    assert "chart" in resp.text.lower()
    assert rid in resp.text


def test_run_page_404(client):
    resp = client.get("/runs/nonexistent-run-id")
    assert resp.status_code == 404


def test_api_endpoints(client):
    overview = client.get("/api/overview").json()
    assert isinstance(overview, list) and overview
    assert {"strategy", "window", "runs", "avg_sharpe"} <= set(overview[0])

    symbols = client.get("/api/strategies/mean_reversion").json()
    assert isinstance(symbols, list) and symbols
    assert {"symbol", "window", "avg_sharpe"} <= set(symbols[0])

    from src.dashboard import queries
    from src.ingestion.schemas import get_clickhouse_client
    rid = queries.list_runs(get_clickhouse_client(), "mean_reversion", "BTC")[0]["run_id"]
    detail = client.get(f"/api/runs/{rid}").json()
    assert detail["run_id"] == rid
    equity = client.get(f"/api/runs/{rid}/equity").json()
    assert isinstance(equity, list) and 0 < len(equity) <= 500
    assert {"ts", "equity"} == set(equity[0])

    assert client.get("/api/runs/nonexistent-run-id").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_app.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.dashboard.app'`.

- [ ] **Step 3: Implement the app**

`src/dashboard/app.py`:

```python
"""FastAPI dashboard: backtesting view (HTML pages + JSON twins)."""

from __future__ import annotations

import clickhouse_connect
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.dashboard import queries
from src.dashboard.config import get_settings

_PKG_DIR = Path(__file__).parent

WINDOW_ORDER = ("IS", "OOS", "PRELIM")


def _client_from_settings():
    s = get_settings()
    return clickhouse_connect.get_client(
        host=s.clickhouse_host, port=s.clickhouse_port,
        username=s.clickhouse_user, password=s.clickhouse_password,
    )


def _fmt(value):
    return round(float(value), 3) if value is not None else None


def _fmt_rows(rows):
    return [{k: (_fmt(v) if isinstance(v, float) else v) for k, v in row.items()} for row in rows]


def create_app() -> FastAPI:
    app = FastAPI(title="Alpha Research Dashboard")
    app.mount("/static", StaticFiles(directory=_PKG_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=_PKG_DIR / "templates")

    def ch():
        return _client_from_settings()

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "clickhouse unavailable"}, status_code=503)
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": str(exc)}, status_code=503
        )

    # ---------- HTML pages ----------

    @app.get("/", response_class=HTMLResponse)
    def overview_page(request: Request):
        rows = _fmt_rows(queries.overview(ch()))
        return templates.TemplateResponse(
            "overview.html", {"request": request, "rows": rows}
        )

    @app.get("/strategies/{strategy}", response_class=HTMLResponse)
    def strategy_page(request: Request, strategy: str):
        rows = _fmt_rows(queries.strategy_symbols(ch(), strategy))
        symbols: dict[str, dict] = {}
        for row in rows:
            symbols.setdefault(row["symbol"], {})[row["window"]] = row
        return templates.TemplateResponse(
            "strategy.html",
            {"request": request, "strategy": strategy,
             "symbols": symbols, "windows": WINDOW_ORDER},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        detail = queries.run_detail(ch(), run_id)
        if detail is None:
            return templates.TemplateResponse(
                "error.html", {"request": request, "message": f"run {run_id} not found"},
                status_code=404,
            )
        points = queries.equity_curve(ch(), run_id)
        # Stringify datetimes for the template's tojson filter (datetime is not JSON-serializable).
        chart_points = [{"ts": str(p["ts"]), "equity": p["equity"]} for p in points]
        return templates.TemplateResponse(
            "run.html",
            {"request": request, "run": detail, "points": chart_points},
        )

    # ---------- JSON twins ----------

    @app.get("/api/overview")
    def api_overview():
        return _fmt_rows(queries.overview(ch()))

    @app.get("/api/strategies/{strategy}")
    def api_strategy(strategy: str):
        return _fmt_rows(queries.strategy_symbols(ch(), strategy))

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str):
        detail = queries.run_detail(ch(), run_id)
        if detail is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in detail.items()}

    @app.get("/api/runs/{run_id}/equity")
    def api_equity(run_id: str):
        return [
            {"ts": str(p["ts"]), "equity": p["equity"]}
            for p in queries.equity_curve(ch(), run_id)
        ]

    return app
```

`src/dashboard/__main__.py`:

```python
"""Boot the dashboard: python -m src.dashboard"""

import uvicorn

from src.dashboard.app import create_app
from src.dashboard.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_app(), host="127.0.0.1", port=settings.dashboard_port,
                log_level="info")


if __name__ == "__main__":
    main()
```

`src/dashboard/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Alpha Research Dashboard{% endblock %}</title>
  <link rel="stylesheet" href="/static/style.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <!-- CDN note: if you run fully offline, download chart.js into static/ and update this tag -->
</head>
<body>
<nav>
  <span class="brand">Alpha Research</span>
  <a href="/" class="active">Backtesting</a>
  <span class="disabled" title="coming in a later iteration">Paper Trading</span>
  <span class="disabled" title="coming in a later iteration">Live Trading</span>
</nav>
<main>
{% block content %}{% endblock %}
</main>
</body>
</html>
```

`src/dashboard/templates/overview.html`:

```html
{% extends "base.html" %}
{% block title %}Backtesting · Alpha Research{% endblock %}
{% block content %}
<h1>Backtesting overview</h1>
<table>
  <thead>
    <tr>
      <th>Strategy</th><th>Window</th><th>Runs</th><th>Avg Sharpe</th>
      <th>Avg Sortino</th><th>Avg MaxDD</th><th>Avg Return</th><th>Avg Win Rate</th>
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}
    <tr class="{{ 'prelim' if row.window == 'PRELIM' }}">
      <td><a href="/strategies/{{ row.strategy }}">{{ row.strategy }}</a></td>
      <td>{{ row.window }}{% if row.window == 'PRELIM' %} <span class="tag">prelim</span>{% endif %}</td>
      <td>{{ row.runs }}</td>
      <td>{{ row.avg_sharpe }}</td>
      <td>{{ row.avg_sortino }}</td>
      <td>{{ row.avg_max_drawdown }}</td>
      <td>{{ row.avg_total_return }}</td>
      <td>{{ row.avg_win_rate }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

`src/dashboard/templates/strategy.html`:

```html
{% extends "base.html" %}
{% block title %}{{ strategy }} · Alpha Research{% endblock %}
{% block content %}
<h1>{{ strategy }}</h1>
<table>
  <thead>
    <tr>
      <th rowspan="2">Symbol</th>
      {% for w in windows %}<th colspan="4" class="{{ 'prelim' if w == 'PRELIM' }}">{{ w }}</th>{% endfor %}
    </tr>
    <tr>
      {% for w in windows %}
      <th>Sharpe</th><th>Return</th><th>MaxDD</th><th>Win Rate</th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for symbol, per_window in symbols.items() %}
    <tr>
      <td>{{ symbol }}</td>
      {% for w in windows %}
        {% set row = per_window.get(w) %}
        {% if row %}
        <td>{{ row.avg_sharpe }}</td><td>{{ row.avg_total_return }}</td>
        <td>{{ row.avg_max_drawdown }}</td><td>{{ row.avg_win_rate }}</td>
        {% else %}
        <td colspan="4" class="muted">—</td>
        {% endif %}
      {% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

`src/dashboard/templates/run.html`:

```html
{% extends "base.html" %}
{% block title %}{{ run.strategy }} {{ run.symbol }} {{ run.window }} · Alpha Research{% endblock %}
{% block content %}
<h1>{{ run.strategy }} · {{ run.symbol }}
  <span class="tag {{ 'prelim' if run.window == 'PRELIM' }}">{{ run.window }}</span></h1>
<div class="cards">
  <div class="card"><span>Total Return</span><b>{{ "%.3f"|format(run.total_return) }}</b></div>
  <div class="card"><span>Sharpe</span><b>{{ "%.3f"|format(run.sharpe) }}</b></div>
  <div class="card"><span>Sortino</span><b>{{ "%.3f"|format(run.sortino) }}</b></div>
  <div class="card"><span>Max Drawdown</span><b>{{ "%.3f"|format(run.max_drawdown) }}</b></div>
  <div class="card"><span>Calmar</span><b>{{ "%.3f"|format(run.calmar) }}</b></div>
  <div class="card"><span>Win Rate</span><b>{{ "%.3f"|format(run.win_rate) }}</b></div>
  <div class="card"><span>Trades</span><b>{{ run.num_trades }}</b></div>
  <div class="card"><span>Run</span><b class="mono">{{ run.run_id }}</b></div>
</div>
<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>
<script>
  const points = {{ points | tojson }};
  new Chart(document.getElementById("equity-chart"), {
    type: "line",
    data: {
      labels: points.map(p => p.ts),
      datasets: [{
        label: "Equity",
        data: points.map(p => p.equity),
        borderColor: "#4fc3f7",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
      }],
    },
    options: {
      animation: false,
      scales: {
        x: { ticks: { maxTicksLimit: 10, color: "#8892a0" }, grid: { color: "#232a35" } },
        y: { ticks: { color: "#8892a0" }, grid: { color: "#232a35" } },
      },
      plugins: { legend: { display: false } },
    },
  });
</script>
{% endblock %}
```

`src/dashboard/templates/error.html`:

```html
{% extends "base.html" %}
{% block title %}Error · Alpha Research{% endblock %}
{% block content %}
<h1>Something went wrong</h1>
<p class="muted">{{ message }}</p>
{% endblock %}
```

`src/dashboard/static/style.css`:

```css
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #0d1117; color: #d7dde5;
  font: 14px/1.5 -apple-system, "SF Mono", "Helvetica Neue", monospace;
}
nav {
  display: flex; gap: 18px; align-items: center;
  padding: 12px 24px; background: #161b22; border-bottom: 1px solid #232a35;
}
nav .brand { font-weight: 700; color: #4fc3f7; margin-right: 12px; }
nav a { color: #d7dde5; text-decoration: none; }
nav a.active { border-bottom: 2px solid #4fc3f7; padding-bottom: 2px; }
nav .disabled { color: #4a5563; cursor: default; }
main { padding: 24px; max-width: 1200px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin-top: 16px; }
th, td { text-align: right; padding: 8px 12px; border-bottom: 1px solid #232a35; }
th:first-child, td:first-child { text-align: left; }
th { color: #8892a0; font-weight: 500; font-size: 12px; text-transform: uppercase; }
a { color: #4fc3f7; }
.tag {
  font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: #232a35; color: #8892a0; vertical-align: middle;
}
.tag.prelim, tr.prelim td { color: #e3b341; }
tr.prelim .tag { background: #3a2f16; color: #e3b341; }
.muted { color: #4a5563; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin: 16px 0 24px; }
.card { background: #161b22; border: 1px solid #232a35; border-radius: 8px; padding: 12px; }
.card span { display: block; color: #8892a0; font-size: 11px; text-transform: uppercase; }
.card b { font-size: 18px; }
.mono { font-size: 12px !important; }
.chart-wrap { background: #161b22; border: 1px solid #232a35; border-radius: 8px; padding: 16px; height: 420px; }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_app.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Full suite + live smoke**

Run: `python -m pytest tests/ -q`
Expected: all 105 tests PASS (96 + 4 queries + 5 app).

Then boot the server and check pages:

```bash
python -m src.dashboard &
sleep 3
curl -s -o /dev/null -w "overview: %{http_code}\n" http://127.0.0.1:8000/
curl -s -o /dev/null -w "strategy: %{http_code}\n" http://127.0.0.1:8000/strategies/mean_reversion
curl -s http://127.0.0.1:8000/api/overview | head -c 200
kill %1
```

Expected: 200s and JSON overview content.

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/app.py src/dashboard/__main__.py src/dashboard/templates/ src/dashboard/static/ tests/test_dashboard_app.py
git commit -m "feat: add dashboard app with backtesting views and JSON API"
```

---

## Self-Review Notes

- **Spec coverage:** config with APP_ENV toggle (Task 1), read-only query layer with all five query functions (Task 1), three HTML pages + four JSON twins (Task 2), error handling (503 JSON/page, 404 on unknown run), downsampling ≤500 (Task 1 query + test), PRELIM tagging (templates), disabled nav slots for future views, 127.0.0.1 binding (`__main__.py`), no new deps beyond fastapi/uvicorn (Task 1 Step 1). All spec sections covered.
- **Type consistency:** `get_settings() -> Settings`, `overview/strategy_symbols/list_runs/run_detail/equity_curve` signatures and dict keys are spelled identically in the interfaces, implementation, tests, and app usage.
- **Test count:** 96 + 4 + 5 = 105 (Task 2 Step 5 states this correctly).
- **Known weak spots accepted up front:** JSON datetime fields are stringified; TestClient hits the real dev DB (consistent with the suite's integration style).
- **Ordering dependency:** Task 2 consumes Task 1's `queries`/`config`. Sequential execution required.
