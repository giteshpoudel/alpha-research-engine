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
            request, "error.html", {"request": request, "message": str(exc)}, status_code=503
        )

    # ---------- HTML pages ----------

    @app.get("/", response_class=HTMLResponse)
    def overview_page(request: Request):
        rows = _fmt_rows(queries.overview(ch()))
        return templates.TemplateResponse(
            request, "overview.html", {"request": request, "rows": rows}
        )

    @app.get("/strategies/{strategy}", response_class=HTMLResponse)
    def strategy_page(request: Request, strategy: str):
        rows = _fmt_rows(queries.strategy_symbols(ch(), strategy))
        symbols: dict[str, dict] = {}
        for row in rows:
            symbols.setdefault(row["symbol"], {})[row["window"]] = row
        return templates.TemplateResponse(
            request, "strategy.html",
            {"request": request, "strategy": strategy,
             "symbols": symbols, "windows": WINDOW_ORDER},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        detail = queries.run_detail(ch(), run_id)
        if detail is None:
            return templates.TemplateResponse(
                request, "error.html", {"request": request, "message": f"run {run_id} not found"},
                status_code=404,
            )
        points = queries.equity_curve(ch(), run_id)
        # Stringify datetimes for the template's tojson filter (datetime is not JSON-serializable).
        chart_points = [{"ts": str(p["ts"]), "equity": p["equity"]} for p in points]
        return templates.TemplateResponse(
            request, "run.html",
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
