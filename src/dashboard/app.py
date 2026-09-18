"""FastAPI dashboard: backtesting + paper-trading views (HTML pages + JSON twins)."""

from __future__ import annotations

import json

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


def _conn():
    """ClickHouse client plus the DB name for the active APP_ENV."""
    s = get_settings()
    client = clickhouse_connect.get_client(
        host=s.clickhouse_host, port=s.clickhouse_port,
        username=s.clickhouse_user, password=s.clickhouse_password,
    )
    return client, s.clickhouse_db


def _fmt(value):
    return round(float(value), 3) if value is not None else None


def _fmt_rows(rows):
    return [{k: (_fmt(v) if isinstance(v, float) else v) for k, v in row.items()} for row in rows]


def _fmt_dict(row):
    return {k: (_fmt(v) if isinstance(v, float) else v) for k, v in row.items()}


def _signal_view(rows):
    """Split stored evaluation rows into coverage and statistics, parsing detail JSON."""
    coverage, stats = [], []
    for row in rows:
        item = {**row, "detail": json.loads(row["detail"]) if row["detail"] else {}}
        (coverage if row["method"] == "coverage" else stats).append(item)
    return coverage, stats


def create_app() -> FastAPI:
    app = FastAPI(title="Alpha Research Dashboard")
    app.mount("/static", StaticFiles(directory=_PKG_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=_PKG_DIR / "templates")

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
        client, db = _conn()
        rows = _fmt_rows(queries.overview(client, db=db))
        return templates.TemplateResponse(
            request, "overview.html", {"request": request, "rows": rows}
        )

    @app.get("/strategies/{strategy}", response_class=HTMLResponse)
    def strategy_page(request: Request, strategy: str):
        client, db = _conn()
        rows = _fmt_rows(queries.strategy_symbols(client, strategy, db=db))
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
        client, db = _conn()
        detail = queries.run_detail(client, run_id, db=db)
        if detail is None:
            return templates.TemplateResponse(
                request, "error.html", {"request": request, "message": f"run {run_id} not found"},
                status_code=404,
            )
        points = queries.equity_curve(client, run_id, db=db)
        # Stringify datetimes for the template's tojson filter (datetime is not JSON-serializable).
        chart_points = [{"ts": str(p["ts"]), "equity": p["equity"]} for p in points]
        return templates.TemplateResponse(
            request, "run.html",
            {"request": request, "run": detail, "points": chart_points},
        )

    @app.get("/paper", response_class=HTMLResponse)
    def paper_page(request: Request):
        client, db = _conn()
        data = queries.paper_summary(client, db=db)
        return templates.TemplateResponse(
            request, "paper.html",
            {"request": request, "active": "paper",
             "summary": _fmt_dict(data["summary"]), "symbols": _fmt_rows(data["symbols"])},
        )

    @app.get("/paper/{symbol}", response_class=HTMLResponse)
    def paper_symbol_page(request: Request, symbol: str):
        client, db = _conn()
        detail = queries.paper_symbol(client, symbol, db=db)
        if detail is None:
            return templates.TemplateResponse(
                request, "error.html", {"request": request, "message": f"paper sleeve {symbol} not found"},
                status_code=404,
            )
        points = queries.paper_equity_curve(client, symbol, db=db)
        trades = queries.paper_trades(client, symbol, db=db)
        chart_points = [{"ts": str(p["ts"]), "equity": p["equity"]} for p in points]
        return templates.TemplateResponse(
            request, "paper_symbol.html",
            {"request": request, "active": "paper", "sleeve": _fmt_dict(detail),
             "points": chart_points, "trades": _fmt_rows(trades)},
        )

    # ---------- JSON twins ----------

    @app.get("/api/overview")
    def api_overview():
        client, db = _conn()
        return _fmt_rows(queries.overview(client, db=db))

    @app.get("/api/strategies/{strategy}")
    def api_strategy(strategy: str):
        client, db = _conn()
        return _fmt_rows(queries.strategy_symbols(client, strategy, db=db))

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str):
        client, db = _conn()
        detail = queries.run_detail(client, run_id, db=db)
        if detail is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in detail.items()}

    @app.get("/api/runs/{run_id}/equity")
    def api_equity(run_id: str):
        client, db = _conn()
        return [
            {"ts": str(p["ts"]), "equity": p["equity"]}
            for p in queries.equity_curve(client, run_id, db=db)
        ]

    @app.get("/api/paper")
    def api_paper():
        client, db = _conn()
        data = queries.paper_summary(client, db=db)
        return {
            "summary": {k: (str(v) if hasattr(v, "isoformat") else v)
                        for k, v in data["summary"].items()},
            "symbols": [{k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}
                        for row in data["symbols"]],
        }

    @app.get("/api/paper/{symbol}")
    def api_paper_symbol(symbol: str):
        client, db = _conn()
        detail = queries.paper_symbol(client, symbol, db=db)
        if detail is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        trades = queries.paper_trades(client, symbol, db=db)
        return {
            **{k: (str(v) if hasattr(v, "isoformat") else v) for k, v in detail.items()},
            "recent_trades": [
                {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in t.items()}
                for t in trades
            ],
        }

    @app.get("/api/paper/{symbol}/equity")
    def api_paper_equity(symbol: str):
        client, db = _conn()
        return [
            {"ts": str(p["ts"]), "equity": p["equity"]}
            for p in queries.paper_equity_curve(client, symbol, db=db)
        ]

    @app.get("/signal", response_class=HTMLResponse)
    def signal_page(request: Request):
        client, db = _conn()
        coverage, stats = _signal_view(queries.signal_results(client, db=db))
        return templates.TemplateResponse(
            request, "signal.html",
            {"request": request, "active": "signal",
             "coverage": coverage, "stats": stats, "has_data": bool(coverage or stats)},
        )

    @app.get("/api/signal")
    def api_signal():
        client, db = _conn()
        return [
            {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}
            for row in queries.signal_results(client, db=db)
        ]

    return app
