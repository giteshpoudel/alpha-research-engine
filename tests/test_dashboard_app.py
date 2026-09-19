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


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert {"ohlcv_latest", "sentiment_latest", "paper_latest",
            "signal_eval_latest", "report_latest"} <= set(body["freshness"])


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
