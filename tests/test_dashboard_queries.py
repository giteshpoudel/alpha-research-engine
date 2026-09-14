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
