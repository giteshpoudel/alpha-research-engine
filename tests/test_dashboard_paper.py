from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.dashboard import queries
from src.dashboard.app import create_app
from src.ingestion.schemas import database_name, get_clickhouse_client

STRATEGY = "TEST_mean_reversion"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture
def ch_client():
    c = get_clickhouse_client()
    yield c
    db = database_name()
    for table in ("paper_equity", "paper_trades", "paper_positions"):
        c.command(f"ALTER TABLE {db}.{table} DELETE WHERE strategy LIKE 'TEST%'",
                  settings={"mutations_sync": 1})


def _seed(c):
    now = datetime.now(timezone.utc)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    later = ts.replace(hour=1)
    c.insert(
        f"{database_name()}.paper_equity",
        [[STRATEGY, "BTC", ts, 1.0, 1.0, 0.0, 100.0, "flat", now],
         [STRATEGY, "BTC", later, 1.05, 0.0, 1.05, 105.0, "long", now]],
        column_names=["strategy", "symbol", "ts", "equity", "cash", "position_value",
                      "mark_price", "status", "created_at"],
    )
    c.insert(
        f"{database_name()}.paper_positions",
        [[STRATEGY, "BTC", "long", 100.0, later, 0.0105, 0.0, 1.0, later, now]],
        column_names=["strategy", "symbol", "status", "entry_price", "entry_ts", "size",
                      "cash", "cost_basis", "last_bar_ts", "updated_at"],
    )
    c.insert(
        f"{database_name()}.paper_trades",
        [["abc123", STRATEGY, "BTC", "entry", ts, 100.0, 0.0105, 1.0, 0.001, 0.0,
          "entry", now]],
        column_names=["trade_id", "strategy", "symbol", "side", "ts", "price", "size",
                      "notional", "fee", "realized_pnl", "reason", "created_at"],
    )


def test_paper_query_shapes(ch_client):
    _seed(ch_client)
    data = queries.paper_summary(ch_client, strategy=STRATEGY)
    assert set(data) == {"summary", "symbols"}
    assert data["symbols"] and data["symbols"][0]["symbol"] == "BTC"
    assert set(data["summary"]) == {"total_return", "sharpe", "total_trades",
                                    "win_rate", "num_sleeves", "last_bar_ts"}
    assert {"enabled", "trailing_return"} <= set(data["symbols"][0])

    detail = queries.paper_symbol(ch_client, "BTC", strategy=STRATEGY)
    assert detail is not None and detail["status"] == "long"
    assert {"enabled", "trailing_return"} <= set(detail)
    assert queries.paper_symbol(ch_client, "NOPE", strategy=STRATEGY) is None

    curve = queries.paper_equity_curve(ch_client, "BTC", strategy=STRATEGY)
    assert len(curve) == 2 and {"ts", "equity"} == set(curve[0])

    trades = queries.paper_trades(ch_client, "BTC", strategy=STRATEGY)
    assert trades and trades[0]["side"] == "entry"


def test_paper_pages_and_api(client):
    assert client.get("/paper").status_code == 200
    api = client.get("/api/paper").json()
    assert set(api) == {"summary", "symbols"}
    assert client.get("/paper/ZZZ").status_code == 404
    assert client.get("/api/paper/ZZZ").status_code == 404
