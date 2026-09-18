import json
from datetime import datetime, timezone

import pytest

from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.paper import runner

STRATEGY = "TEST_mean_reversion"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 15, tzinfo=timezone.utc)


@pytest.fixture
def ch_client():
    c = get_clickhouse_client()
    yield c
    db = database_name()
    for table in ("paper_equity", "paper_trades", "paper_positions"):
        c.command(f"ALTER TABLE {db}.{table} DELETE WHERE strategy LIKE 'TEST%'",
                  settings={"mutations_sync": 1})
    c.command(f"ALTER TABLE {db}.tuned_params DELETE WHERE strategy LIKE 'TEST%'",
              settings={"mutations_sync": 1})


def _snapshot(c, strategy, symbol):
    equity = c.query(
        f"SELECT ts, equity FROM {database_name()}.paper_equity FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} ORDER BY ts",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    trades = c.query(
        f"SELECT count() FROM {database_name()}.paper_trades FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String}",
        parameters={"s": strategy, "y": symbol},
    ).result_rows[0][0]
    return equity, trades


def test_replay_deterministic_and_idempotent(ch_client):
    n1 = runner.replay(ch_client, STRATEGY, "BTC", START, END)
    assert n1 > 0
    first = _snapshot(ch_client, STRATEGY, "BTC")

    n2 = runner.replay(ch_client, STRATEGY, "BTC", START, END)
    second = _snapshot(ch_client, STRATEGY, "BTC")

    assert n1 == n2
    assert first == second  # identical rows, no duplicates


def test_step_forward_is_idempotent(ch_client):
    runner.replay(ch_client, STRATEGY, "ETH", START, END)
    assert runner.step_forward(ch_client, STRATEGY, "ETH") > 0
    assert runner.step_forward(ch_client, STRATEGY, "ETH") == 0


def test_resolve_params_falls_back_to_defaults(ch_client):
    assert runner.resolve_params(ch_client, STRATEGY, "LTC") == dict(mean_reversion.MR_DEFAULTS)


def test_signal_series_uses_time_appropriate_params(ch_client):
    from datetime import timedelta

    from src.backtesting.data import load_ohlcv

    v1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    v2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
    # v1 enters readily; v2's z_entry is unreachable, so it never enters.
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [[STRATEGY, "BTC", json.dumps({"window": 24, "z_entry": -1.0, "z_exit": 0.0}),
          1.0, 1.0, 8, datetime.now(timezone.utc), v1],
         [STRATEGY, "BTC", json.dumps({"window": 24, "z_entry": -99.0, "z_exit": 0.0}),
          1.0, 1.0, 8, datetime.now(timezone.utc), v2]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at", "valid_from"],
    )
    prices = load_ohlcv(ch_client, "BTC", interval="1h", start=v1,
                        end=v1 + timedelta(days=90))["close"]
    entries, _ = runner._signal_series(ch_client, STRATEGY, "BTC", prices)
    assert entries.loc[:v2].any()
    assert not entries.loc[v2:].any()


def test_trailing_return_is_causal(ch_client):
    from datetime import timedelta

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # At as_of, the last known equity is 0.8 and the point 30d earlier is 1.0.
    series = [(base, 1.0), (base + timedelta(days=31), 0.8)]
    assert runner._trailing_return(series, base + timedelta(days=31)) == pytest.approx(-0.2)
    assert runner._trailing_return([(base, 1.0)], base) is None  # no point <= cutoff


def test_replay_writes_control(ch_client):
    runner.replay(ch_client, STRATEGY, "LTC", START, END)
    rows = ch_client.query(
        f"SELECT enabled, trailing_return FROM {database_name()}.paper_controls FINAL "
        "WHERE strategy = {s:String} AND symbol = 'LTC'",
        parameters={"s": STRATEGY},
    ).result_rows
    assert rows and rows[0][0] == 1  # <30d history → enabled by default


def test_resolve_params_uses_tuned(ch_client):
    params = {"window": 36, "z_entry": -2.2, "z_exit": 0.2}
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [[STRATEGY, "SOL", json.dumps(params), 1.0, 1.0, 8, datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
    assert runner.resolve_params(ch_client, STRATEGY, "SOL") == params
