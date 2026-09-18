from datetime import datetime, timezone

import pytest

from src.backtesting.runner import compute_run_id, main, run_backtest
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_run_id_deterministic_and_param_sensitive():
    base = compute_run_id("mean_reversion", "BTC", "1h", '{"window": 24}',
                          "IS", datetime(2022, 1, 1, tzinfo=timezone.utc),
                          datetime(2024, 12, 31, 23, tzinfo=timezone.utc))
    again = compute_run_id("mean_reversion", "BTC", "1h", '{"window": 24}',
                           "IS", datetime(2022, 1, 1, tzinfo=timezone.utc),
                           datetime(2024, 12, 31, 23, tzinfo=timezone.utc))
    other = compute_run_id("mean_reversion", "BTC", "1h", '{"window": 48}',
                           "IS", datetime(2022, 1, 1, tzinfo=timezone.utc),
                           datetime(2024, 12, 31, 23, tzinfo=timezone.utc))
    assert base == again
    assert base != other
    assert len(base) == 16


@pytest.fixture
def ch_client():
    return get_clickhouse_client()


def test_run_backtest_stores_and_collapses(ch_client):
    run_id = None
    try:
        # small real IS window via a thin wrapper: run on BTC IS 2022-01 only
        import src.backtesting.runner as runner
        original_bounds = runner.WINDOWS["IS"]
        runner.WINDOWS["IS"] = (
            original_bounds[0],
            __import__("datetime").datetime(2022, 1, 31, 23, tzinfo=timezone.utc),
        )
        run_id = run_backtest(ch_client, "TEST_mean_reversion", "BTC", "IS", fee=0.0)
        run_id2 = run_backtest(ch_client, "TEST_mean_reversion", "BTC", "IS", fee=0.0)
        assert run_id == run_id2
        rows = ch_client.query(
            f"SELECT count(), any(sharpe), any(num_trades) FROM {database_name()}.backtest_runs FINAL "
            "WHERE run_id = {r:String}", parameters={"r": run_id},
        ).result_rows
        assert rows[0][0] == 1  # collapsed, not duplicated
        eq = ch_client.query(
            f"SELECT count() FROM {database_name()}.backtest_equity FINAL "
            "WHERE run_id = {r:String}", parameters={"r": run_id},
        ).result_rows
        assert eq[0][0] > 700  # one month of hourly equity points
    finally:
        runner.WINDOWS["IS"] = original_bounds
        if run_id:
            ch_client.command(
                f"ALTER TABLE {database_name()}.backtest_runs DELETE WHERE run_id = {{r:String}}",
                parameters={"r": run_id}, settings={"mutations_sync": 1})
            ch_client.command(
                f"ALTER TABLE {database_name()}.backtest_equity DELETE WHERE run_id = {{r:String}}",
                parameters={"r": run_id}, settings={"mutations_sync": 1})


def test_sentiment_momentum_runs_on_oos_window(ch_client):
    run_id = None
    try:
        run_id = run_backtest(ch_client, "TEST_sentiment_momentum", "BTC", "OOS", fee=0.0)
        rows = ch_client.query(
            f"SELECT window, count() FROM {database_name()}.backtest_runs FINAL "
            "WHERE run_id = {r:String} GROUP BY window",
            parameters={"r": run_id},
        ).result_rows
        assert [r[0] for r in rows] == ["OOS"]
    finally:
        if run_id:
            ch_client.command(
                f"ALTER TABLE {database_name()}.backtest_runs DELETE WHERE run_id = {{r:String}}",
                parameters={"r": run_id}, settings={"mutations_sync": 1})
            ch_client.command(
                f"ALTER TABLE {database_name()}.backtest_equity DELETE WHERE run_id = {{r:String}}",
                parameters={"r": run_id}, settings={"mutations_sync": 1})


def test_main_runs_requested_strategy_only(monkeypatch, ch_client):
    seen = []
    monkeypatch.setattr("src.backtesting.runner.run_backtest",
                        lambda ch, strategy, symbol, window, fee=0.001: seen.append((strategy, symbol, window)) or "x")
    monkeypatch.setattr("src.backtesting.runner.get_clickhouse_client", lambda: ch_client)
    main(["--strategy", "mean_reversion", "--symbols", "BTC,ETH", "--window", "IS"])
    assert seen == [("mean_reversion", "BTC", "IS"), ("mean_reversion", "ETH", "IS")]
