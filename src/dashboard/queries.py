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
