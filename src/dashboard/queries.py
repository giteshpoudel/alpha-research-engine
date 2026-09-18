"""Read-only ClickHouse queries for the dashboard.

Windows are always separate rows/columns — nothing here ever averages
across IS/OOS (guardrail #1 applies to the UI too).

Every function takes an optional ``db`` so the dashboard can honor the
``APP_ENV`` toggle (dev → ``CLICKHOUSE_DB``, prod → ``PROD_CLICKHOUSE_DB``);
when omitted it falls back to the shared ``database_name()``.
"""

from __future__ import annotations

import math

import pandas as pd

from src.ingestion.schemas import database_name

# Fixed hourly observations per year for paper-trading Sharpe annualization.
_PERIODS_PER_YEAR = 8760


def _db(db: str | None) -> str:
    return db or database_name()


def _strategy_clause(strategy: str | None) -> tuple[str, dict]:
    """``None`` excludes test rows; an explicit strategy matches exactly."""
    if strategy is None:
        return "NOT startsWith(strategy, 'TEST_')", {}
    return "strategy = {strategy:String}", {"strategy": strategy}


def overview(client, db: str | None = None) -> list[dict]:
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
        FROM {_db(db)}.backtest_runs FINAL
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


def strategy_symbols(client, strategy: str, db: str | None = None) -> list[dict]:
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
        FROM {_db(db)}.backtest_runs FINAL
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


def list_runs(client, strategy: str, symbol: str, db: str | None = None) -> list[dict]:
    rows = client.query(
        f"""
        SELECT run_id, window, start_ts, end_ts, sharpe, total_return
        FROM {_db(db)}.backtest_runs FINAL
        WHERE strategy = {{strategy:String}} AND symbol = {{symbol:String}}
        ORDER BY window, start_ts
        """,
        parameters={"strategy": strategy, "symbol": symbol},
    ).result_rows
    return [
        dict(zip(("run_id", "window", "start_ts", "end_ts", "sharpe", "total_return"), row))
        for row in rows
    ]


def run_detail(client, run_id: str, db: str | None = None) -> dict | None:
    rows = client.query(
        f"""
        SELECT run_id, strategy, symbol, interval, params_json, window,
               start_ts, end_ts, total_return, sharpe, sortino,
               max_drawdown, calmar, win_rate, num_trades, created_at
        FROM {_db(db)}.backtest_runs FINAL
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


def equity_curve(client, run_id: str, max_points: int = 500,
                 db: str | None = None) -> list[dict]:
    count = client.query(
        f"SELECT count() FROM {_db(db)}.backtest_equity FINAL "
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
            FROM {_db(db)}.backtest_equity FINAL
            WHERE run_id = {{run_id:String}}
        )
        WHERE rn % {{stride:UInt32}} = 0
        ORDER BY ts
        """,
        parameters={"run_id": run_id, "stride": stride},
    ).result_rows
    return [{"ts": row[0], "equity": float(row[1])} for row in rows]


# ---------- Paper trading ----------


def paper_summary(client, db: str | None = None,
                  strategy: str | None = None) -> dict:
    """Per-symbol paper sleeve rows plus equal-weight aggregate KPIs."""
    dbn = _db(db)
    clause, params = _strategy_clause(strategy)

    equity_rows = client.query(
        f"""
        SELECT
            symbol,
            argMin(equity, ts) AS first_equity,
            argMax(equity, ts) AS last_equity,
            argMax(mark_price, ts) AS mark_price,
            argMax(status, ts) AS status,
            max(ts) AS last_bar_ts
        FROM {dbn}.paper_equity FINAL
        WHERE {clause}
        GROUP BY symbol
        ORDER BY symbol
        """,
        parameters=params,
    ).result_rows
    positions = {
        r[0]: r for r in client.query(
            f"SELECT symbol, status, entry_price, entry_ts "
            f"FROM {dbn}.paper_positions FINAL WHERE {clause}",
            parameters=params,
        ).result_rows
    }
    trade_stats = {
        r[0]: (int(r[1]), int(r[2]), int(r[3])) for r in client.query(
            f"""
            SELECT symbol, count() AS n,
                   countIf(side = 'exit' AND realized_pnl > 0) AS wins,
                   countIf(side = 'exit') AS exits
            FROM {dbn}.paper_trades FINAL WHERE {clause}
            GROUP BY symbol
            """,
            parameters=params,
        ).result_rows
    }
    controls = {
        r[0]: (bool(r[1]), float(r[2])) for r in client.query(
            f"SELECT symbol, enabled, trailing_return "
            f"FROM {dbn}.paper_controls FINAL WHERE {clause}",
            parameters=params,
        ).result_rows
    }

    symbols: list[dict] = []
    for symbol, first_equity, last_equity, mark_price, status, last_bar_ts in equity_rows:
        pos = positions.get(symbol)
        num_trades, wins, exits = trade_stats.get(symbol, (0, 0, 0))
        enabled, trailing = controls.get(symbol, (True, 0.0))
        first_equity = float(first_equity)
        last_equity = float(last_equity)
        symbols.append({
            "symbol": symbol,
            "status": status,
            "entry_price": float(pos[2]) if pos else 0.0,
            "entry_ts": pos[3] if pos else None,
            "mark_price": float(mark_price),
            "equity": last_equity,
            "total_return": (last_equity / first_equity - 1.0) if first_equity else 0.0,
            "pnl": last_equity - 1.0,
            "num_trades": num_trades,
            "win_rate": (wins / exits) if exits else 0.0,
            "enabled": enabled,
            "trailing_return": trailing,
            "last_bar_ts": last_bar_ts,
        })

    agg = client.query(
        f"SELECT ts, avg(equity) AS equity FROM {dbn}.paper_equity FINAL "
        f"WHERE {clause} GROUP BY ts ORDER BY ts",
        parameters=params,
    ).result_rows
    total_return = sharpe = 0.0
    last_bar_ts = None
    if agg:
        series = pd.Series([float(e) for _, e in agg])
        total_return = float(series.iloc[-1]) / float(series.iloc[0]) - 1.0
        rets = series.pct_change().dropna()
        sd = float(rets.std(ddof=0))
        if rets.size and math.isfinite(sd) and sd > 0:
            sharpe = float(rets.mean()) / sd * math.sqrt(_PERIODS_PER_YEAR)
        last_bar_ts = agg[-1][0]

    total_trades = sum(v[0] for v in trade_stats.values())
    exits = sum(v[2] for v in trade_stats.values())
    wins = sum(v[1] for v in trade_stats.values())
    summary = {
        "total_return": total_return,
        "sharpe": sharpe,
        "total_trades": total_trades,
        "win_rate": (wins / exits) if exits else 0.0,
        "num_sleeves": len(symbols),
        "last_bar_ts": last_bar_ts,
    }
    return {"summary": summary, "symbols": symbols}


def paper_symbol(client, symbol: str, db: str | None = None,
                 strategy: str | None = None) -> dict | None:
    dbn = _db(db)
    clause, params = _strategy_clause(strategy)
    params = {**params, "symbol": symbol}
    rows = client.query(
        f"""
        SELECT argMin(equity, ts) AS first_equity, argMax(equity, ts) AS last_equity,
               argMax(mark_price, ts) AS mark_price, argMax(status, ts) AS status,
               max(ts) AS last_bar_ts
        FROM {dbn}.paper_equity FINAL
        WHERE {clause} AND symbol = {{symbol:String}}
        HAVING count() > 0
        """,
        parameters=params,
    ).result_rows
    if not rows or rows[0][0] is None:
        return None
    first_equity, last_equity, mark_price, status, last_bar_ts = rows[0]
    pos = client.query(
        f"SELECT entry_price, entry_ts FROM {dbn}.paper_positions FINAL "
        f"WHERE {clause} AND symbol = {{symbol:String}} LIMIT 1",
        parameters=params,
    ).result_rows
    ctrl = client.query(
        f"SELECT enabled, trailing_return FROM {dbn}.paper_controls FINAL "
        f"WHERE {clause} AND symbol = {{symbol:String}} LIMIT 1",
        parameters=params,
    ).result_rows
    first_equity = float(first_equity)
    last_equity = float(last_equity)
    return {
        "symbol": symbol,
        "status": status,
        "entry_price": float(pos[0][0]) if pos else 0.0,
        "entry_ts": pos[0][1] if pos else None,
        "mark_price": float(mark_price),
        "equity": last_equity,
        "total_return": (last_equity / first_equity - 1.0) if first_equity else 0.0,
        "pnl": last_equity - 1.0,
        "enabled": bool(ctrl[0][0]) if ctrl else True,
        "trailing_return": float(ctrl[0][1]) if ctrl else 0.0,
        "last_bar_ts": last_bar_ts,
    }


def paper_equity_curve(client, symbol: str, db: str | None = None,
                       strategy: str | None = None, max_points: int = 500) -> list[dict]:
    dbn = _db(db)
    clause, params = _strategy_clause(strategy)
    params = {**params, "symbol": symbol}
    count = client.query(
        f"SELECT count() FROM {dbn}.paper_equity FINAL "
        f"WHERE {clause} AND symbol = {{symbol:String}}",
        parameters=params,
    ).result_rows[0][0]
    if count == 0:
        return []
    stride = max(1, (count + max_points - 1) // max_points)
    rows = client.query(
        f"""
        SELECT ts, equity FROM (
            SELECT ts, equity, row_number() OVER (ORDER BY ts) - 1 AS rn
            FROM {dbn}.paper_equity FINAL
            WHERE {clause} AND symbol = {{symbol:String}}
        )
        WHERE rn % {{stride:UInt32}} = 0
        ORDER BY ts
        """,
        parameters={**params, "stride": stride},
    ).result_rows
    return [{"ts": row[0], "equity": float(row[1])} for row in rows]


def paper_trades(client, symbol: str, db: str | None = None,
                 strategy: str | None = None, limit: int = 50) -> list[dict]:
    dbn = _db(db)
    clause, params = _strategy_clause(strategy)
    rows = client.query(
        f"""
        SELECT side, ts, price, size, notional, fee, realized_pnl
        FROM {dbn}.paper_trades FINAL
        WHERE {clause} AND symbol = {{symbol:String}}
        ORDER BY ts DESC
        LIMIT {{limit:UInt32}}
        """,
        parameters={**params, "symbol": symbol, "limit": limit},
    ).result_rows
    return [
        dict(zip(("side", "ts", "price", "size", "notional", "fee", "realized_pnl"), row))
        for row in rows
    ]


# ---------- Signal evaluation ----------


def signal_results(client, db: str | None = None) -> list[dict]:
    rows = client.query(
        f"""
        SELECT bucket_size, feature, horizon, method, value, n, tstat,
               insufficient, detail, computed_at
        FROM {_db(db)}.signal_eval_results FINAL
        ORDER BY method, bucket_size, feature, horizon
        """
    ).result_rows
    return [
        dict(zip(("bucket_size", "feature", "horizon", "method", "value", "n",
                  "tstat", "insufficient", "detail", "computed_at"), row))
        for row in rows
    ]
