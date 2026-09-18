"""Deterministic data collectors for the research agents (no LLM).

Everything the agents reason about is computed here, so the LLM surface
stays small and every number is testable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.backtesting.data import load_ohlcv
from src.backtesting.strategies.mean_reversion import MR_DEFAULTS
from src.ingestion.schemas import database_name
from src.ingestion.tickers import TICKER_ALIASES
from src.meta_learning.params import get_tuned_params

_LOOKBACK_DAYS = 30


def spike_flags(velocity: float, engagement_ratio: float) -> list[str]:
    flags = []
    if velocity > 2.0:
        flags.append("mention_spike")
    if engagement_ratio > 2.0:
        flags.append("engagement_spike")
    return flags


def _returns_30d(ch_client, symbol: str) -> pd.Series:
    start = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
    close = load_ohlcv(ch_client, symbol, start=start)["close"]
    return close.pct_change().dropna()


def _params_for_symbol(ch_client, symbol: str, strategy: str = "mean_reversion") -> dict:
    """Tuned params for the symbol, falling back to strategy defaults."""
    tuned = get_tuned_params(ch_client, strategy, symbol)
    return dict(tuned) if tuned else dict(MR_DEFAULTS)


def _z_state(close: pd.Series, params: dict | None = None) -> tuple[str, float]:
    params = params or MR_DEFAULTS
    window = params["window"]
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = float((close.iloc[-1] - ma.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] else 0.0
    return ("in" if z <= params["z_entry"] else "out"), z


def collect_risk_data(ch_client) -> dict:
    returns = {}
    for symbol in TICKER_ALIASES:
        try:
            returns[symbol] = _returns_30d(ch_client, symbol)
        except ValueError:
            continue
    frame = pd.DataFrame(returns).dropna()
    corr = frame.corr()
    pairs = []
    symbols = list(corr.columns)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pairs.append({"pair": f"{symbols[i]} vs {symbols[j]}",
                          "corr": float(corr.iloc[i, j])})
    pairs.sort(key=lambda p: -p["corr"])

    vols = [{"symbol": s, "ann_vol": float(frame[s].std() * np.sqrt(24 * 365))}
            for s in symbols]
    vols.sort(key=lambda v: -v["ann_vol"])

    signal_states = []
    for symbol in TICKER_ALIASES:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=7)
            close = load_ohlcv(ch_client, symbol, start=start)["close"]
            state, z = _z_state(close, _params_for_symbol(ch_client, symbol))
            signal_states.append({"symbol": symbol, "state": state, "z": round(z, 3)})
        except ValueError:
            signal_states.append({"symbol": symbol, "state": "out", "z": 0.0})

    rows = ch_client.query(
        f"""
        SELECT symbol, equity FROM (
            SELECT r.symbol AS symbol, e.ts AS ts, e.equity AS equity,
                   row_number() OVER (PARTITION BY r.symbol ORDER BY e.ts DESC) AS rn
            FROM {database_name()}.backtest_equity AS e FINAL
            JOIN {database_name()}.backtest_runs AS r FINAL ON e.run_id = r.run_id
            WHERE r.strategy = 'mean_reversion' AND r.window = 'OOS'
              AND r.params_json LIKE '%"tuned"%' AND NOT startsWith(r.symbol, 'TEST')
        ) WHERE rn = 1
        """
    ).result_rows
    tuned_drawdowns = []
    for symbol, latest_equity in rows:
        dd_rows = ch_client.query(
            f"""
            SELECT symbol, min(dd) AS max_dd FROM (
                SELECT r.symbol AS symbol, e.ts AS ts, e.equity AS equity,
                       max(e.equity) OVER (PARTITION BY r.symbol ORDER BY e.ts
                                           ROWS UNBOUNDED PRECEDING) AS peak,
                       e.equity / peak - 1.0 AS dd
                FROM {database_name()}.backtest_equity AS e FINAL
                JOIN {database_name()}.backtest_runs AS r FINAL ON e.run_id = r.run_id
                WHERE r.strategy = 'mean_reversion' AND r.window = 'OOS'
                  AND r.params_json LIKE '%"tuned"%' AND NOT startsWith(r.symbol, 'TEST')
            ) GROUP BY symbol
            """
        ).result_rows
        current_dd = {s: float(d) for s, d in dd_rows}.get(symbol, 0.0)
        tuned_drawdowns.append({"symbol": symbol, "current_dd": round(current_dd, 4)})

    return {
        "top_correlated": pairs[:5],
        "top_volatile": vols[:5],
        "signal_states": signal_states,
        "tuned_drawdowns": tuned_drawdowns,
    }


def collect_macro_data(ch_client) -> dict:
    rows = ch_client.query(
        f"""
        SELECT ticker, weighted_score, velocity, engagement_ratio FROM (
            SELECT ticker, bucket_start, weighted_score, velocity, engagement_ratio,
                   row_number() OVER (PARTITION BY ticker ORDER BY bucket_start DESC) AS rn
            FROM {database_name()}.sentiment_metrics FINAL
            WHERE bucket_size = '1h'
        ) WHERE rn = 1
        """
    ).result_rows
    sentiment = [
        {"ticker": t, "weighted_score": round(float(s), 3),
         "velocity": round(float(v), 2), "engagement_ratio": round(float(e), 2),
         "flags": spike_flags(float(v), float(e))}
        for t, s, v, e in rows
    ]

    price_changes = []
    for symbol in TICKER_ALIASES:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=2)
            close = load_ohlcv(ch_client, symbol, start=start)["close"]
            day_ago = close.index[-1] - pd.Timedelta(hours=24)
            past = close.loc[:day_ago]
            if len(past) > 0:
                change = float(close.iloc[-1] / past.iloc[-1] - 1.0)
                price_changes.append({"symbol": symbol, "pct_24h": round(change * 100, 2)})
        except ValueError:
            continue

    funding_rows = ch_client.query(
        f"""
        SELECT symbol, funding_rate FROM (
            SELECT symbol, ts, funding_rate,
                   row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
            FROM {database_name()}.funding_rates FINAL
        ) WHERE rn = 1 ORDER BY abs(funding_rate) DESC LIMIT 3
        """
    ).result_rows
    funding_extremes = [{"symbol": s, "rate": float(r)} for s, r in funding_rows]

    post_rows = ch_client.query(
        f"""
        SELECT post_id, source, tickers, likes, sentiment_score, text
        FROM {database_name()}.sentiment_posts FINAL
        WHERE published_at >= now64(3) - INTERVAL 2 DAY
          AND sentiment_score IS NOT NULL
        ORDER BY likes DESC LIMIT 3
        """
    ).result_rows
    top_posts = [
        {"post_id": pid, "source": src, "tickers": list(tks), "likes": int(lks),
         "score": round(float(sc), 3),
         "text_excerpt": (txt[:197] + "...") if len(txt) > 200 else txt}
        for pid, src, tks, lks, sc, txt in post_rows
    ]

    return {
        "sentiment": sentiment,
        "price_changes_24h": price_changes,
        "funding_extremes": funding_extremes,
        "top_posts": top_posts,
    }


def collect_portfolio_data(ch_client) -> dict:
    """Paper-trading / feedback-loop state: sleeves, halts, param versions."""
    db = database_name()
    sleeve_rows = ch_client.query(
        f"""
        SELECT symbol, argMax(equity, ts) AS equity, max(ts) AS last_bar_ts
        FROM {db}.paper_equity FINAL
        WHERE NOT startsWith(strategy, 'TEST_')
        GROUP BY symbol ORDER BY symbol
        """
    ).result_rows
    controls = {
        r[0]: (bool(r[1]), float(r[2])) for r in ch_client.query(
            f"SELECT symbol, enabled, trailing_return FROM {db}.paper_controls FINAL "
            "WHERE NOT startsWith(strategy, 'TEST_')"
        ).result_rows
    }
    sleeves = []
    for symbol, equity, _ in sleeve_rows:
        enabled, trailing = controls.get(symbol, (True, 0.0))
        sleeves.append({
            "symbol": symbol,
            "equity": round(float(equity), 4),
            "total_return": round(float(equity) - 1.0, 4),
            "trailing_return": round(trailing, 4),
            "enabled": enabled,
        })
    equal_weight = (sum(s["equity"] for s in sleeves) / len(sleeves)) if sleeves else 0.0
    params = ch_client.query(
        f"""
        SELECT symbol, max(valid_from) AS valid_from
        FROM {db}.tuned_params FINAL
        WHERE strategy = 'mean_reversion' AND NOT startsWith(symbol, 'TEST')
        GROUP BY symbol ORDER BY symbol
        """
    ).result_rows
    as_of = max((r[2] for r in sleeve_rows), default=None)
    return {
        "as_of": str(as_of) if as_of else None,
        "equal_weight_equity": round(equal_weight, 4),
        "equal_weight_return": round(equal_weight - 1.0, 4) if sleeves else 0.0,
        "sleeves": sleeves,
        "halted": [s["symbol"] for s in sleeves if not s["enabled"]],
        "param_valid_from": {s: str(v) for s, v in params},
    }


def collect_signal_data(ch_client) -> dict:
    """Strongest statistically significant sentiment predictive results."""
    rows = ch_client.query(
        f"""
        SELECT bucket_size, feature, horizon, value, n, tstat
        FROM {database_name()}.signal_eval_results FINAL
        WHERE method = 'ic_pooled' AND insufficient = 0
        ORDER BY abs(value) DESC LIMIT 5
        """
    ).result_rows
    return {"top_ic": [
        {"bucket": b, "feature": f, "horizon": h, "ic": round(float(v), 4),
         "n": int(n), "t": round(float(t), 2)}
        for b, f, h, v, n, t in rows
    ]}
