"""Optuna walk-forward parameter tuning — IS window only (guardrail #1).

Never imports or references OOS/PRELIM windows. The tripwire assertion in
walk_forward_folds raises if any fold bound escapes the IS window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import optuna

from src.backtesting import fees
from src.backtesting.data import load_funding, load_ohlcv
from src.backtesting.engine import (
    run_funding_backtest,
    run_maker_backtest,
    run_signal_backtest,
)
from src.backtesting.strategies import registry
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.universe import universe

# funding_arb retired 2026-09 (fee drag > carry); it stays in the engine but is
# not a tunable signal strategy. All registry strategies are tuned.
TUNABLE_STRATEGIES = registry.STRATEGY_NAMES
DEFAULT_FEE = 0.001
_TRAIN_MONTHS, _VAL_MONTHS, _STEP_MONTHS = 12, 3, 3
_IS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
_IS_END_EXCLUSIVE = datetime(2025, 1, 1, tzinfo=timezone.utc)  # IS end candle + 1h

# Intraday (non-1h) tuning uses a shorter, later window because 5m history only
# starts mid-2025 and the default 12+3 folds wouldn't fit.
_INTRADAY_IS_START = datetime(2025, 6, 1, tzinfo=timezone.utc)
_INTRADAY_IS_END = datetime(2026, 6, 1, tzinfo=timezone.utc)
_INTRADAY_TRAIN, _INTRADAY_VAL, _INTRADAY_STEP = 3, 1, 1

optuna.logging.set_verbosity(optuna.logging.WARNING)


def add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    return dt.replace(year=dt.year + month // 12, month=month % 12 + 1)


# Backwards-compatible private alias.
_add_months = add_months


def _window_for(strategy: str) -> tuple[datetime, datetime, int, int, int]:
    if registry.interval_of(strategy) != "1h":
        return (_INTRADAY_IS_START, _INTRADAY_IS_END,
                _INTRADAY_TRAIN, _INTRADAY_VAL, _INTRADAY_STEP)
    return _IS_START, _IS_END_EXCLUSIVE, _TRAIN_MONTHS, _VAL_MONTHS, _STEP_MONTHS


def folds_for(strategy: str) -> list[tuple[datetime, datetime, datetime, datetime]]:
    start, end, train, val, step = _window_for(strategy)
    return walk_forward_folds(start, end, train, val, step)


def walk_forward_folds(is_start: datetime = _IS_START,
                       is_end_exclusive: datetime = _IS_END_EXCLUSIVE,
                       train_months: int = _TRAIN_MONTHS,
                       val_months: int = _VAL_MONTHS,
                       step_months: int = _STEP_MONTHS
                       ) -> list[tuple[datetime, datetime, datetime, datetime]]:
    folds = []
    start = is_start
    while True:
        train_end = add_months(start, train_months)
        val_end = add_months(train_end, val_months)
        if val_end > is_end_exclusive:
            break
        folds.append((start, train_end, train_end, val_end))
        start = add_months(start, step_months)
    # Guardrail #1 tripwire: every bound must sit inside the [is_start, is_end) window.
    for train_start, train_end, val_start, val_end in folds:
        if not (train_start >= is_start and val_end <= is_end_exclusive):
            raise ValueError(
                f"fold escapes IS window: {(train_start, train_end, val_start, val_end)}"
            )
    return folds


# Per-process cache of price/funding series keyed by exact window, so batch
# workers tuning many strategies for one symbol don't re-query the same folds.
_DATA_CACHE: dict = {}
_CACHE_MAX = 512


def _load_train_data(ch_client, strategy: str, symbol: str, start: datetime, end: datetime):
    base, interval = registry.normalize(strategy)
    key = (base, interval, symbol, start, end)
    cached = _DATA_CACHE.get(key)
    if cached is not None:
        return cached
    if base == "funding_arb":
        data = load_funding(ch_client, symbol, start=start, end=end)
    else:
        data = load_ohlcv(ch_client, symbol, interval=interval, start=start, end=end)
    if len(_DATA_CACHE) >= _CACHE_MAX:
        _DATA_CACHE.clear()
    _DATA_CACHE[key] = data
    return data


def _fees_for(ch_client, symbol: str, fee: float | None):
    """(execution fee, taker fallback fee). Maker mode pays maker fees."""
    maker = fees.execution_mode() == "maker"
    if fee is None:
        fee = fees.fee_for_symbol(ch_client, symbol,
                                  mode="maker" if maker else "taker")
    taker_fee = fees.fee_for_symbol(ch_client, symbol, mode="taker") if maker else None
    return fee, taker_fee


def _run(data, strategy: str, params: dict, fee: float,
         taker_fee: float | None = None):
    if registry.base_name(strategy) == "funding_arb":
        return run_funding_backtest(data, threshold=params["threshold"], fee=fee)
    close = data["close"] if hasattr(data, "columns") else data
    entries, exits = registry.signals(strategy, close, params)
    if fees.execution_mode() == "maker" and hasattr(data, "columns"):
        return run_maker_backtest(data, entries, exits, fee=fee,
                                  limit_bps=fees.maker_limit_bps(),
                                  max_wait=fees.maker_max_wait(),
                                  taker_fee=taker_fee if taker_fee is not None else fee)
    return run_signal_backtest(close, entries, exits, fee=fee)


def _suggest(trial, strategy: str) -> dict:
    if registry.base_name(strategy) == "funding_arb":
        return {"threshold": trial.suggest_float("threshold", 1e-5, 1e-3, log=True)}
    return registry.SEARCH_SPACES[registry.base_name(strategy)](trial)


def fit(ch_client, strategy: str, symbol: str, n_trials: int = 60,
        seed: int = 42, fee: float | None = None,
        is_start: datetime | None = None, is_end_exclusive: datetime | None = None
        ) -> tuple[dict, list]:
    """Walk-forward tune one (strategy, symbol). Returns (record, folds).

    Does not publish; the caller decides (champion/challenger adoption).
    Intraday strategies use their own (shorter, later) IS window.
    """
    fee, taker_fee = _fees_for(ch_client, symbol, fee)
    folds = (folds_for(strategy) if is_start is None or is_end_exclusive is None
             else walk_forward_folds(is_start, is_end_exclusive))
    fold_params, fold_train_sharpes, fold_val_sharpes = [], [], []
    for train_start, _, val_start, val_end in folds:
        try:
            train_data = _load_train_data(ch_client, strategy, symbol, train_start, val_start)
            val_data = _load_train_data(ch_client, strategy, symbol, val_start, val_end)
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(
                lambda trial: _run(train_data, strategy, _suggest(trial, strategy),
                                   fee, taker_fee).sharpe,
                n_trials=n_trials,
            )
        except ValueError as exc:
            print(f"{strategy} {symbol}: skipping fold "
                  f"{train_start.date()}->{val_end.date()}: {exc}")
            continue
        params = dict(study.best_params)
        fold_params.append(params)
        fold_train_sharpes.append(float(study.best_value))
        fold_val_sharpes.append(float(_run(val_data, strategy, params, fee, taker_fee).sharpe))

    if not fold_params:
        raise ValueError(f"no tunable folds for {strategy} {symbol}")
    best = max(range(len(fold_params)), key=lambda i: fold_val_sharpes[i])
    record = {
        "params": fold_params[best],
        "train_sharpe": fold_train_sharpes[best],
        "validation_sharpe": sum(fold_val_sharpes) / len(fold_val_sharpes),
        "folds": len(fold_params),
    }
    return record, folds


def fit_grid(ch_client, strategy: str, symbol: str, fee: float | None = None,
             is_start: datetime | None = None, is_end_exclusive: datetime | None = None
             ) -> tuple[dict, list]:
    """Fast walk-forward tuning by scanning a coarse param grid per fold."""
    fee, taker_fee = _fees_for(ch_client, symbol, fee)
    grid = registry.grid(strategy)
    if not grid:
        raise ValueError(f"no grid for {strategy}")
    folds = (folds_for(strategy) if is_start is None or is_end_exclusive is None
             else walk_forward_folds(is_start, is_end_exclusive))
    fold_params, fold_train, fold_val = [], [], []
    for train_start, _, val_start, val_end in folds:
        try:
            train_data = _load_train_data(ch_client, strategy, symbol, train_start, val_start)
            val_data = _load_train_data(ch_client, strategy, symbol, val_start, val_end)
        except ValueError:
            continue
        best, best_score = None, float("-inf")
        for params in grid:
            score = _run(train_data, strategy, params, fee, taker_fee).sharpe
            if score > best_score:
                best, best_score = params, score
        if best is None:
            continue
        fold_params.append(best)
        fold_train.append(best_score)
        fold_val.append(float(_run(val_data, strategy, best, fee, taker_fee).sharpe))
    if not fold_params:
        raise ValueError(f"no tunable folds for {strategy} {symbol}")
    index = max(range(len(fold_params)), key=lambda i: fold_val[i])
    record = {
        "params": fold_params[index],
        "train_sharpe": fold_train[index],
        "validation_sharpe": sum(fold_val) / len(fold_val),
        "folds": len(fold_params),
    }
    return record, folds


def publish(ch_client, strategy: str, symbol: str, record: dict, valid_from: datetime) -> None:
    """Insert a versioned tuned_params row."""
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [[strategy, symbol, json.dumps(record["params"]), record["train_sharpe"],
          record["validation_sharpe"], record["folds"], datetime.now(timezone.utc),
          valid_from]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at", "valid_from"],
    )


def validate_params(ch_client, strategy: str, symbol: str, params: dict,
                    folds: list, fee: float | None = None) -> float | None:
    """Mean validation Sharpe of a fixed param set over the given folds."""
    fee, taker_fee = _fees_for(ch_client, symbol, fee)
    values = []
    for _, _, val_start, val_end in folds:
        try:
            data = _load_train_data(ch_client, strategy, symbol, val_start, val_end)
        except ValueError:
            continue
        values.append(float(_run(data, strategy, params, fee, taker_fee).sharpe))
    return sum(values) / len(values) if values else None


def trade_edge(ch_client, strategy: str, symbol: str, params: dict,
               folds: list | None = None, fee: float | None = None) -> float | None:
    """Mean net-of-fee PnL per trade across folds (min-edge filter input)."""
    fee, taker_fee = _fees_for(ch_client, symbol, fee)
    folds = folds if folds is not None else folds_for(strategy)
    edges = []
    for _, _, val_start, val_end in folds:
        try:
            data = _load_train_data(ch_client, strategy, symbol, val_start, val_end)
        except ValueError:
            continue
        edges.append(_run(data, strategy, params, fee, taker_fee).avg_trade_return)
    return sum(edges) / len(edges) if edges else None


DEFAULT_ADOPT_MARGIN = 0.01


def adopt_candidate(candidate_val: float | None, incumbent_val: float | None,
                    margin: float = 0.0) -> bool:
    """Champion/challenger: adopt only if the candidate beats the incumbent by ``margin``."""
    if incumbent_val is None:
        return True
    if candidate_val is None:
        return False
    return candidate_val >= incumbent_val + margin


def tune(ch_client, strategy: str, symbol: str, n_trials: int = 60,
         seed: int = 42, fee: float = DEFAULT_FEE,
         is_start: datetime = _IS_START, is_end_exclusive: datetime = _IS_END_EXCLUSIVE,
         valid_from: datetime | None = None) -> dict:
    """Tune and publish unconditionally (used for the initial fixed-IS tune).

    ``valid_from`` defaults to ``is_end_exclusive``, so re-running is idempotent.
    """
    record, _ = fit(ch_client, strategy, symbol, n_trials=n_trials, seed=seed, fee=fee,
                    is_start=is_start, is_end_exclusive=is_end_exclusive)
    publish(ch_client, strategy, symbol, record,
            valid_from if valid_from is not None else is_end_exclusive)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward tune strategy parameters (IS only)")
    parser.add_argument("--strategy", default="all",
                        choices=[*TUNABLE_STRATEGIES, "all"])
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args(argv)

    strategies = TUNABLE_STRATEGIES if args.strategy == "all" else (args.strategy,)
    ch_client = get_clickhouse_client()
    symbols = universe(ch_client) if args.symbols == "all" else tuple(args.symbols.split(","))
    for strategy in strategies:
        for symbol in symbols:
            try:
                record = tune(ch_client, strategy, symbol, n_trials=args.trials)
                print(f"{strategy} {symbol}: tuned {record['params']} "
                      f"(val sharpe {record['validation_sharpe']:.3f})")
            except Exception as exc:
                print(f"{strategy} {symbol}: FAILED: {exc}")


if __name__ == "__main__":
    main()
