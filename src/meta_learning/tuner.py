"""Optuna walk-forward parameter tuning — IS window only (guardrail #1).

Never imports or references OOS/PRELIM windows. The tripwire assertion in
walk_forward_folds raises if any fold bound escapes the IS window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import optuna

from src.backtesting.data import WINDOWS, load_funding, load_ohlcv
from src.backtesting.engine import run_funding_backtest, run_signal_backtest
from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES

TUNABLE_STRATEGIES = ("mean_reversion", "funding_arb")
DEFAULT_FEE = 0.001
_TRAIN_MONTHS, _VAL_MONTHS, _STEP_MONTHS = 12, 3, 3
_IS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
_IS_END_EXCLUSIVE = datetime(2025, 1, 1, tzinfo=timezone.utc)  # IS end candle + 1h

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    return dt.replace(year=dt.year + month // 12, month=month % 12 + 1)


def walk_forward_folds() -> list[tuple[datetime, datetime, datetime, datetime]]:
    folds = []
    start = _IS_START
    while True:
        train_end = _add_months(start, _TRAIN_MONTHS)
        val_end = _add_months(train_end, _VAL_MONTHS)
        if val_end > _IS_END_EXCLUSIVE:
            break
        folds.append((start, train_end, train_end, val_end))
        start = _add_months(start, _STEP_MONTHS)
    # Guardrail #1 tripwire: every bound must sit inside the IS window.
    for train_start, train_end, val_start, val_end in folds:
        if not (train_start >= WINDOWS["IS"][0] and val_end <= _IS_END_EXCLUSIVE):
            raise ValueError(
                f"fold escapes IS window: {(train_start, train_end, val_start, val_end)}"
            )
    return folds


def _load_train_data(ch_client, strategy: str, symbol: str, start: datetime, end: datetime):
    if strategy == "funding_arb":
        return load_funding(ch_client, symbol, start=start, end=end)
    return load_ohlcv(ch_client, symbol, start=start, end=end)["close"]


def _run(data, strategy: str, params: dict, fee: float):
    if strategy == "funding_arb":
        return run_funding_backtest(data, threshold=params["threshold"], fee=fee)
    entries, exits = mean_reversion.signals(
        data, window=params["window"], z_entry=params["z_entry"], z_exit=params["z_exit"]
    )
    return run_signal_backtest(data, entries, exits, fee=fee)


def _suggest(trial, strategy: str) -> dict:
    if strategy == "funding_arb":
        return {"threshold": trial.suggest_float("threshold", 1e-5, 1e-3, log=True)}
    return {
        "window": trial.suggest_int("window", 12, 72),
        "z_entry": trial.suggest_float("z_entry", -3.5, -1.0),
        "z_exit": trial.suggest_float("z_exit", -0.5, 0.5),
    }


def tune(ch_client, strategy: str, symbol: str, n_trials: int = 60,
         seed: int = 42, fee: float = DEFAULT_FEE) -> dict:
    """Walk-forward tune one (strategy, symbol). Upserts tuned_params. Returns the record."""
    base_strategy = strategy.removeprefix("TEST_")
    folds = walk_forward_folds()
    fold_params, fold_train_sharpes, fold_val_sharpes = [], [], []
    for train_start, _, val_start, val_end in folds:
        try:
            train_data = _load_train_data(ch_client, base_strategy, symbol, train_start, val_start)
            val_data = _load_train_data(ch_client, base_strategy, symbol, val_start, val_end)
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(
                lambda trial: _run(train_data, base_strategy, _suggest(trial, base_strategy), fee).sharpe,
                n_trials=n_trials,
            )
        except ValueError as exc:
            print(f"{strategy} {symbol}: skipping fold "
                  f"{train_start.date()}->{val_end.date()}: {exc}")
            continue
        params = dict(study.best_params)
        fold_params.append(params)
        fold_train_sharpes.append(float(study.best_value))
        fold_val_sharpes.append(float(_run(val_data, base_strategy, params, fee).sharpe))

    if not fold_params:
        raise ValueError(f"no tunable folds for {strategy} {symbol}")
    best = max(range(len(fold_params)), key=lambda i: fold_val_sharpes[i])
    record = {
        "params": fold_params[best],
        "train_sharpe": fold_train_sharpes[best],
        "validation_sharpe": sum(fold_val_sharpes) / len(fold_val_sharpes),
        "folds": len(fold_params),
    }
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [[strategy, symbol, json.dumps(record["params"]), record["train_sharpe"],
          record["validation_sharpe"], record["folds"], datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
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
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    ch_client = get_clickhouse_client()
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
