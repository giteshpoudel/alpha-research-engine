"""Optuna walk-forward parameter tuning — IS window only (guardrail #1).

Never imports or references OOS/PRELIM windows. The tripwire assertion in
walk_forward_folds raises if any fold bound escapes the IS window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import optuna

from src.backtesting.data import load_funding, load_ohlcv
from src.backtesting.engine import run_funding_backtest, run_signal_backtest
from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES

# funding_arb retired 2026-09: Phase 5 OOS comparison showed fee drag exceeds
# carry at retail venues in every variant (tuned thresholds either sit out the
# market or lose heavily). Code and historical results remain; it is excluded
# from future tuning runs.
TUNABLE_STRATEGIES = ("mean_reversion",)
DEFAULT_FEE = 0.001
_TRAIN_MONTHS, _VAL_MONTHS, _STEP_MONTHS = 12, 3, 3
_IS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
_IS_END_EXCLUSIVE = datetime(2025, 1, 1, tzinfo=timezone.utc)  # IS end candle + 1h

optuna.logging.set_verbosity(optuna.logging.WARNING)


def add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    return dt.replace(year=dt.year + month // 12, month=month % 12 + 1)


# Backwards-compatible private alias.
_add_months = add_months


def walk_forward_folds(is_start: datetime = _IS_START,
                       is_end_exclusive: datetime = _IS_END_EXCLUSIVE
                       ) -> list[tuple[datetime, datetime, datetime, datetime]]:
    folds = []
    start = is_start
    while True:
        train_end = add_months(start, _TRAIN_MONTHS)
        val_end = add_months(train_end, _VAL_MONTHS)
        if val_end > is_end_exclusive:
            break
        folds.append((start, train_end, train_end, val_end))
        start = add_months(start, _STEP_MONTHS)
    # Guardrail #1 tripwire: every bound must sit inside the [is_start, is_end) window.
    for train_start, train_end, val_start, val_end in folds:
        if not (train_start >= is_start and val_end <= is_end_exclusive):
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


def fit(ch_client, strategy: str, symbol: str, n_trials: int = 60,
        seed: int = 42, fee: float = DEFAULT_FEE,
        is_start: datetime = _IS_START, is_end_exclusive: datetime = _IS_END_EXCLUSIVE
        ) -> tuple[dict, list]:
    """Walk-forward tune one (strategy, symbol). Returns (record, folds).

    Does not publish; the caller decides (champion/challenger adoption).
    """
    base_strategy = strategy.removeprefix("TEST_")
    folds = walk_forward_folds(is_start, is_end_exclusive)
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
                    folds: list, fee: float = DEFAULT_FEE) -> float | None:
    """Mean validation Sharpe of a fixed param set over the given folds."""
    base_strategy = strategy.removeprefix("TEST_")
    values = []
    for _, _, val_start, val_end in folds:
        try:
            data = _load_train_data(ch_client, base_strategy, symbol, val_start, val_end)
        except ValueError:
            continue
        values.append(float(_run(data, base_strategy, params, fee).sharpe))
    return sum(values) / len(values) if values else None


def adopt_candidate(candidate_val: float | None, incumbent_val: float | None) -> bool:
    """Champion/challenger: adopt only if the candidate is not worse."""
    if incumbent_val is None:
        return True
    if candidate_val is None:
        return False
    return candidate_val >= incumbent_val


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
