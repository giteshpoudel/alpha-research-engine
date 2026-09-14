# Phase 5 Meta-Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optuna walk-forward parameter tuning (IS window only) for mean_reversion and funding_arb, plus an OOS comparison simulation of static vs tuned vs causal-allocator variants — all stored in ClickHouse alongside existing backtest results.

**Architecture:** `src/meta_learning/tuner.py` (walk-forward folds + Optuna objectives, IS data only) writes `tuned_params`; `src/meta_learning/compare.py` runs OOS variants (static/tuned/allocator) through the Phase 3 engine and stores via a `store_result` helper extracted from `runner.py`; `src/meta_learning/allocator.py` is the causal strategy picker.

**Tech Stack:** Python 3.11, `optuna` (new), existing engine/data/runner modules, clickhouse-connect, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-phase5-meta-learning-design.md`

## Global Constraints

- **Guardrail #1 tripwire:** `tuner.py` asserts every fold bound lies inside `WINDOWS["IS"]` and raises otherwise. Tuner code must never reference `WINDOWS["OOS"]` or `WINDOWS["PRELIM"]`.
- Walk-forward: 12-month train + 3-month validate, 3-month step, 8 folds. IS end is treated as the exclusive bound `2025-01-01 00:00 UTC` (= `WINDOWS["IS"][1]` + 1h) so the final fold's validate window fits.
- Optuna: `TPESampler(seed=42)`, `direction="maximize"`, 60 trials/fold default. Objective = train-window Sharpe from the Phase 3 engine (0.0-guarded).
- Final params per strategy×symbol = the fold params with the best validation Sharpe; `validation_sharpe` = mean of all folds' validation Sharpes under that fold's params; `train_sharpe` = the selected fold's best train Sharpe.
- No NaN/inf in stored metrics (engine guarantees; do not bypass `engine` functions).
- `sentiment_momentum` is NEVER tuned (documented exclusion).
- Deterministic behavior: seeded sampler; allocator picks depend only on data strictly before the rebalance point.
- Comparison results go to the existing `backtest_runs`/`backtest_equity` with `variant` inside `params_json` — never schema changes to those tables; never overwrite Phase 3 static runs (run_ids differ because params_json differs).
- Test hygiene: tuning/compare tests use `TEST_`-prefixed strategy names and symbol `TEST` where they write, teardown deletes with `mutations_sync`. The real tuning run uses untruncated strategies on real symbols (that's the deliverable, stored permanently).
- The existing 105 tests must stay green. New dependency: `optuna` only.

---

### Task 1: optuna dependency + `tuned_params` schema

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ingestion/schemas.py` (one DDL template + register)
- Test: `tests/test_schemas.py` (append)

**Interfaces:**
- Consumes: `create_clickhouse_schema`, `database_name`.
- Produces: table `tuned_params` — `strategy LowCardinality(String), symbol String, params_json String, train_sharpe Float32, validation_sharpe Float32, folds UInt8, tuned_at DateTime64(3)`, `ReplacingMergeTree`, `ORDER BY (strategy, symbol)`. Task 2 upserts into it.

- [ ] **Step 1: Dependencies**

Append to `requirements.txt`:

```
optuna>=4.0
```

Run: `/Users/giteshpoudel/Documents/GitHub/alpha-research-engine/.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

Append to `tests/test_schemas.py`:

```python
def test_tuned_params_table(ch_client):
    rows = ch_client.query(f"DESCRIBE TABLE {database_name()}.tuned_params").result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["strategy"] == "LowCardinality(String)"
    assert cols["symbol"] == "String"
    assert cols["params_json"] == "String"
    assert cols["train_sharpe"] == "Float32"
    assert cols["validation_sharpe"] == "Float32"
    assert cols["folds"] == "UInt8"
    assert cols["tuned_at"] == "DateTime64(3)"
    engine = ch_client.query(
        "SELECT engine FROM system.tables WHERE database = {db:String} AND name = 'tuned_params'",
        parameters={"db": database_name()},
    ).result_rows
    assert "ReplacingMergeTree" in engine[0][0]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py -v -k tuned_params`
Expected: FAIL — `Code: 60 ... Table alpha.tuned_params does not exist`.

- [ ] **Step 4: Add the DDL to `src/ingestion/schemas.py`**

Add after `_BACKTEST_EQUITY_DDL`:

```python
# One row per (strategy, symbol): re-tuning replaces it.
_TUNED_PARAMS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.tuned_params
(
    strategy LowCardinality(String),
    symbol String,
    params_json String,
    train_sharpe Float32,
    validation_sharpe Float32,
    folds UInt8,
    tuned_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol)
"""
```

Extend the DDL loop:

```python
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL, _SENTIMENT_METRICS_DDL,
                _BACKTEST_RUNS_DDL, _BACKTEST_EQUITY_DDL, _TUNED_PARAMS_DDL):
        client.command(ddl.format(db=db))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: all tests PASS (24 in this file).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/ingestion/schemas.py tests/test_schemas.py
git commit -m "feat: add tuned_params table and optuna dependency"
```

---

### Task 2: Walk-forward tuner (`tuner.py`)

**Files:**
- Create: `src/meta_learning/__init__.py` (empty)
- Create: `src/meta_learning/tuner.py`
- Test: `tests/test_tuner.py`

**Interfaces:**
- Consumes: `backtesting.data.WINDOWS/load_ohlcv/load_funding`, `backtesting.engine.run_signal_backtest/run_funding_backtest`, `backtesting.strategies.mean_reversion`, `ingestion.schemas.database_name/get_clickhouse_client`.
- Produces:
  - `TUNABLE_STRATEGIES = ("mean_reversion", "funding_arb")`
  - `walk_forward_folds() -> list[tuple[datetime, datetime, datetime, datetime]]` — 8 folds, `(train_start, train_end, val_start, val_end)`
  - `tune(ch_client, strategy, symbol, n_trials=60, seed=42, fee=0.001) -> dict` — `{params, train_sharpe, validation_sharpe, folds}`; upserts `tuned_params`
  - `main(argv=None) -> None` — CLI: `--strategy {mean_reversion,funding_arb,all}`, `--symbols BTC,ETH|all`, `--trials N`
- Used by Task 3: `walk_forward_folds`, `tune`'s stored output read via a plain SELECT (no import of tuner needed by compare).

- [ ] **Step 1: Write the failing tests**

`tests/test_tuner.py`:

```python
from datetime import datetime, timezone

import pytest

from src.meta_learning.tuner import TUNABLE_STRATEGIES, tune, walk_forward_folds
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_walk_forward_folds_structure():
    folds = walk_forward_folds()
    assert len(folds) == 8
    for train_start, train_end, val_start, val_end in folds:
        assert train_end == val_start
        assert (train_end - train_start).days in (365, 366)
        assert (val_end - val_start).days in (89, 90, 91, 92)
        # tripwire: everything inside IS (IS end candle is 2024-12-31 23:00)
        assert train_start >= datetime(2022, 1, 1, tzinfo=timezone.utc)
        assert val_end <= datetime(2025, 1, 1, tzinfo=timezone.utc)
    # validate windows never overlap
    val_windows = [(f[2], f[3]) for f in folds]
    for (a_start, a_end), (b_start, b_end) in zip(val_windows, val_windows[1:]):
        assert a_end <= b_start


def test_tune_stores_tuned_params():
    ch = get_clickhouse_client()
    try:
        result = tune(ch, "TEST_mean_reversion", "BTC", n_trials=5, seed=42)
        assert set(result) == {"params", "train_sharpe", "validation_sharpe", "folds"}
        assert result["folds"] == 8
        params = result["params"]
        assert 12 <= params["window"] <= 72
        assert -3.5 <= params["z_entry"] <= -1.0
        assert -0.5 <= params["z_exit"] <= 0.5
        rows = ch.query(
            f"SELECT params_json, folds FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = 'TEST_mean_reversion' AND symbol = 'BTC'"
        ).result_rows
        assert len(rows) == 1
        assert rows[0][1] == 8
        # re-tune replaces, never duplicates
        tune(ch, "TEST_mean_reversion", "BTC", n_trials=5, seed=42)
        rows2 = ch.query(
            f"SELECT count() FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = 'TEST_mean_reversion' AND symbol = 'BTC'"
        ).result_rows
        assert rows2[0][0] == 1
    finally:
        ch.command(
            f"ALTER TABLE {database_name()}.tuned_params DELETE WHERE strategy = 'TEST_mean_reversion'",
            settings={"mutations_sync": 1},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tuner.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.meta_learning.tuner'`.

- [ ] **Step 3: Implement `src/meta_learning/tuner.py`**

```python
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
        assert train_start >= WINDOWS["IS"][0] and val_end <= _IS_END_EXCLUSIVE, (
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
        train_data = _load_train_data(ch_client, base_strategy, symbol, train_start, val_start)
        val_data = _load_train_data(ch_client, base_strategy, symbol, val_start, val_end)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            lambda trial: _run(train_data, base_strategy, _suggest(trial, base_strategy), fee).sharpe,
            n_trials=n_trials,
        )
        params = dict(study.best_params)
        fold_params.append(params)
        fold_train_sharpes.append(float(study.best_value))
        fold_val_sharpes.append(float(_run(val_data, base_strategy, params, fee).sharpe))

    best = max(range(len(folds)), key=lambda i: fold_val_sharpes[i])
    record = {
        "params": fold_params[best],
        "train_sharpe": fold_train_sharpes[best],
        "validation_sharpe": sum(fold_val_sharpes) / len(fold_val_sharpes),
        "folds": len(folds),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tuner.py -v`
Expected: 2 tests PASS (the tune test takes ~1-3 min: 8 folds × 5 trials × engine calls incl. numba JIT on first).

- [ ] **Step 5: Commit**

```bash
git add src/meta_learning/__init__.py src/meta_learning/tuner.py tests/test_tuner.py
git commit -m "feat: add Optuna walk-forward tuner (IS window only)"
```

---

### Task 3: Allocator + comparison runner (`allocator.py`, `compare.py`, `store_result` extraction)

**Files:**
- Modify: `src/backtesting/runner.py` (extract `store_result`; no behavior change)
- Create: `src/meta_learning/allocator.py`
- Create: `src/meta_learning/compare.py`
- Test: `tests/test_allocator.py`, `tests/test_compare.py`

**Interfaces:**
- Consumes: `backtesting.data`, `backtesting.engine`, `tuned_params` table (Task 2), `backtesting.runner` internals.
- Produces:
  - `runner.store_result(ch_client, strategy, symbol, interval, params_json, window, start_ts, end_ts, result) -> str` — the run_id + dual-insert logic, now public (runner.run_backtest keeps its signature and delegates)
  - `allocator.pick_strategy(trailing_returns: dict[str, float | None], default: str = "mean_reversion") -> str`
  - `allocator.allocator_windows(oos_start, oos_end, rebalance_days=7, trailing_days=30) -> list[tuple[datetime, datetime]]`
  - `compare.run_variant(ch_client, strategy, symbol, variant, params, fee=0.001) -> str` — OOS backtest with explicit params, stored via `store_result` with `{"variant": ..., **params}` in params_json
  - `compare.run_allocator(ch_client, symbol, fee=0.001) -> str` — causal composite, stored as strategy `allocator` with variant `allocator`
  - `compare.main(argv=None) -> None` — CLI: `python -m src.meta_learning.compare --symbols all`

- [ ] **Step 1: Write the failing tests**

`tests/test_allocator.py`:

```python
from datetime import datetime, timedelta, timezone

from src.meta_learning.allocator import allocator_windows, pick_strategy

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_pick_strategy_argmax():
    assert pick_strategy({"mean_reversion": 0.05, "funding_arb": 0.12}) == "funding_arb"
    assert pick_strategy({"mean_reversion": 0.2, "funding_arb": -0.1}) == "mean_reversion"


def test_pick_strategy_defaults_on_empty_history():
    assert pick_strategy({"mean_reversion": None, "funding_arb": None}) == "mean_reversion"
    assert pick_strategy({"mean_reversion": 0.0, "funding_arb": 0.0}) == "mean_reversion"


def test_allocator_windows():
    end = T0 + timedelta(days=20)
    windows = allocator_windows(T0, end, rebalance_days=7)
    assert windows[0] == (T0, T0 + timedelta(days=7))
    assert windows[-1][1] == end  # last window clamped
    for (_, a_end), (b_start, _) in zip(windows, windows[1:]):
        assert a_end == b_start  # contiguous, no gaps
```

`tests/test_compare.py`:

```python
import json

import pytest

from src.meta_learning.compare import run_allocator, run_variant
from src.ingestion.schemas import database_name, get_clickhouse_client

MR_PARAMS = {"window": 24, "z_entry": -2.0, "z_exit": 0.0}
FA_PARAMS = {"threshold": 0.0003}


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    for table in ("backtest_runs", "backtest_equity"):
        client.command(
            f"ALTER TABLE {db}.{table} DELETE WHERE strategy LIKE 'TEST%'",
            settings={"mutations_sync": 1},
        )
    client.command(
        f"ALTER TABLE {db}.tuned_params DELETE WHERE strategy LIKE 'TEST%'",
        settings={"mutations_sync": 1},
    )


def _stored_run_ids(ch_client):
    rows = ch_client.query(
        f"SELECT run_id FROM {database_name()}.backtest_runs FINAL WHERE strategy LIKE 'TEST%'"
    ).result_rows
    return {r[0] for r in rows}


def test_run_variant_stores_variant_and_distinct_ids(ch_client):
    rid_tuned = run_variant(ch_client, "TEST_mean_reversion", "BTC", "tuned", MR_PARAMS)
    rid_static = run_variant(ch_client, "TEST_mean_reversion", "BTC", "static", MR_PARAMS)
    assert rid_tuned != rid_static  # variant inside params_json -> different run_ids
    rows = ch_client.query(
        f"SELECT params_json, window FROM {database_name()}.backtest_runs FINAL "
        "WHERE strategy = 'TEST_mean_reversion'"
    ).result_rows
    variants = {json.loads(r[0])["variant"] for r in rows}
    assert variants == {"tuned", "static"}
    assert all(r[1] == "OOS" for r in rows)


def test_run_allocator_causal_composite(ch_client):
    # Seed tuned params the allocator reads (TEST_-prefixed strategy names)
    from datetime import datetime, timezone
    ch_client.insert(
        f"{database_name()}.tuned_params",
        [["TEST_mean_reversion", "BTC", json.dumps(MR_PARAMS), 1.0, 1.0, 8,
          datetime.now(timezone.utc)],
         ["TEST_funding_arb", "BTC", json.dumps(FA_PARAMS), 1.0, 1.0, 8,
          datetime.now(timezone.utc)]],
        column_names=["strategy", "symbol", "params_json", "train_sharpe",
                      "validation_sharpe", "folds", "tuned_at"],
    )
    rid = run_allocator(ch_client, "BTC")
    rows = ch_client.query(
        f"SELECT strategy, params_json FROM {database_name()}.backtest_runs FINAL "
        "WHERE run_id = {r:String}", parameters={"r": rid},
    ).result_rows
    assert rows[0][0] == "allocator"
    assert json.loads(rows[0][1])["variant"] == "allocator"
    eq = ch_client.query(
        f"SELECT count() FROM {database_name()}.backtest_equity FINAL "
        "WHERE run_id = {r:String}", parameters={"r": rid},
    ).result_rows
    assert eq[0][0] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_allocator.py tests/test_compare.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.meta_learning.allocator'`.

- [ ] **Step 3: Extract `store_result` in `src/backtesting/runner.py`**

Move the run_id + dual-insert block out of `run_backtest` into a public function (no behavior change — `run_backtest` keeps its signature and delegates):

```python
def store_result(ch_client, strategy: str, symbol: str, interval: str,
                 params_json: str, window: str, start_ts: datetime,
                 end_ts: datetime, result) -> str:
    """Compute run_id and insert one backtest_runs row plus equity rows. Returns run_id."""
    run_id = compute_run_id(strategy, symbol, interval, params_json, window, start_ts, end_ts)
    now = datetime.now(timezone.utc)
    db = database_name()
    ch_client.insert(
        f"{db}.backtest_runs",
        [[run_id, strategy, symbol, interval, params_json, window, start_ts, end_ts,
          result.total_return, result.sharpe, result.sortino, result.max_drawdown,
          result.calmar, result.win_rate, result.num_trades, now]],
        column_names=list(_RUN_COLUMNS),
    )
    equity_rows = [
        [run_id, ts.to_pydatetime(), float(value)]
        for ts, value in result.equity_curve.items()
    ]
    ch_client.insert(
        f"{db}.backtest_equity", equity_rows,
        column_names=["run_id", "ts", "equity"],
    )
    return run_id
```

`run_backtest` becomes:

```python
def run_backtest(ch_client, strategy: str, symbol: str, window: str,
                 fee: float = DEFAULT_FEE) -> str:
    """Run one (strategy, symbol, window) backtest and store results. Returns run_id."""
    params_json, start_ts, end_ts, result, window = _execute(ch_client, strategy, symbol, window, fee)
    return store_result(ch_client, strategy, symbol, "1h", params_json, window,
                        start_ts, end_ts, result)
```

Run: `python -m pytest tests/test_runner.py -v`
Expected: 3 existing tests still PASS (behavior unchanged).

- [ ] **Step 4: Implement `src/meta_learning/allocator.py`**

```python
"""Causal trailing-performance strategy picker for the comparison simulation.

Picks depend only on data strictly before the rebalance point (no lookahead).
"""

from __future__ import annotations

from datetime import datetime, timedelta


def pick_strategy(trailing_returns: dict[str, float | None],
                  default: str = "mean_reversion") -> str:
    """Argmax over trailing returns; default when history is empty or all zero."""
    best, best_ret = default, None
    for strategy, ret in trailing_returns.items():
        if ret is not None and ret != 0.0 and (best_ret is None or ret > best_ret):
            best, best_ret = strategy, ret
    return best


def allocator_windows(oos_start: datetime, oos_end: datetime,
                      rebalance_days: int = 7, trailing_days: int = 30) -> list[tuple[datetime, datetime]]:
    """Contiguous (start, end) rebalance windows covering [oos_start, oos_end)."""
    windows = []
    cursor = oos_start
    while cursor < oos_end:
        end = min(cursor + timedelta(days=rebalance_days), oos_end)
        windows.append((cursor, end))
        cursor = end
    return windows
```

- [ ] **Step 5: Implement `src/meta_learning/compare.py`**

```python
"""OOS comparison simulation: static vs tuned vs causal allocator.

No optimization anywhere in this module — it consumes tuned_params produced
by tuner.py and evaluates on the OOS window only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtesting.data import WINDOWS, load_funding, load_ohlcv, slice_window
from src.backtesting.engine import (
    BacktestResult,
    _metrics,
    run_funding_backtest,
    run_signal_backtest,
)
from src.backtesting.runner import store_result
from src.backtesting.strategies import mean_reversion
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES
from src.meta_learning.allocator import allocator_windows, pick_strategy

DEFAULT_FEE = 0.001
_REBALANCE_DAYS = 7
_TRAILING_DAYS = 30


def _tuned_params(ch_client, strategy: str, symbol: str) -> dict:
    rows = ch_client.query(
        f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} LIMIT 1",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    if not rows:
        raise ValueError(f"no tuned params for {strategy} {symbol} — run tuner first")
    return json.loads(rows[0][0])


def _execute_variant(ch_client, strategy: str, symbol: str, params: dict, fee: float):
    """OOS backtest with explicit params. Returns (start_ts, end_ts, result)."""
    base_strategy = strategy.removeprefix("TEST_")
    start, _ = WINDOWS["OOS"]
    if base_strategy == "funding_arb":
        funding = slice_window(load_funding(ch_client, symbol, start=start), "OOS")
        result = run_funding_backtest(funding, threshold=params["threshold"], fee=fee)
    else:
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], "OOS")
        entries, exits = mean_reversion.signals(prices, **params)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
    equity = result.equity_curve
    return equity.index[0].to_pydatetime(), equity.index[-1].to_pydatetime(), result


def run_variant(ch_client, strategy: str, symbol: str, variant: str,
                params: dict, fee: float = DEFAULT_FEE) -> str:
    """Store one OOS variant run. Returns run_id."""
    start_ts, end_ts, result = _execute_variant(ch_client, strategy, symbol, params, fee)
    params_json = json.dumps({"variant": variant, **params})
    return store_result(ch_client, strategy, symbol, "1h", params_json, "OOS",
                        start_ts, end_ts, result)


def _segment_return(result: BacktestResult) -> float:
    return result.total_return


def run_allocator(ch_client, symbol: str, fee: float = DEFAULT_FEE) -> str:
    """Causal composite: at each rebalance, run the strategy with the best
    trailing-30-day return (computed strictly from data before the rebalance
    point). Stored as strategy 'allocator' with variant 'allocator'."""
    oos_start, _ = WINDOWS["OOS"]
    prices = slice_window(load_ohlcv(ch_client, symbol, start=oos_start)["close"], "OOS")
    funding = slice_window(load_funding(ch_client, symbol, start=oos_start), "OOS")
    mr_params = _tuned_params(ch_client, "TEST_mean_reversion", symbol) if _has_test_params(ch_client) \
        else _tuned_params(ch_client, "mean_reversion", symbol)
    fa_params = _tuned_params(ch_client, "TEST_funding_arb", symbol) if _has_test_params(ch_client) \
        else _tuned_params(ch_client, "funding_arb", symbol)

    oos_end = prices.index[-1].to_pydatetime()
    equity_points: dict[pd.Timestamp, float] = {}
    composite = 1.0
    segment_pnls: list[float] = []
    for win_start, win_end in allocator_windows(oos_start, oos_end, _REBALANCE_DAYS):
        trail_start = win_start - timedelta(days=_TRAILING_DAYS)
        trailing = {}
        if trail_start >= oos_start:
            trail_mr = prices.loc[trail_start:win_start]
            trail_fa = funding.loc[trail_start:win_start]
            trailing["mean_reversion"] = (
                _segment_return(run_signal_backtest(
                    trail_mr, *mean_reversion.signals(trail_mr, **mr_params), fee=fee))
                if len(trail_mr) > 48 else None
            )
            trailing["funding_arb"] = (
                _segment_return(run_funding_backtest(trail_fa, threshold=fa_params["threshold"], fee=fee))
                if len(trail_fa) > 0 else None
            )
        else:
            trailing = {"mean_reversion": None, "funding_arb": None}
        picked = pick_strategy(trailing)

        seg_prices = prices.loc[win_start:win_end]
        if picked == "funding_arb":
            seg_data = funding.loc[win_start:win_end]
            seg_result = run_funding_backtest(seg_data, threshold=fa_params["threshold"], fee=fee)
        else:
            seg_result = run_signal_backtest(
                seg_prices, *mean_reversion.signals(seg_prices, **mr_params), fee=fee)
        composite *= 1.0 + seg_result.total_return
        segment_pnls.append(seg_result.total_return)
        equity_points[pd.Timestamp(win_end)] = composite

    equity = pd.Series(equity_points).sort_index()
    result = _metrics(equity, segment_pnls)
    start_ts = equity.index[0].to_pydatetime()
    end_ts = equity.index[-1].to_pydatetime()
    params_json = json.dumps({
        "variant": "allocator", "rebalance_days": _REBALANCE_DAYS,
        "trailing_days": _TRAILING_DAYS,
    })
    return store_result(ch_client, "allocator", symbol, "1h", params_json, "OOS",
                        start_ts, end_ts, result)


def _has_test_params(ch_client) -> bool:
    rows = ch_client.query(
        f"SELECT count() FROM {database_name()}.tuned_params FINAL WHERE startsWith(strategy, 'TEST_')"
    ).result_rows
    return rows[0][0] > 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OOS comparison: static vs tuned vs allocator")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = parser.parse_args(argv)

    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    ch_client = get_clickhouse_client()
    print(f"{'symbol':<8}{'variant':<12}{'total_return':>13}{'sharpe':>9}{'max_dd':>9}")
    for symbol in symbols:
        for strategy in ("mean_reversion", "funding_arb"):
            try:
                tuned = _tuned_params(ch_client, strategy, symbol)
                static = (dict(mean_reversion.MR_DEFAULTS) if strategy == "mean_reversion"
                          else {"threshold": 0.0001})
                for variant, params in (("static", static), ("tuned", tuned)):
                    start_ts, end_ts, result = _execute_variant(ch_client, strategy, symbol, params, args.fee)
                    params_json = json.dumps({"variant": variant, **params})
                    store_result(ch_client, strategy, symbol, "1h", params_json, "OOS",
                                 start_ts, end_ts, result)
                    print(f"{symbol:<8}{strategy+'/'+variant:<12}"
                          f"{result.total_return:>13.3f}{result.sharpe:>9.3f}{result.max_drawdown:>9.3f}")
            except Exception as exc:
                print(f"{symbol:<8}{strategy}: FAILED: {exc}")
        try:
            run_allocator(ch_client, symbol, fee=args.fee)
            print(f"{symbol:<8}allocator: done")
        except Exception as exc:
            print(f"{symbol:<8}allocator: FAILED: {exc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_allocator.py tests/test_compare.py -v`
Expected: 5 tests PASS (3 allocator + 2 compare; compare tests run real OOS backtests incl. numba JIT on first).

- [ ] **Step 7: Full suite + real runs**

Run: `python -m pytest tests/ -q`
Expected: all tests PASS (105 + 1 schema + 2 tuner + 3 allocator + 2 compare = 113).

Real tuning run (background OK — 24 strategy×symbol combos × 8 folds × 60 trials, ~10-30 min):

```bash
python -m src.meta_learning.tuner --strategy all --symbols all
```

Expected: 24 `tuned {...}` lines, no FAILED lines.

Then the comparison:

```bash
python -m src.meta_learning.compare --symbols all
```

Expected: 48 static/tuned lines + 12 allocator lines, no FAILED lines.

Verify:

```bash
curl -s --user "alpha:" "http://localhost:8123/?query=SELECT%20strategy%2C%20window%2C%20count()%20FROM%20alpha.backtest_runs%20FINAL%20WHERE%20NOT%20startsWith(strategy%2C%27TEST_%27)%20GROUP%20BY%20strategy%2C%20window%20ORDER%20BY%20strategy%2C%20window%20FORMAT%20PrettyCompact"
```

Expected: allocator OOS ×12, funding_arb OOS ×36 (Phase 3 static + static/tuned variants… note: exact counts vary by params_json variants — the check is that allocator and tuned/static rows exist for all 12 symbols), plus `tuned_params` has 24 rows:

```bash
curl -s --user "alpha:" "http://localhost:8123/?query=SELECT%20count()%20FROM%20alpha.tuned_params%20FINAL"
```

Expected: 24.

- [ ] **Step 8: Commit**

```bash
git add src/backtesting/runner.py src/meta_learning/allocator.py src/meta_learning/compare.py tests/test_allocator.py tests/test_compare.py
git commit -m "feat: add causal allocator and OOS static-vs-tuned comparison"
```

---

## Self-Review Notes

- **Spec coverage:** tuned_params schema (Task 1), walk-forward tuner with folds/objectives/CLI (Task 2), allocator pick/windows (Task 3), run_variant/run_allocator/compare CLI (Task 3), store_result extraction DRY (Task 3), real tuning + comparison runs (Task 3 Step 7). Guardrail tripwire (Task 2 code + Global Constraints). All spec sections covered.
- **Type consistency:** `walk_forward_folds() -> 8×(dt,dt,dt,dt)`, `tune(ch_client, str, str, n_trials, seed, fee) -> dict{params, train_sharpe, validation_sharpe, folds}`, `pick_strategy(dict, default) -> str`, `allocator_windows(dt, dt, 7, 30) -> list`, `run_variant(ch_client, str, str, str, dict, fee) -> str`, `run_allocator(ch_client, str, fee) -> str`, `store_result(9 args) -> str` — spelled identically in interfaces, code, and tests.
- **Test count:** 105 + 1 + 2 + 3 + 2 = 113 (Task 3 Step 7 states this).
- **Ordering dependency:** Task 3 consumes Task 2's table and the Phase 3 engine; runner extraction (Step 3) precedes compare tests.
- **Known awkwardness called out:** `run_allocator`'s `_has_test_params` probe (lets tests inject TEST_-prefixed tuned params) is test seam, documented; allocator win_rate is per-segment (not per-trade) — acceptable for the comparison table, documented in spec's allocator definition.
