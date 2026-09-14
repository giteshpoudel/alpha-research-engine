# Phase 3 Backtesting Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 3 backtesting framework: data access layer over ClickHouse, three strategy signal functions, a VectorBT-based engine with benchmark metrics, results persisted to `backtest_runs`/`backtest_equity`, and a runner CLI.

**Architecture:** Pure strategy functions produce entry/exit signals; `engine.py` wraps VectorBT execution and computes metrics from equity curves (one shared metrics implementation, no VectorBT stats-API dependency); `runner.py` orchestrates windows/symbols and stores results with deterministic run IDs.

**Tech Stack:** Python 3.11, `vectorbt` (new), `pandas`/`numpy` (new, transitive), `clickhouse-connect`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-phase3-backtesting-design.md`

## Global Constraints

- **Guardrail #1 (no lookahead):** window slicing happens ONLY in `data.py`; strategies receive pre-sliced data. IS = 2022-01-01→2024-12-31 23:00 UTC, OOS = 2025-01-01→now, PRELIM = 2026-08-25→now.
- **No optimization anywhere** — strategies run on fixed default parameters only.
- **No VectorBT stats-API dependency** — metrics are computed in `engine.py` from equity curves and trade PnL lists (shared implementation for both backtest kinds). VectorBT is used only for execution simulation (`Portfolio.from_signals`).
- Funding arb is a carry PnL simulation (not a VectorBT signal backtest): per-bar return = funding rate while positioned, minus `2 × fee` on each position flip; basis risk assumed zero (documented).
- Annualization: 1h bars → `periods_per_year = 8760`. If a metric is undefined (zero variance, zero drawdown, no trades), return 0.0 — never NaN/inf in stored results.
- `sentiment_momentum` runs ONLY on the PRELIM window; every result row is tagged `PRELIM`.
- Deterministic `run_id`: SHA-256[:16] of `strategy|symbol|interval|params_json|window|start_iso|end_iso` — re-runs collapse under ReplacingMergeTree.
- Test hygiene: e2e tests use strategy names prefixed `TEST_`, capture their `run_id`, and delete both result tables `WHERE run_id = <captured>` in teardown. Never touch real rows. Synthetic-series tests must not touch the DB or network.
- The existing 73 tests must stay green. New deps: `vectorbt` only (pandas/numpy/numba are its transitive deps).
- Run tests from the repo root with the venv active: `python -m pytest tests/ -v`.

---

### Task 1: vectorbt dependency + results schema

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ingestion/schemas.py` (two DDL templates + register them)
- Create: `src/backtesting/__init__.py` (empty)
- Test: `tests/test_schemas.py` (append), `tests/test_backtesting_setup.py` (new)

**Interfaces:**
- Consumes: existing `create_clickhouse_schema(client)`, `database_name()`.
- Produces: tables `backtest_runs` and `backtest_equity` (exact columns per the spec) — Task 5 inserts into them. `vectorbt` importable in the venv.

- [ ] **Step 1: Update requirements and install**

Append to `requirements.txt`:

```
vectorbt>=0.26
plotly<6
```

(vectorbt's theme registration references plotly's removed `scattermapbox` API; plotly 6+/7+ breaks import. The controller verified this exact failure and the fix: vectorbt 1.1.0 + plotly 5.24.1 + pandas 3.0.5 + numpy 2.4.6 + numba 0.67.0 imports cleanly on this Python 3.11/arm64 venv.)

Run: `/Users/giteshpoudel/Documents/GitHub/alpha-research-engine/.venv/bin/pip install -r requirements.txt`
Expected: installs cleanly; `python -c "import vectorbt"` succeeds.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_schemas.py`:

```python
def test_backtest_results_tables(ch_client):
    rows = ch_client.query(
        f"DESCRIBE TABLE {database_name()}.backtest_runs"
    ).result_rows
    cols = {r[0]: r[1] for r in rows}
    assert cols["run_id"] == "String"
    assert cols["strategy"] == "LowCardinality(String)"
    assert cols["symbol"] == "String"
    assert cols["interval"] == "LowCardinality(String)"
    assert cols["params_json"] == "String"
    assert cols["window"] == "LowCardinality(String)"
    assert cols["start_ts"] == "DateTime64(3)"
    assert cols["end_ts"] == "DateTime64(3)"
    for metric in ("total_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"):
        assert cols[metric] == "Float32"
    assert cols["num_trades"] == "UInt32"
    assert cols["created_at"] == "DateTime64(3)"

    eq = ch_client.query(f"DESCRIBE TABLE {database_name()}.backtest_equity").result_rows
    eq_cols = {r[0]: r[1] for r in eq}
    assert eq_cols["run_id"] == "String"
    assert eq_cols["ts"] == "DateTime64(3)"
    assert eq_cols["equity"] == "Float64"
```

Create `tests/test_backtesting_setup.py`:

```python
def test_vectorbt_importable():
    import vectorbt as vbt
    assert vbt.__version__


def test_pandas_numpy_available():
    import numpy as np
    import pandas as pd
    s = pd.Series([1.0, 2.0, 3.0])
    assert float(np.nanmean(s)) == 2.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v -k backtest_results; python -m pytest tests/test_backtesting_setup.py -v`
Expected: schema test FAILs — `Code: 60 ... Table alpha.backtest_runs does not exist`. Setup tests PASS already (if vectorbt import fails, STOP and report BLOCKED with the error — do not work around it).

- [ ] **Step 4: Add the DDL to `src/ingestion/schemas.py`**

Add after `_SENTIMENT_METRICS_DDL`:

```python
# run_id is a deterministic hash of strategy+symbol+params+window+dates, so
# re-running a backtest collapses to one row (idempotent, no version column).
_BACKTEST_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.backtest_runs
(
    run_id String,
    strategy LowCardinality(String),
    symbol String,
    interval LowCardinality(String),
    params_json String,
    window LowCardinality(String),
    start_ts DateTime64(3),
    end_ts DateTime64(3),
    total_return Float32,
    sharpe Float32,
    sortino Float32,
    max_drawdown Float32,
    calmar Float32,
    win_rate Float32,
    num_trades UInt32,
    created_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol, window, run_id)
"""

_BACKTEST_EQUITY_DDL = """
CREATE TABLE IF NOT EXISTS {db}.backtest_equity
(
    run_id String,
    ts DateTime64(3),
    equity Float64
)
ENGINE = ReplacingMergeTree
ORDER BY (run_id, ts)
"""
```

Extend the DDL loop in `create_clickhouse_schema`:

```python
    for ddl in (_OHLCV_DDL, _FUNDING_RATES_DDL, _SENTIMENT_POSTS_DDL, _SENTIMENT_METRICS_DDL,
                _BACKTEST_RUNS_DDL, _BACKTEST_EQUITY_DDL):
        client.command(ddl.format(db=db))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py tests/test_backtesting_setup.py -v`
Expected: all tests PASS (21 in test_schemas + 2 setup).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/ingestion/schemas.py src/backtesting/__init__.py tests/test_schemas.py tests/test_backtesting_setup.py
git commit -m "feat: add backtest results schema and vectorbt dependency"
```

---

### Task 2: Data layer (`data.py`)

**Files:**
- Create: `src/backtesting/data.py`
- Test: `tests/test_backtesting_data.py`

**Interfaces:**
- Consumes: `database_name` from `src/ingestion/schemas.py`; live DB with real OHLCV/funding/sentiment data.
- Produces (used by Tasks 3–5):
  - `WINDOWS: dict[str, tuple[datetime, datetime | None]]` — `IS` → (2022-01-01 UTC, 2024-12-31 23:00 UTC); `OOS` → (2025-01-01 UTC, None); `PRELIM` → (2026-08-25 UTC, None)
  - `load_ohlcv(ch_client, symbol: str, interval: str = "1h", start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame` — UTC-indexed, columns `open, high, low, close, volume`, ascending
  - `load_funding(ch_client, symbol: str, start=None, end=None) -> pd.Series` — funding rate by UTC ts, name `funding_rate`
  - `load_sentiment(ch_client, symbol: str, bucket: str = "1h") -> pd.DataFrame` — columns `weighted_score, velocity, engagement_ratio` by UTC ts (`bucket_start`)
  - `slice_window(df_or_series, window: str)` — rows in `[WINDOWS[window][0], WINDOWS[window][1] or ∞)`

- [ ] **Step 1: Write the failing tests**

`tests/test_backtesting_data.py`:

```python
from datetime import datetime, timezone

import pytest

from src.backtesting.data import (
    WINDOWS,
    load_funding,
    load_ohlcv,
    load_sentiment,
    slice_window,
)
from src.ingestion.schemas import get_clickhouse_client

IS_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ch_client():
    return get_clickhouse_client()


def test_windows_definition():
    assert WINDOWS["IS"] == (
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc),
    )
    assert WINDOWS["OOS"] == (datetime(2025, 1, 1, tzinfo=timezone.utc), None)
    assert WINDOWS["PRELIM"] == (datetime(2026, 8, 25, tzinfo=timezone.utc), None)


def test_load_ohlcv_btc_is_window(ch_client):
    df = load_ohlcv(ch_client, "BTC", interval="1h",
                    start=IS_START, end=datetime(2022, 2, 1, tzinfo=timezone.utc))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 744  # 31 days * 24h
    assert df.index[0] == IS_START
    assert df.index.is_monotonic_increasing
    assert df.index.tz is not None
    first = df.iloc[0]
    assert first["close"] == pytest.approx(46670.23)  # verified reference candle


def test_load_funding_btc(ch_client):
    s = load_funding(ch_client, "BTC",
                     start=datetime(2023, 5, 12, tzinfo=timezone.utc),
                     end=datetime(2023, 5, 15, tzinfo=timezone.utc))
    assert s.name == "funding_rate"
    assert len(s) > 0
    assert s.index.tz is not None
    assert (s.abs() < 0.01).all()  # funding rates are small fractions


def test_load_sentiment_and_slice_window(ch_client):
    df = load_sentiment(ch_client, "BTC", bucket="1h")
    assert list(df.columns) == ["weighted_score", "velocity", "engagement_ratio"]
    prelim = slice_window(df, "PRELIM")
    assert len(prelim) == len(df)  # all sentiment data is in the PRELIM window
    is_slice = slice_window(df, "IS")
    assert len(is_slice) == 0  # nothing in the IS window (guardrail #1 sanity)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtesting_data.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.backtesting.data'`.

- [ ] **Step 3: Implement `src/backtesting/data.py`**

```python
"""Data access for backtesting: ClickHouse -> pandas, with window discipline.

All window slicing lives here (guardrail #1): strategies only ever receive
pre-sliced data and cannot reach outside their window by accident.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.ingestion.schemas import database_name

WINDOWS: dict[str, tuple[datetime, datetime | None]] = {
    "IS": (datetime(2022, 1, 1, tzinfo=timezone.utc),
           datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc)),
    "OOS": (datetime(2025, 1, 1, tzinfo=timezone.utc), None),
    "PRELIM": (datetime(2026, 8, 25, tzinfo=timezone.utc), None),
}


def _time_filter(start, end):
    clauses, params = [], {}
    if start is not None:
        clauses.append("ts >= {start:DateTime64(3)}")
        params["start"] = start
    if end is not None:
        clauses.append("ts < {end:DateTime64(3)}")
        params["end"] = end
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _utc_index(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    return idx


def load_ohlcv(ch_client, symbol: str, interval: str = "1h",
               start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    tf, params = _time_filter(start, end)
    params["s"] = symbol
    params["i"] = interval
    rows = ch_client.query(
        f"SELECT ts, open, high, low, close, volume FROM {database_name()}.ohlcv FINAL "
        f"WHERE symbol = {{s:String}} AND interval = {{i:String}}{tf} ORDER BY ts",
        parameters=params,
    ).result_rows
    if not rows:
        raise ValueError(f"no ohlcv data for {symbol} {interval} in [{start}, {end}]")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = _utc_index(df.pop("ts"))
    return df


def load_funding(ch_client, symbol: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.Series:
    tf, params = _time_filter(start, end)
    params["s"] = symbol
    rows = ch_client.query(
        f"SELECT ts, funding_rate FROM {database_name()}.funding_rates FINAL "
        f"WHERE symbol = {{s:String}}{tf} ORDER BY ts",
        parameters=params,
    ).result_rows
    if not rows:
        raise ValueError(f"no funding data for {symbol} in [{start}, {end}]")
    idx = _utc_index([r[0] for r in rows])
    return pd.Series([float(r[1]) for r in rows], index=idx, name="funding_rate")


def load_sentiment(ch_client, symbol: str, bucket: str = "1h") -> pd.DataFrame:
    rows = ch_client.query(
        f"SELECT bucket_start, weighted_score, velocity, engagement_ratio "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE ticker = {s:String} AND bucket_size = {b:String} ORDER BY bucket_start",
        parameters={"s": symbol, "b": bucket},
    ).result_rows
    if not rows:
        raise ValueError(f"no sentiment metrics for {symbol} {bucket}")
    df = pd.DataFrame(rows, columns=["ts", "weighted_score", "velocity", "engagement_ratio"])
    df.index = _utc_index(df.pop("ts"))
    return df


def slice_window(data, window: str):
    """Rows of a UTC-indexed DataFrame/Series within WINDOWS[window]."""
    start, end = WINDOWS[window]
    sliced = data.loc[start:]
    if end is not None:
        sliced = sliced.loc[:end]
    return sliced
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtesting_data.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/data.py tests/test_backtesting_data.py
git commit -m "feat: add backtesting data layer with window discipline"
```

---

### Task 3: Strategy signal functions (`strategies/`)

**Files:**
- Create: `src/backtesting/strategies/__init__.py` (empty)
- Create: `src/backtesting/strategies/mean_reversion.py`
- Create: `src/backtesting/strategies/sentiment_momentum.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Consumes: pandas only (no DB).
- Produces (used by Task 5):
  - `mean_reversion.MR_DEFAULTS = {"window": 24, "z_entry": -2.0, "z_exit": 0.0}` and `mean_reversion.signals(close: pd.Series, window=24, z_entry=-2.0, z_exit=0.0) -> tuple[pd.Series, pd.Series]` (entries, exits — boolean, crossing-based)
  - `sentiment_momentum.SM_DEFAULTS = {"mom_window": 24, "mom_threshold": 0.0, "sent_threshold": 0.1}` and `sentiment_momentum.signals(close: pd.Series, sentiment: pd.Series, mom_window=24, mom_threshold=0.0, sent_threshold=0.1) -> tuple[pd.Series, pd.Series]`
  (Funding arb has no signal function — it's a carry simulation in `engine.py`, Task 4.)

- [ ] **Step 1: Write the failing tests**

`tests/test_strategies.py`:

```python
import numpy as np
import pandas as pd

from src.backtesting.strategies import mean_reversion, sentiment_momentum

IDX = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")


def _series(values):
    return pd.Series(values, index=IDX[: len(values)], dtype=float)


def test_mean_reversion_crossings():
    # 24 bars at 100 -> z=0; bar 24 crashes to 80 (z very negative); then reverts
    values = [100.0] * 24 + [80.0, 81.0, 82.0, 95.0] + [100.0] * 20
    close = _series(values)
    entries, exits = mean_reversion.signals(close, window=24, z_entry=-2.0, z_exit=0.0)
    assert entries.dtype == bool and exits.dtype == bool
    assert entries.iloc[24]  # first bar below z_entry
    assert entries.sum() == 1  # crossing-based: no repeated entries while depressed
    assert exits.sum() >= 1
    assert exits.idxmax() > entries.idxmax()  # exit comes after entry


def test_mean_reversion_no_signal_in_flat_market():
    close = _series([100.0] * 48)
    entries, exits = mean_reversion.signals(close)
    assert not entries.any()
    assert not exits.any()


def test_sentiment_momentum_gating():
    close = _series([100.0] * 24 + [101.0] * 24)  # +1% momentum after 24 bars
    bullish = _series([0.5] * 48)
    bearish = _series([-0.5] * 48)
    e_bull, _ = sentiment_momentum.signals(close, bullish)
    e_bear, _ = sentiment_momentum.signals(close, bearish)
    assert e_bull.sum() == 1  # momentum + bullish sentiment -> one entry
    assert e_bull.iloc[24]
    assert not e_bear.any()  # bearish sentiment blocks the entry


def test_sentiment_momentum_exit_on_sentiment_flip():
    close = _series([100.0] * 24 + [101.0] * 24)
    sent = _series([0.5] * 30 + [-0.5] * 18)  # flips bearish at bar 30
    entries, exits = sentiment_momentum.signals(close, sent)
    assert entries.sum() == 1
    assert exits.iloc[30]  # exit exactly when sentiment flips below threshold
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.backtesting.strategies'`.

- [ ] **Step 3: Implement the strategies**

`src/backtesting/strategies/mean_reversion.py`:

```python
"""Mean reversion (spot, long-only): z-score crossing signals."""

from __future__ import annotations

import pandas as pd

MR_DEFAULTS = {"window": 24, "z_entry": -2.0, "z_exit": 0.0}


def signals(close: pd.Series, window: int = 24,
            z_entry: float = -2.0, z_exit: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Enter when z crosses to at-or-below z_entry; exit when z crosses to at-or-above z_exit.

    The ~(prev <=/>= threshold) form is used instead of prev >/< threshold so a
    NaN previous z (any all-flat rolling window gives std=0) doesn't swallow the
    crossing. Identical behavior whenever the previous z is non-NaN.
    """
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = (close - ma) / sd
    entries = (z <= z_entry) & ~(z.shift(1) <= z_entry)
    exits = (z >= z_exit) & ~(z.shift(1) >= z_exit)
    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)
```

`src/backtesting/strategies/sentiment_momentum.py`:

```python
"""Sentiment-adjusted momentum (spot, long-only): momentum gated by sentiment."""

from __future__ import annotations

import pandas as pd

SM_DEFAULTS = {"mom_window": 24, "mom_threshold": 0.0, "sent_threshold": 0.1}


def signals(close: pd.Series, sentiment: pd.Series, mom_window: int = 24,
            mom_threshold: float = 0.0, sent_threshold: float = 0.1) -> tuple[pd.Series, pd.Series]:
    """Long only when 24h momentum and sentiment both exceed their thresholds.

    Sentiment is forward-filled onto the price index (last known score, never
    future data).
    """
    mom = close.pct_change(mom_window)
    sent = sentiment.reindex(close.index).ffill()
    condition = (mom > mom_threshold) & (sent > sent_threshold)
    condition = condition.fillna(False).astype(bool)
    entries = condition & ~condition.shift(1, fill_value=False)
    exits = ~condition & condition.shift(1, fill_value=False)
    return entries, exits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/strategies/__init__.py src/backtesting/strategies/mean_reversion.py src/backtesting/strategies/sentiment_momentum.py tests/test_strategies.py
git commit -m "feat: add mean-reversion and sentiment-momentum signal functions"
```

---

### Task 4: Engine (`engine.py`)

**Files:**
- Create: `src/backtesting/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `vectorbt`, pandas/numpy.
- Produces (used by Task 5):
  - `@dataclass BacktestResult` — `total_return, sharpe, sortino, max_drawdown, calmar, win_rate: float`, `num_trades: int`, `equity_curve: pd.Series`
  - `run_signal_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series, fee: float = 0.001) -> BacktestResult`
  - `run_funding_backtest(funding: pd.Series, threshold: float = 0.0001, fee: float = 0.001) -> BacktestResult`

Metric definitions (exact): annualize with `periods_per_year = 8760`; `sharpe = mean(r)/std(r)·√ppy` (0.0 when std=0); `sortino` uses downside std (0.0 when no downside bars or their std=0); `max_drawdown = min(equity/cummax−1)` (≤ 0); `calmar = annualized_return/|max_dd|` (0.0 when max_dd=0); `win_rate` = share of profitable closed trades (0.0 when none); `num_trades` = closed-trade count (signal) or completed position count (funding).

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import run_funding_backtest, run_signal_backtest

IDX = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")


def test_signal_backtest_monotonic_up_no_trades():
    close = pd.Series(np.linspace(100, 200, 100), index=IDX)
    entries = pd.Series([True] + [False] * 99, index=IDX)
    exits = pd.Series([False] * 99 + [True], index=IDX)
    result = run_signal_backtest(close, entries, exits, fee=0.0)
    assert result.total_return == pytest.approx(1.0, rel=1e-3)  # 100 -> 200
    assert result.max_drawdown <= 0.0
    assert abs(result.max_drawdown) < 0.05
    assert result.num_trades == 1
    assert result.win_rate == 1.0
    assert len(result.equity_curve) == 100
    # metrics are finite, never NaN
    for v in (result.sharpe, result.sortino, result.calmar):
        assert np.isfinite(v)


def test_signal_backtest_losing_trade():
    close = pd.Series(np.linspace(200, 100, 100), index=IDX)
    entries = pd.Series([True] + [False] * 99, index=IDX)
    exits = pd.Series([False] * 99 + [True], index=IDX)
    result = run_signal_backtest(close, entries, exits, fee=0.0)
    assert result.total_return == pytest.approx(-0.5, rel=1e-2)
    assert result.win_rate == 0.0
    assert result.max_drawdown == pytest.approx(-0.5, abs=0.02)


def test_funding_backtest_hand_computed():
    # 10 hourly funding periods: 6 above threshold (0.001), 4 below
    rates = [0.001] * 6 + [0.00001] * 4
    funding = pd.Series(rates, index=IDX[:10])
    result = run_funding_backtest(funding, threshold=0.0005, fee=0.0)
    # positioned exactly for the 6 high bars: accrual = 6 * 0.001, zero fees
    assert result.total_return == pytest.approx(0.006, rel=1e-6)
    assert result.num_trades == 1
    assert result.win_rate == 1.0


def test_funding_backtest_fees_and_undefined_metrics():
    rates = [0.001] * 6 + [0.00001] * 4
    funding = pd.Series(rates, index=IDX[:10])
    result = run_funding_backtest(funding, threshold=0.0005, fee=0.001)
    # entry costs 2*fee, exit costs 2*fee -> total_return = 0.006 - 0.004
    assert result.total_return == pytest.approx(0.002, rel=1e-3)
    # flat funding series -> zero variance -> sharpe must be 0.0, not NaN
    flat = pd.Series([0.001] * 10, index=IDX[:10])
    flat_result = run_funding_backtest(flat, threshold=0.0005, fee=0.0)
    assert flat_result.sharpe == 0.0
    assert flat_result.num_trades == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.backtesting.engine'`.

- [ ] **Step 3: Implement `src/backtesting/engine.py`**

```python
"""Backtest engine: VectorBT execution + one shared metrics implementation.

Metrics are computed from equity curves (or an explicit per-bar returns
series for the additive funding path) and per-trade PnL lists -- never
from VectorBT's stats API, so both backtest kinds share identical math.
Annualization assumes 1h bars (8760 periods/year). Undefined metrics
(zero variance, zero drawdown, no trades) return 0.0, never NaN/inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

PERIODS_PER_YEAR = 8760  # 1h bars


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series


STD_EPS = 1e-12  # below this, std/downside-std are treated as zero


def _metrics(equity: pd.Series, trade_pnls: list[float],
             returns: pd.Series | None = None,
             total_return: float | None = None) -> BacktestResult:
    if returns is None:
        returns = equity.pct_change().dropna()
    if total_return is None:
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    std = float(returns.std(ddof=0))
    sharpe = (float(returns.mean() / std * math.sqrt(PERIODS_PER_YEAR))
              if std > STD_EPS else 0.0)
    downside = returns[returns < 0]
    dstd = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = (float(returns.mean() / dstd * math.sqrt(PERIODS_PER_YEAR))
               if dstd > STD_EPS else 0.0)

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    n = len(returns)
    if n > 0 and total_return > -1.0:
        annualized = (1.0 + total_return) ** (PERIODS_PER_YEAR / n) - 1.0
    else:
        annualized = -1.0
    calmar = float(annualized / abs(max_dd)) if max_dd < 0 else 0.0

    wins = [p > 0 for p in trade_pnls]
    win_rate = float(sum(wins) / len(wins)) if wins else 0.0
    return BacktestResult(
        total_return=total_return, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate,
        num_trades=len(trade_pnls), equity_curve=equity,
    )


def run_signal_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series,
                        fee: float = 0.001) -> BacktestResult:
    """Long-only signal backtest via VectorBT execution."""
    pf = vbt.Portfolio.from_signals(close, entries, exits, fees=fee,
                                    freq="1h", init_cash=1.0)
    equity = pf.value().rename("equity")
    equity = equity / equity.iloc[0]
    pnls = [float(p) for p in pf.trades.records_readable["PnL"]]
    return _metrics(equity, pnls)


def run_funding_backtest(funding: pd.Series, threshold: float = 0.0001,
                         fee: float = 0.001) -> BacktestResult:
    """Funding-rate carry simulation (short perp + long spot).

    Per-bar PnL = funding rate while positioned, on fixed notional; each
    position flip costs 2 * fee (two legs), with entry charged at the window
    start when the series opens positioned. The equity curve is additive,
    equity = 1 + cumsum(pnl), anchored at 1.0 one bar before the window, so
    total_return == equity.iloc[-1] / equity.iloc[0] - 1 == sum(pnl) exactly.
    Sharpe/sortino are computed from the additive per-bar PnL series (with an
    epsilon zero-variance guard), and max drawdown from the additive curve.
    Basis risk assumed zero (documented approximation).
    """
    positioned = funding > threshold
    pos = positioned.astype(int)
    flips = pos.diff().fillna(pos.iloc[0]).abs()
    cost = flips * 2.0 * fee
    pnl = funding.where(positioned, 0.0) - cost
    equity = (1.0 + pnl.cumsum()).rename("equity")
    step = (funding.index[1] - funding.index[0]) if len(funding) > 1 \
        else pd.Timedelta(hours=1)
    anchor = pd.Series([1.0], index=[funding.index[0] - step])
    equity = pd.concat([anchor, equity])

    trade_pnls: list[float] = []
    in_position = False
    accrual = 0.0
    for is_pos, bar_pnl in zip(positioned, pnl):
        if is_pos and not in_position:
            in_position = True
            accrual = float(bar_pnl)  # includes entry cost
        elif is_pos:
            accrual += float(bar_pnl)
        elif in_position:
            in_position = False
            accrual += float(bar_pnl)  # includes exit cost (bar_pnl is -cost here)
            trade_pnls.append(accrual)
    if in_position:
        trade_pnls.append(accrual)  # open position closed at window end
    return _metrics(equity, trade_pnls, returns=pnl,
                    total_return=float(pnl.sum()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 4 tests PASS. (Note: VectorBT emits numba JIT compilation on first use — the first run may take 10-60s. That's normal, not a hang.)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine.py tests/test_engine.py
git commit -m "feat: add backtest engine with shared metrics"
```

---

### Task 5: Runner CLI + results storage (`runner.py`)

**Files:**
- Create: `src/backtesting/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `data.load_ohlcv/load_funding/load_sentiment/slice_window/WINDOWS`; `engine.run_signal_backtest/run_funding_backtest/BacktestResult`; `mean_reversion.signals` + `MR_DEFAULTS`; `sentiment_momentum.signals` + `SM_DEFAULTS`; `database_name`, `get_clickhouse_client`.
- Produces:
  - `STRATEGIES: tuple[str, ...]` = `("mean_reversion", "funding_arb", "sentiment_momentum")`
  - `compute_run_id(strategy, symbol, interval, params_json, window, start_ts, end_ts) -> str`
  - `run_backtest(ch_client, strategy, symbol, window, fee=0.001) -> str` — returns run_id; inserts one `backtest_runs` row + the equity curve rows
  - `main(argv=None) -> None` — CLI: `--strategy {mean_reversion,funding_arb,sentiment_momentum,all}`, `--symbols BTC,ETH` or `all`, `--window {IS,OOS,PRELIM,ALL}`
- Runner CLI: `python -m src.backtesting.runner --strategy mean_reversion --symbols BTC,ETH --window IS`
- Window rules: `ALL` = IS + OOS; `sentiment_momentum` runs PRELIM only (even under `--window ALL` or `--window IS`); `funding_arb` starts at max(window start, 2023-05-12) since funding data starts there.

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:

```python
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


def test_main_runs_requested_strategy_only(monkeypatch, ch_client):
    seen = []
    monkeypatch.setattr("src.backtesting.runner.run_backtest",
                        lambda ch, strategy, symbol, window, fee=0.001: seen.append((strategy, symbol, window)) or "x")
    monkeypatch.setattr("src.backtesting.runner.get_clickhouse_client", lambda: ch_client)
    main(["--strategy", "mean_reversion", "--symbols", "BTC,ETH", "--window", "IS"])
    assert seen == [("mean_reversion", "BTC", "IS"), ("mean_reversion", "ETH", "IS")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.backtesting.runner'`.

- [ ] **Step 3: Implement `src/backtesting/runner.py`**

```python
"""Backtest runner: windows x symbols x strategies -> ClickHouse results.

Usage:
    python -m src.backtesting.runner --strategy mean_reversion --symbols BTC,ETH --window IS
    python -m src.backtesting.runner --strategy all --symbols all --window ALL

run_id is a deterministic hash of strategy+symbol+interval+params+window+
dates, so re-running collapses under ReplacingMergeTree (idempotent).
No optimization: strategies always run on their fixed default parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from src.backtesting.data import (
    WINDOWS,
    load_funding,
    load_ohlcv,
    load_sentiment,
    slice_window,
)
from src.backtesting.engine import (
    run_funding_backtest,
    run_signal_backtest,
)
from src.backtesting.strategies import mean_reversion, sentiment_momentum
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES

STRATEGIES = ("mean_reversion", "funding_arb", "sentiment_momentum")
DEFAULT_FEE = 0.001
FUNDING_EARLIEST = datetime(2023, 5, 12, tzinfo=timezone.utc)

_RUN_COLUMNS = (
    "run_id", "strategy", "symbol", "interval", "params_json", "window",
    "start_ts", "end_ts", "total_return", "sharpe", "sortino",
    "max_drawdown", "calmar", "win_rate", "num_trades", "created_at",
)


def compute_run_id(strategy: str, symbol: str, interval: str, params_json: str,
                   window: str, start_ts: datetime, end_ts: datetime) -> str:
    payload = "|".join([
        strategy, symbol, interval, params_json, window,
        start_ts.isoformat(), end_ts.isoformat(),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _execute(ch_client, strategy: str, symbol: str, window: str, fee: float):
    """Run one backtest; returns (params_json, start_ts, end_ts, result)."""
    base_strategy = strategy.removeprefix("TEST_")
    if base_strategy == "funding_arb":
        start, end = WINDOWS[window]
        start = max(start, FUNDING_EARLIEST)
        # Load from start unbounded, then slice: loader `end` is exclusive but
        # WINDOWS ends are inclusive candle timestamps (boundary convention).
        funding = slice_window(load_funding(ch_client, symbol, start=start), window)
        result = run_funding_backtest(funding, fee=fee)
        params = {"threshold": 0.0001}
    elif base_strategy == "sentiment_momentum":
        start, end = WINDOWS["PRELIM"]
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], "PRELIM")
        sentiment = load_sentiment(ch_client, symbol)["weighted_score"]
        entries, exits = sentiment_momentum.signals(prices, sentiment, **sentiment_momentum.SM_DEFAULTS)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
        params = dict(sentiment_momentum.SM_DEFAULTS)
        window = "PRELIM"
    else:
        start, end = WINDOWS[window]
        prices = slice_window(load_ohlcv(ch_client, symbol, start=start)["close"], window)
        entries, exits = mean_reversion.signals(prices, **mean_reversion.MR_DEFAULTS)
        result = run_signal_backtest(prices, entries, exits, fee=fee)
        params = dict(mean_reversion.MR_DEFAULTS)
    equity = result.equity_curve
    return json.dumps(params), equity.index[0].to_pydatetime(), equity.index[-1].to_pydatetime(), result, window


def run_backtest(ch_client, strategy: str, symbol: str, window: str,
                 fee: float = DEFAULT_FEE) -> str:
    """Run one (strategy, symbol, window) backtest and store results. Returns run_id."""
    params_json, start_ts, end_ts, result, window = _execute(ch_client, strategy, symbol, window, fee)
    run_id = compute_run_id(strategy, symbol, "1h", params_json, window, start_ts, end_ts)
    now = datetime.now(timezone.utc)
    db = database_name()
    ch_client.insert(
        f"{db}.backtest_runs",
        [[run_id, strategy, symbol, "1h", params_json, window, start_ts, end_ts,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests and store results")
    parser.add_argument("--strategy", default="all",
                        choices=[*STRATEGIES, "all"])
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--window", default="ALL", choices=["IS", "OOS", "PRELIM", "ALL"])
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = parser.parse_args(argv)

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    windows = ("IS", "OOS") if args.window == "ALL" else (args.window,)

    ch_client = get_clickhouse_client()
    for strategy in strategies:
        for symbol in symbols:
            for window in windows:
                if strategy == "sentiment_momentum":
                    window = "PRELIM"
                try:
                    run_id = run_backtest(ch_client, strategy, symbol, window, fee=args.fee)
                    print(f"{strategy} {symbol} {window}: done ({run_id})")
                except Exception as exc:
                    print(f"{strategy} {symbol} {window}: FAILED: {exc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: 3 tests PASS (first engine-backed run may take ~30-60s for numba JIT).

- [ ] **Step 5: Full suite + real runs**

Run: `python -m pytest tests/ -v`
Expected: all 92 tests PASS (73 existing + 2 setup + 2 schema (backtest tables) — wait: 73 + 2 (setup) + 1 (schema) + 4 (data) + 4 (strategies) + 4 (engine) + 3 (runner) = 91. Trust pytest's actual count; every test must pass.)

Then produce the real result set:

Run: `python -m src.backtesting.runner --strategy all --symbols all --window ALL`
Expected: `mean_reversion`/`funding_arb` × 12 symbols × 2 windows + `sentiment_momentum` × 12 × PRELIM = 60 `done` lines, no `FAILED` lines (numba JIT slows the first run).

Verify:

Run: `curl -s --user "alpha:" "http://localhost:8123/?query=SELECT%20strategy%2C%20window%2C%20count()%2C%20round(avg(sharpe)%2C3)%2C%20round(avg(total_return)%2C3)%20FROM%20alpha.backtest_runs%20FINAL%20GROUP%20BY%20strategy%2C%20window%20ORDER%20BY%20strategy%2C%20window%20FORMAT%20PrettyCompact"`
Expected: `funding_arb IS/OOS`, `mean_reversion IS/OOS`, `sentiment_momentum PRELIM` rows with count 12 each.

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/runner.py tests/test_runner.py
git commit -m "feat: add backtest runner CLI with results storage"
```

---

## Self-Review Notes

- **Spec coverage:** data layer with window discipline (Task 2), three strategies incl. funding carry in engine (Tasks 3–4), benchmark metrics with exact definitions (Task 4), results tables + deterministic run_id + equity storage (Tasks 1, 5), runner CLI with ALL semantics and PRELIM tagging (Task 5), no-optimization constraint (Global Constraints + runner always uses defaults), fees default 0.001 (Tasks 4–5), testing strategy incl. e2e with captured-run_id teardown (Task 5). All spec sections covered.
- **Type consistency:** `load_ohlcv/load_funding/load_sentiment/slice_window` signatures, `signals(close, ...) -> (entries, exits)`, `run_signal_backtest(close, entries, exits, fee) -> BacktestResult`, `run_funding_backtest(funding, threshold, fee) -> BacktestResult`, `compute_run_id(7 args) -> str`, `run_backtest(ch_client, strategy, symbol, window, fee) -> str`, `main(argv=None) -> None` — spelled identically in interfaces, code, and tests.
- **Test count:** 73 + 2 (setup) + 1 (schema) + 4 (data) + 4 (strategies) + 4 (engine) + 3 (runner) = 91. Task 5 Step 5's prose arithmetic is corrected here.
- **Ordering dependency:** Task 5 consumes Tasks 2–4; Task 1's tables are required by Task 5. Sequential execution required.
- **Known plan risks called out in-line:** numba JIT first-run latency (Task 4 Step 4, Task 5 Step 4); VectorBT `trades.records_readable["PnL"]` column name is verified by the engine tests against the installed version (Task 1 ensures the install exists first).
- **TEST_ prefix handling:** the runner strips `TEST_` to find strategy logic (`removeprefix`) but stores the prefixed name, keeping e2e rows identifiable and deletable.
