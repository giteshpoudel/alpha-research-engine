# Paper Trading Implementation Plan (Iteration 1: Forward Simulation on Live Data)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forward-simulate the tuned mean-reversion strategy on live 1h market data — one independent $1.00 sleeve per symbol — persisting virtual positions, trades, and equity, and surfacing them in a read-only dashboard "Paper Trading" view.

**Architecture:** New `src/paper/` package: `executor.py` (pure state machine, no I/O), `store.py` (ClickHouse persistence), `runner.py` (CLI: `--replay` / `--once`). Reuses `mean_reversion.signals`, `backtesting.data.load_ohlcv`, tuned params, and the existing dashboard (queries/app/templates). New scheduler plist runs `--once` hourly.

**Tech Stack:** Python 3.11, clickhouse-connect, pandas, FastAPI + Jinja2 + Chart.js, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-paper-trading-design.md`

## Global Constraints

- Only **closed** bars are ever used; no lookahead. Freshness cursor is `max_ts` over `ohlcv` for the symbol/interval.
- All writes deterministic and idempotent: equity keyed by `(strategy, symbol, ts)`, trades by content-addressed `trade_id = sha256(strategy|symbol|ts|side)[:16]`, position snapshot keyed by `(strategy, symbol)`.
- Fill assumed at the **signal bar's close**; 0.1%/side (`backtesting.runner.DEFAULT_FEE`); all-in per sleeve; long-only.
- Each sleeve starts at equity `1.0`.
- Tuning stays IS-only; paper trades OOS + live data and never tunes.
- Dashboard stays read-only; binds `127.0.0.1`; `TEST_`-prefixed rows are always excluded from the UI.
- Existing 126 tests must stay green. Test style matches the suite: live ClickHouse, `TEST_`-prefixed isolation, `mutations_sync` cleanup.

---

### Task 1: Storage tables

**Files:**
- Modify: `src/ingestion/schemas.py`
- Test: `tests/test_schemas.py` (append)

**Interfaces produced (Tasks 3–5 use):** tables `paper_equity`, `paper_trades`, `paper_positions`.

- [ ] Add three `ReplacingMergeTree` DDL constants and register them in `create_clickhouse_schema`:

```python
_PAPER_EQUITY_DDL = """
CREATE TABLE IF NOT EXISTS {db}.paper_equity
(
    strategy LowCardinality(String),
    symbol String,
    ts DateTime64(3),
    equity Float64,
    cash Float64,
    position_value Float64,
    mark_price Float64,
    status LowCardinality(String),
    created_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol, ts)
"""

_PAPER_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS {db}.paper_trades
(
    trade_id String,
    strategy LowCardinality(String),
    symbol String,
    side LowCardinality(String),
    ts DateTime64(3),
    price Float64,
    size Float64,
    notional Float64,
    fee Float64,
    realized_pnl Float64,
    reason LowCardinality(String),
    created_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol, trade_id)
"""

_PAPER_POSITIONS_DDL = """
CREATE TABLE IF NOT EXISTS {db}.paper_positions
(
    strategy LowCardinality(String),
    symbol String,
    status LowCardinality(String),
    entry_price Float64,
    entry_ts Nullable(DateTime64(3)),
    size Float64,
    cash Float64,
    cost_basis Float64,
    last_bar_ts DateTime64(3),
    updated_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (strategy, symbol)
"""
```

- [ ] Append the three constants to the DDL tuple in `create_clickhouse_schema` and add a test asserting each table exists and has the expected columns (`system.columns`).
- [ ] Run `python -m src.ingestion.schemas`, then `python -m pytest tests/test_schemas.py -v`.

---

### Task 2: Tuned-param helper

**Files:**
- Create: `src/meta_learning/params.py`
- Modify: `src/meta_learning/compare.py` (`_tuned_params` delegates)
- Test: `tests/test_meta_params.py` (new)

**Interfaces produced (Task 5 uses):**
- `params.get_tuned_params(client, strategy, symbol) -> dict | None`

- [ ] Implement:

```python
"""Shared tuned-parameter lookup (strategy x symbol)."""
from __future__ import annotations
import json
from src.ingestion.schemas import database_name


def get_tuned_params(client, strategy: str, symbol: str) -> dict | None:
    rows = client.query(
        f"SELECT params_json FROM {database_name()}.tuned_params FINAL "
        "WHERE strategy = {s:String} AND symbol = {y:String} LIMIT 1",
        parameters={"s": strategy, "y": symbol},
    ).result_rows
    return json.loads(rows[0][0]) if rows else None
```

- [ ] In `compare.py`, replace the body of `_tuned_params` to raise when the helper returns `None` (preserving the existing error message and behavior).
- [ ] Test: seed a `TEST_mean_reversion`/`BTC` row, assert helper returns the parsed dict; assert `None` for a missing key; clean up.

---

### Task 3: Executor (pure state machine)

**Files:**
- Create: `src/paper/__init__.py` (empty)
- Create: `src/paper/executor.py`
- Test: `tests/test_paper_executor.py` (new, no DB)

**Interfaces produced (Tasks 4–5 use):** `SleeveState`, `Trade`, `EquityRow`, `step(...)`.

- [ ] Implement dataclasses and `step`:

```python
@dataclass
class SleeveState:
    status: str = "flat"          # "flat" | "long"
    entry_price: float = 0.0
    entry_ts: datetime | None = None
    size: float = 0.0
    cash: float = 1.0
    cost_basis: float = 0.0


def step(state, bar_ts, close, entry_sig, exit_sig, fee):
    """Advance one sleeve over one closed bar. Pure; returns (state, trade|None, equity_row)."""
    trade = None
    if state.status == "flat":
        if entry_sig:
            cost = state.cash
            fee_amt = cost * fee
            size = (cost - fee_amt) / close
            state = SleeveState("long", close, bar_ts, size, 0.0, cost)
            trade = Trade("entry", bar_ts, close, size, cost, fee_amt, 0.0, "entry")
    elif exit_sig:
        gross = state.size * close
        fee_amt = gross * fee
        proceeds = gross - fee_amt
        trade = Trade("exit", bar_ts, close, state.size, gross, fee_amt,
                      proceeds - state.cost_basis, "exit")
        state = SleeveState("flat", 0.0, None, 0.0, proceeds, 0.0)
    position_value = state.size * close if state.status == "long" else 0.0
    equity = state.cash + position_value
    return state, trade, EquityRow(bar_ts, equity, state.cash, position_value, close, state.status)
```

- [ ] Tests: entry math (`size = (cash·(1−fee))/price`, equity `cash_prev·(1−fee)`), exit realized PnL net of both fees, flat no-op, long+entry ignored, flat+exit ignored. Expected raise: none.

---

### Task 4: Store

**Files:**
- Create: `src/paper/store.py`
- Test: covered by `tests/test_paper.py` (Task 5)

**Interfaces produced (Task 5 uses):**
- `make_trade_id(strategy, symbol, ts, side) -> str`
- `load_state(client, strategy, symbol) -> SleeveState`
- `upsert_position(client, strategy, symbol, state, last_bar_ts)`
- `insert_equity(client, strategy, symbol, rows) -> int`
- `insert_trades(client, strategy, symbol, trades) -> int`
- `last_paper_ts(client, strategy, symbol) -> datetime | None`
- `latest_closed_bar_ts(client, symbol, interval="1h") -> datetime | None`

- [ ] Implement with the column tuples from Task 1. `load_state` returns a fresh `SleeveState()` (cash=1.0) when no snapshot row. All `maxOrNull` results are tz-normalized to aware UTC. `make_trade_id` uses `sha256` over `strategy|symbol|ts.isoformat()|side`.

---

### Task 5: Runner CLI (replay + forward)

**Files:**
- Create: `src/paper/runner.py`
- Create: `src/paper/__main__.py`
- Test: `tests/test_paper.py` (new, live ClickHouse)

**Interfaces produced (Tasks 6–7 use):**
- `resolve_params(client, strategy, symbol) -> dict` — tuned params, else `mean_reversion.MR_DEFAULTS`.
- `replay(client, strategy, symbol, start, end=None, fee=DEFAULT_FEE) -> int` — deterministic rebuild, starts flat at `start`.
- `step_forward(client, strategy, symbol, fee=DEFAULT_FEE) -> int` — processes closed bars after the last recorded bar; seeds from `WINDOWS["OOS"][0]` when no history exists.
- `main(argv)` — `--replay --from YYYY-MM-DD [--to YYYY-MM-DD]` or `--once [--no-topup]`; `--symbols`, `--strategy`, `--fee`.

- [ ] Implement `_run_range`: load `load_ohlcv(symbol, "1h", start=process_from − 200h, end=latest+1h)`, compute `mean_reversion.signals(prices, **params)`, iterate bars in `[process_from, end]` through `executor.step`, then batch-insert equity/trades and upsert the position snapshot. Signal warmup uses the prior 200 bars (indicator warmup only — no lookahead).
- [ ] `replay` forces `initial_state=SleeveState()`; `step_forward` loads the persisted snapshot and resumes at `last_paper_ts + 1h`.
- [ ] `--once` performs an incremental 1h OHLCV top-up first (`binance_us.backfill_ohlcv` for non-MATIC, `coinbase.backfill_matic` for MATIC, `start=now−3d`), then steps each requested symbol. Failures per symbol are logged and skipped.
- [ ] Tests:
  - `replay` over a 2-week window returns > 0 bars and writes equity/trades.
  - Run `replay` twice → identical `(ts, equity)` rows and identical trade count (idempotent, no dupes).
  - `step_forward` twice → second call returns 0.
  - `resolve_params` falls back to `MR_DEFAULTS` with no tuned row; uses a seeded `TEST_` tuned row when present.
  - Fixture deletes `TEST%` rows from all three paper tables (and seeded `tuned_params`) with `mutations_sync`.

---

### Task 6: Scheduler

**Files:**
- Create: `config/com.alpha-research.paper.plist`

- [ ] Hourly job (minute `:05`) running `.venv/bin/python -m src.paper.runner --once` with `WorkingDirectory` the repo and logs to `data/paper.log`, mirroring the existing plists. Add `data/paper.log` to nothing (already covered by `data/` in `.gitignore`).

---

### Task 7: Dashboard view

**Files:**
- Modify: `src/dashboard/queries.py` (add `db` param + paper queries)
- Modify: `src/dashboard/app.py` (paper routes; source `db` from settings)
- Modify: `src/dashboard/templates/base.html` (enable Paper Trading nav)
- Create: `src/dashboard/templates/paper.html`, `src/dashboard/templates/paper_symbol.html`
- Test: `tests/test_dashboard_paper.py` (new)

**Interfaces:**
- `queries.paper_summary(client, db=None, strategy=None) -> {"summary": {...}, "symbols": [...]}`
- `queries.paper_symbol(client, symbol, db=None, strategy=None) -> dict | None`
- `queries.paper_equity_curve(client, symbol, db=None, strategy=None, max_points=500) -> list[dict]`
- `queries.paper_trades(client, symbol, db=None, strategy=None, limit=50) -> list[dict]`

- [ ] Add module helper `_db(db) -> db or database_name()` and `db: str | None = None` to every existing query, replacing `database_name()` calls. This fixes the `APP_ENV=prod` DB-name bug: the app passes `Settings.clickhouse_db`.
- [ ] Add paper queries. `strategy=None` filters `NOT startsWith(strategy, 'TEST_')`; an explicit strategy filters exactly (for tests). `paper_summary` groups latest/first equity per symbol from `paper_equity FINAL`, joins `paper_positions FINAL`, counts trades (win = `side='exit' AND realized_pnl > 0`), and computes an equal-weight aggregate curve (`avg(equity) GROUP BY ts`) for total return + hourly Sharpe.
- [ ] Routes: `GET /paper`, `GET /paper/{symbol}` (404 when no rows), `GET /api/paper`, `GET /api/paper/{symbol}`, `GET /api/paper/{symbol}/equity`.
- [ ] Enable nav: `<a href="/paper" class="{{ 'active' if active|default('backtesting') == 'paper' }}">Paper Trading</a>`; pass `active="paper"` from paper routes.
- [ ] `paper.html`: data-freshness banner, aggregate KPI cards, per-symbol table (status, entry, mark, return, unrealized PnL, trades, last bar); `paper_symbol.html`: cards + Chart.js equity curve + recent trades table.
- [ ] Tests: `/paper` and `/api/paper` return 200 with the documented keys (empty state allowed); unknown symbol → 404; query shape test against seeded `TEST_` rows.

---

### Task 8: Docs

**Files:**
- Modify: `README.md` (roadmap: Check Phase 4/5 + Dashboard; add Paper Trading; test count)

- [ ] Update the roadmap checkboxes, the "Status" line, and test count; add a "Paper trading" run line (`python -m src.paper.runner --replay` / `--once`) to Quickstart.

---

## Verification

- `python -m pytest tests/ -v` — all green (126 existing + new).
- `python -m src.paper.runner --replay --from 2025-01-01` seeds history; `python -m src.paper.runner --once` is a no-op until the next closed bar.
- Dashboard at `127.0.0.1:8000/paper` renders positions, KPIs, and the equity chart.
