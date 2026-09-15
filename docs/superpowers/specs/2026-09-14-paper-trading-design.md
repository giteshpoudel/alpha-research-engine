# Paper Trading Design — Forward Simulation on Live Data

Date: 2026-09-14
Scope: Iteration 1 of paper trading. Forward-simulate the **tuned mean-reversion** strategy on live market data (1h bars), one **independent $1.00 sleeve per symbol**, persisting virtual positions, trades, and equity. Expose a read-only dashboard "Paper Trading" view. No live exchange execution, no order routing, no shorting.

Authorized by user decisions this session:
- Cadence: **hourly forward sim** on closed 1h bars.
- Strategy: **tuned mean reversion only** (funding arb retired; sentiment history thin).
- Capital model: **independent $1.00 sleeve per symbol** (mirrors the backtest engine's per strategy×symbol unit equity).

## Decisions

- **Execution timing:** a signal is evaluated on a bar's close and the virtual fill is assumed at that same bar's close. This matches VectorBT's `from_signals` close-fill used by the Phase 3/5 backtests, so paper numbers stay comparable to the `compare.py` tuned OOS curve. Documented approximation: live fills happen minutes after close at a slightly different price and with slippage; both are ignored (fee only).
- **Fees:** 0.1% per side (`DEFAULT_FEE`), same as backtesting.
- **Sizing:** all-in per sleeve. Entry invests all cash (`size = cash·(1−fee)/price`, cash→0); exit returns `size·price·(1−fee)` to cash. Equity = cash + size·mark_price.
- **Signal source:** `src/backtesting/strategies/mean_reversion.signals` (unchanged), evaluated on a rolling window of the most recent closed 1h closes (window = `max(200, tuned_window + 5)` bars).
- **Parameters:** latest tuned params per symbol from `tuned_params` (`strategy='mean_reversion'`, `LIMIT 1 FINAL`), falling back to `MR_DEFAULTS` when absent. No re-tuning in this iteration.
- **No lookahead:** only **closed** bars are ever used. The freshness cursor is `max_ts` over `ohlcv` (already closed-candle-only by ingestion).
- **Idempotency:** every write is deterministic. Equity rows are keyed by `(strategy, symbol, ts)`; trades by a content-addressed `trade_id = sha256(strategy|symbol|ts|side)[:16]`. Re-running replay or a forward step is a no-op.
- **New dependency:** none.

## Architecture: `src/paper/`

- `models.py` — dataclasses: `SleeveState(status, entry_price, entry_ts, size, cash)`, `Trade`, `EquityRow`.
- `executor.py` — pure state machine, no I/O:
  `step(state, bar_ts, close, entry_sig, exit_sig, fee) -> (new_state, trade|None, equity_row)`.
  - flat + `entry_sig` → long at `close`; long + `exit_sig` → flat; otherwise mark to market.
  - Entry/exit ignored if the opposite state already holds (matches VectorBT `from_signals` semantics).
- `store.py` — ClickHouse persistence (all `FINAL` reads, parameterized writes): `load_state`, `upsert_state`, `insert_trade`, `insert_equity`, `last_paper_ts`, `load_recent_closes`.
- `runner.py` — CLI + orchestration:
  - `python -m src.paper.runner --replay --from 2025-01-01 [--to YYYY-MM-DD]` — deterministic rebuild over historical bars (seeds the dashboard immediately and self-checks against `compare.py` tuned results).
  - `python -m src.paper.runner --once [--symbols ...]` — incremental top-up + step every closed bar after each sleeve's last processed bar.
  - `--no-topup` skips the incremental price fetch.
  - `__main__.py` — `python -m src.paper`.
- **Tuned-param helper extraction:** the private `_tuned_params` in `src/meta_learning/compare.py:34` is promoted to `src/meta_learning/params.py::get_tuned_params(client, strategy, symbol)` and reused by both `compare.py` and `paper`.
- **Reuse:** `mean_reversion.signals`, `backtesting.data.load_ohlcv` / windows, `ingestion.schemas.get_clickhouse_client` / `database_name`, `ingestion.backfill` incremental functions for the hourly OHLCV top-up.

## Storage (new tables in `src/ingestion/schemas.py`)

All `ReplacingMergeTree`, consistent with existing tables.

| Table | Columns | ORDER BY |
|---|---|---|
| `paper_equity` | strategy, symbol, ts, equity, cash, position_value, mark_price, status, created_at | (strategy, symbol, ts) |
| `paper_trades` | trade_id, strategy, symbol, side, ts, price, size, notional, fee, realized_pnl, reason, created_at | (strategy, symbol, trade_id) |
| `paper_positions` | strategy, symbol, status, entry_price, entry_ts, size, cash, last_bar_ts, updated_at | (strategy, symbol) |

`paper_positions` is a convenience snapshot of current state; it is fully derivable from `paper_trades`. DDL is idempotent (`CREATE TABLE IF NOT EXISTS`).

## Cadence & scheduling

- New launchd plist `config/com.alpha-research.paper.plist`: hourly at minute `:05`, runs `python -m src.paper.runner --once`.
- `--once` first performs an incremental 1h OHLCV top-up (reusing `src/ingestion/backfill`), then steps. This keeps marks fresh without a second scheduler; the existing daily 06:17 backfill remains the catch-up/daily+bars job.
- Data reality documented: bars are 1h only; marks can be at most ~1h old. No sub-hour paper marks in this iteration.

## Dashboard (`src/dashboard/`)

- New queries in `queries.py` (read-only, `FINAL`): `paper_overview`, `paper_symbol`, `paper_equity_curve`, `paper_trades`, `paper_freshness`.
- Routes in `app.py`:
  - `GET /paper` (HTML overview) and `GET /paper/{symbol}` (HTML detail)
  - `GET /api/paper`, `GET /api/paper/{symbol}`, `GET /api/paper/{symbol}/equity`
- Template `paper.html`: per-symbol cards (status, entry, mark, unrealized PnL, equity, last bar) + aggregate KPIs (equal-weight return, paper Sharpe, # trades, win rate) + equity chart + recent trades table + a data-freshness banner ("last closed bar …"). Chart.js as in existing views.
- Enable the disabled nav slot `src/dashboard/templates/base.html:15` → link to `/paper`.
- **Fixes the tracked `APP_ENV=prod` bug** (`progress.md:65`): dashboard queries take an explicit `db` argument sourced from `Settings.clickhouse_db` (dev → `CLICKHOUSE_DB`, prod → `PROD_CLICKHOUSE_DB`), instead of calling the unprefixed `database_name()`.
- Scope note: paper PnL is explicit in this iteration; no fake live-trading data.

## Testing (`tests/test_paper.py`, live ClickHouse like the rest of the suite)

- Executor state machine: flat→long→flat transitions, fee math, PnL, opposite-signal ignored.
- Replay determinism: same inputs → identical `paper_equity`/`paper_trades`; re-run produces no duplicates.
- Forward step idempotency: re-running `--once` on the same closed bar writes nothing new.
- Tuned-param fallback to `MR_DEFAULTS` when `tuned_params` has no row.
- Dashboard: `/paper` renders 200 with expected markers; `/api/paper` shape; unknown symbol → 404.

## Non-goals this iteration

- No live exchange execution, keys, or order routing.
- No slippage/order-book model; fee-only.
- No shorting, leverage, or portfolio-level risk sizing.
- No multi-strategy allocation (allocator stays in `meta_learning`).
- No automated re-tuning, no websockets/real-time refresh, no auth (binds 127.0.0.1).
- No sub-hourly bars or live trade prints.

## Guardrails honored

- Strict IS/OOS separation: tuning stays IS-only; paper trades OOS + live data and never tunes.
- Publication-time semantics irrelevant here (price-only strategy).
- Idempotent, re-runnable writes.
- Local-only; no push to GitHub unless explicitly requested.
