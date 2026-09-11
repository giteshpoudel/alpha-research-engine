# Phase 3 Design — Core Backtesting Framework

Date: 2026-09-11
Scope: Blueprint Phase 3 — backtesting engine integration (VectorBT), three strategy signal functions, benchmark metrics (Sharpe, Sortino, Max Drawdown, Calmar, Win Rate), results persisted to ClickHouse for the future dashboard. No parameter optimization (Phase 5). Dashboard UI, paper trading, and live trading are out of scope.

## Decisions

- **Engine:** VectorBT (vectorized; fast across the ~500k-candle dataset; its parameter-sweep design is what Phase 5 Optuna walk-forward needs).
- **IS/OOS split:** In-Sample = 2022-01-01 → 2024-12-31; Out-of-Sample = 2025-01-01 → present (guardrail #1: strict separation).
- **Sentiment-Adjusted Momentum:** framework now, evaluation later — sentiment history only starts 2026-08-25 (pipeline start). The strategy runs only on the `PRELIM` window and every result is tagged `PRELIM`; no tuning or conclusions from it. Mean Reversion and Funding Rate Arbitrage get full IS/OOS backtests.
- **No optimization in Phase 3:** strategies run on fixed default parameters on both windows. Tuning is Phase 5, which keeps IS/OOS separation structurally clean.
- **Fees:** configurable per-strategy constant, default 0.1% per side.
- **Funding arb basis assumption:** we hold spot prices (Binance.US/Coinbase) and funding rates (Hyperliquid) but no perp price series, so spot/perp basis risk is assumed zero. Documented approximation.

## Data layer: `src/backtesting/data.py`

- `load_ohlcv(ch_client, symbol, interval="1h", start=None, end=None) -> pd.DataFrame` — FINAL read, UTC-indexed, columns open/high/low/close/volume. Prefers `binance_us`, falls back to `coinbase` (MATIC).
- `load_funding(ch_client, symbol, start=None, end=None) -> pd.Series` — funding rate by ts (exchange `hyperliquid`).
- `load_sentiment(ch_client, symbol, bucket="1h") -> pd.DataFrame` — `weighted_score`, `velocity`, `engagement_ratio` from `sentiment_metrics`.
- `window_bounds(window: str) -> tuple[datetime, datetime | None]` — `IS` → (2022-01-01, 2024-12-31 23:00); `OOS` → (2025-01-01, None); `PRELIM` → (2026-08-25, None). Strategies only ever receive pre-sliced data — window discipline is structural, not by convention.

## Strategies: `src/backtesting/strategies/`

Pure functions `(data, params) -> (entries, exits)` as pandas boolean Series aligned to the price index.

### `mean_reversion.py` (spot, long-only)
- Signal: `z = (close − rolling_mean) / rolling_std` over a 24-bar (24h) window.
- Defaults: `z_entry = −2.0`, `z_exit = 0.0`, `window = 24`. Enter long when z crosses ≤ z_entry; exit when z ≥ z_exit.

### `funding_arb.py` (market-neutral carry)
- Position: when hourly funding rate > `funding_threshold` (default 0.0001, ≈0.01%), hold short-perp/long-spot; flat otherwise.
- PnL model (not a VectorBT signal strategy): per-bar return = `funding_rate` while positioned, minus `2 × fee` on each entry and exit (two legs). Basis PnL assumed zero (documented).
- Evaluated IS/OOS starting 2023-05-12 (earliest funding data).

### `sentiment_momentum.py` (spot, long-only, PRELIM only)
- Signal: 24h return > `mom_threshold` (default 0) AND sentiment `weighted_score` > `sent_threshold` (default 0.1); exit when either fails.
- Runs only on `PRELIM`; results tagged `PRELIM`.

## Engine: `src/backtesting/engine.py`

- `run_signal_backtest(prices: pd.Series, entries, exits, fee: float) -> BacktestResult` — wraps `vectorbt.Portfolio.from_signals` (long-only, fees both sides) and extracts metrics.
- `run_funding_backtest(funding: pd.Series, threshold: float, fee: float) -> BacktestResult` — carry PnL simulation (equity curve from accumulated funding − fees).
- `BacktestResult` dataclass: `total_return, sharpe, sortino, max_drawdown, calmar, win_rate, num_trades, equity_curve: pd.Series`. Ratios annualized from 1h bars (√(24×365)); win rate = share of funding periods with positive accrual for the arb, share of profitable closed trades for signal strategies.

## Results storage (schema addition in `src/ingestion/schemas.py`)

### `backtest_runs`
| Column | Type |
| --- | --- |
| run_id | String — SHA-256[:16] of strategy+symbol+interval+params_json+window+start+end (deterministic → idempotent re-runs) |
| strategy | LowCardinality(String) |
| symbol | String |
| interval | LowCardinality(String) |
| params_json | String |
| window | LowCardinality(String) — `IS` / `OOS` / `PRELIM` |
| start_ts / end_ts | DateTime64(3) |
| total_return, sharpe, sortino, max_drawdown, calmar, win_rate | Float32 |
| num_trades | UInt32 |
| created_at | DateTime64(3) |

Engine: `ReplacingMergeTree`, `ORDER BY (strategy, symbol, window, run_id)`.

### `backtest_equity`
`run_id String, ts DateTime64(3), equity Float64` — `ReplacingMergeTree`, `ORDER BY (run_id, ts)`. Hourly equity points for dashboard curves.

## Runner: `src/backtesting/runner.py`

- CLI: `python -m src.backtesting.runner --strategy mean_reversion --symbols BTC,ETH --window IS` (also `funding_arb`, `sentiment_momentum`; `--window IS|OOS|PRELIM|ALL`).
- Computes the deterministic `run_id`, runs engine(s), inserts `backtest_runs` + `backtest_equity` rows, prints a metrics table. Re-running is a no-op content-wise (same run_id collapses).
- Default symbol set: all 12 tickers. Default window: `ALL` (IS+OOS; PRELIM only for sentiment_momentum).

## Error handling

- A symbol that fails (missing data, engine error) is logged and skipped; other symbols still run.
- All reads are FINAL with UTC-normalized indexes; strategies raise immediately on empty input rather than producing empty "results".

## Testing

- Synthetic series with known properties: z-score entries/exits at exact thresholds; funding PnL hand-computed over a 5-bar fixture; metric sanity (monotonic-up series buy-and-hold → positive return, max_dd ≈ 0; win rate 1.0 for always-winning carry).
- `run_id` determinism: same inputs → same id; different param → different id.
- Runner e2e (live ClickHouse, small real window, strategy name `TEST_*`, teardown deletes by run prefix): rows land in both tables, re-run collapses to same FINAL count.
- New dependencies: `vectorbt` (pulls numba/numpy/pandas) — install compatibility with Python 3.11/arm64 verified in the plan's first task before any code depends on it.

## Known limitations (documented)

- Funding arb assumes zero basis risk (no perp price series). A later Hyperliquid-candle integration (~6 months available) can add true basis PnL for the recent window.
- `PRELIM` sentiment results are not statistically meaningful (~2 weeks); they exist to prove the plumbing.
- Funding rates are Hyperliquid's; prices are Binance.US/Coinbase spot. Cross-venue funding arb is an approximation of the real trade.
