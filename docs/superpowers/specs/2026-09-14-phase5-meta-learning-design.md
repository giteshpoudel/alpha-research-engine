# Phase 5 Design — Meta-Learner & Auto-Tuning (Walk-Forward + OOS Comparison)

Date: 2026-09-14
Scope: Blueprint Phase 5 core — Optuna walk-forward parameter tuning on the IS window, and a historical out-of-sample simulation comparing static vs tuned parameters vs a causal trailing-performance allocator. Live paper trading (continuous, live data) and the multi-armed bandit with regime detection are out of scope (the allocator here is the minimal causal variant).

## Decisions (made under auto mode; user may veto)

- **Guardrail #1 is structural:** the tuner only ever loads IS-window data (2022-01-01 → 2024-12-31 23:00). The OOS comparison is a separate module that consumes tuned params and performs zero optimization.
- **Sentiment Momentum excluded from tuning** — ~2 weeks of sentiment history is statistically meaningless. It keeps its PRELIM-only status.
- **Walk-forward:** 12-month train + 3-month validate, stepped 3 months across the 36-month IS window (8 folds). Optuna (TPE sampler, fixed seed) maximizes train Sharpe per fold; final params per strategy×symbol = the fold params with the best validation Sharpe; `validation_sharpe` stored is the mean of per-fold validation Sharpes under those final params.
- **Parameters:** mean_reversion → `window` int 12–72, `z_entry` float −3.5…−1.0, `z_exit` float −0.5…0.5; funding_arb → `threshold` float 1e-5…1e-3 (log scale). 60 trials per fold per strategy×symbol.
- **Allocator variant:** at each 7-day rebalance point in the OOS window, pick per symbol the strategy (mean_reversion or funding_arb, tuned params) with the best trailing-30-day return computed causally from OOS data up to that point (before any data exists: mean_reversion). Equal-notional composite equity curve.
- **Storage:** new `tuned_params` table; comparison runs stored in the existing `backtest_runs`/`backtest_equity` with `params_json` carrying `{"variant": "static"|"tuned"|"allocator", ...}` so the dashboard keeps working unchanged.
- **Objective/metrics:** reuse `engine._metrics` semantics; tuning objective = train Sharpe (0.0-guarded as in the engine).

## New schema (in `src/ingestion/schemas.py`)

### `tuned_params`
| Column | Type |
| --- | --- |
| strategy | LowCardinality(String) |
| symbol | String |
| params_json | String |
| train_sharpe | Float32 — mean train Sharpe of the selected fold |
| validation_sharpe | Float32 — mean validation Sharpe across folds for the selected params |
| folds | UInt8 |
| tuned_at | DateTime64(3) |

Engine: `ReplacingMergeTree`, `ORDER BY (strategy, symbol)` — re-tuning replaces the row.

## New modules in `src/meta_learning/`

### `tuner.py`
- `walk_forward_folds() -> list[tuple[datetime, datetime, datetime, datetime]]` — (train_start, train_end, val_start, val_end) for the 8 folds of the IS window.
- `objective_for(strategy) -> callable` — builds an Optuna objective from trial params → train-window backtest Sharpe (uses `engine.run_signal_backtest` / `run_funding_backtest` with default fee).
- `tune(ch_client, strategy, symbol, n_trials=60, seed=42) -> dict` — runs all folds, returns `{params, train_sharpe, validation_sharpe, folds}` and upserts into `tuned_params`.
- CLI: `python -m src.meta_learning.tuner --strategy mean_reversion --symbols BTC,ETH` (defaults: both tunable strategies × all 12 symbols).

### `allocator.py`
- `pick_strategy(trailing_returns: dict[str, float], default="mean_reversion") -> str` — argmax, or default when all trailing returns are None/zero history.
- `allocator_windows(oos_start, oos_end, rebalance_days=7, trailing_days=30) -> list[tuple[datetime, datetime]]` — rebalance checkpoints.

### `compare.py`
- `run_variant(ch_client, strategy, symbol, variant, params) -> str` — runs one OOS backtest with explicit params (bypassing the Phase 3 runner's defaults) and stores into `backtest_runs`/`backtest_equity` with `variant` in params_json. Reuses `engine` directly.
- `run_allocator(ch_client, symbol) -> str` — builds the causal composite equity curve (picks per 7-day window by trailing-30-day tuned-strategy return), computes metrics, stores as strategy `allocator`.
- CLI: `python -m src.meta_learning.compare --symbols all` — for each strategy×symbol: static (existing Phase 3 OOS run) vs tuned vs allocator; prints a comparison table (strategy, symbol, variant, total_return, sharpe, max_dd, win_rate).

## Error handling

- A symbol/strategy fold that fails is logged and skipped; tuning continues.
- Tuning is idempotent: `tuned_params` replaces on (strategy, symbol); comparison run_ids are deterministic (variant is inside params_json, so static/tuned/allocator never collide).
- Tuner raises immediately if asked for data outside the IS window (assertion on fold bounds — a code-level guardrail #1 tripwire).

## Testing

- `walk_forward_folds`: exact fold count (8), non-overlapping validate windows, all bounds inside IS.
- Objective: mean_reversion on a synthetic z-score-friendly series — tuned optimum beats a bad parameter corner.
- `pick_strategy`: argmax, default on empty.
- `tune` on one symbol (BTC, reduced trials) writes a `tuned_params` row with sane values (TEST strategy name, teardown delete).
- `run_variant` stores rows whose params_json contains the variant; run_ids differ across variants for the same symbol.
- Allocator: on a synthetic two-strategy fixture where one strategy dominates the first half and the other the second, the composite picks the leader in each half (causality asserted: the pick at time T depends only on data < T).
- New dependencies: `optuna` only.

## Known limitations (documented)

- Allocator is a minimal causal trailing-return rule, not a contextual bandit; regime detection is future work.
- Tuning optimizes Sharpe only; multi-objective (Sharpe vs drawdown) is future work.
- Tuned params are per-symbol, not cross-asset; no portfolio-level position sizing yet.
