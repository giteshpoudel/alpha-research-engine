# Feedback Loop Implementation Plan (Rolling re-tune + versioned params + allocation gating)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the backtest↔paper loop guardrail-safely: monthly rolling re-tune with an embargo, versioned params, paper trading using time-appropriate params, and per-symbol on/off gating by realized trailing return.

**Spec:** design notes in this file; code anchors `src/meta_learning/tuner.py`, `src/meta_learning/params.py`, `src/paper/runner.py`, `src/ingestion/schemas.py`.

## Global Constraints
- **Guardrail #1:** at deploy `T`, tuning uses data `< T − embargo` only; everything after `T` is live OOS.
- **Idempotent** Replace keys; **no retroactive rewriting** of paper history.
- Defaults: embargo **7 days**, rolling IS **36 months**, halt when trailing-30d return **< 0**.
- Existing 153 tests stay green; TDD.

### Task 1 — Storage
- `tuned_params`: add `valid_from DateTime64(3)`, ORDER BY `(strategy, symbol, valid_from)`. Idempotent migration in `create_clickhouse_schema`: if `valid_from` missing, `RENAME` → create new → copy with `valid_from = 2025-01-01` → drop legacy.
- New `paper_controls(strategy, symbol, enabled UInt8, trailing_return Float64, as_of, updated_at)` ORDER BY `(strategy, symbol)`.
- Tests: columns/engine; migration idempotent.

### Task 2 — `src/meta_learning/params.py`
- `get_tuned_params(client, strategy, symbol, as_of=None)`; `get_tuned_param_history(client, strategy, symbol) -> list[(valid_from, params)]`.
- `compare`/`collectors` keep latest behavior.

### Task 3 — `src/meta_learning/tuner.py`
- `walk_forward_folds(is_start, is_end_exclusive)` (defaults = current constants); `tune(..., is_start, is_end_exclusive, valid_from=None)` with `valid_from` defaulting to `is_end_exclusive`; insert includes `valid_from`.

### Task 4 — `src/meta_learning/retune.py` + monthly plist
- `deploy = align_hour(now)`, `is_end = deploy − embargo`, `is_start = is_end − 36mo`; `tune(valid_from=deploy)` per symbol. CLI. `config/com.alpha-research.retune.plist` (1st, 03:30).

### Task 5 — Paper param timeline (`src/paper/runner.py`)
- Compute `mean_reversion.signals` once per param version over the full price series; per bar select the signal from the latest `valid_from <= ts` (defaults before the first). Fresh replay with one regime is bit-identical.

### Task 6 — Gating
- `executor.step(..., enabled=True)`: disabled ⇒ no entries, force-exit open longs (reason `halted`).
- `_run_range`: at each day boundary compute trailing-30d return from equity so far (+ prior stored equity) and set `enabled`; upsert `paper_controls`.

### Task 7 — Dashboard
- `paper_summary`/`paper_symbol` join `paper_controls` for `enabled` + `trailing_return`; halted badge on `/paper`.

### Task 8 — Docs
- README roadmap + ledger; tests for all of the above.

## Verification
- `python -m pytest tests/ -v` green.
- `python -m src.meta_learning.retune --symbols BTC` adds a `valid_from` row.
- Param-timeline pickup with a single regime is a no-regression change. Note:
  allocation gating **intentionally** changes paper results (halts negative
  trailing-return sleeves), so a post-gating replay differs from the prior run.
