# Signal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how predictive the sentiment feed is of forward crypto returns (1h + 24h buckets; 1h/4h/24h horizons), persist the results, surface them in a dashboard "Signal" view, and fix two related correctness issues.

**Architecture:** `src/research/signal_eval.py` — pure statistics functions plus data assembly and a CLI. New `signal_eval_results` table. Dashboard `queries.py`/`app.py`/`signal.html`. `load_sentiment` is corrected to index at bucket end.

**Tech Stack:** Python 3.11, pandas, numpy, clickhouse-connect, FastAPI + Jinja2. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-signal-evaluation-design.md`

## Global Constraints

- **No lookahead:** features indexed at `bucket_end = bucket_start + bucket_size`; forward returns strictly after; exact hourly close lookups; right-censored horizons skipped and counted.
- **Honesty:** every result reports `n` and an `insufficient` flag (`n_periods < 20` for IC, `< 30` for quantile spreads). Descriptive only, no causal claims.
- **Determinism:** block bootstrap uses a fixed seed; `signal_eval_results` is Replace-keyed → re-runs collapse.
- Read-only over market/sentiment data; only `signal_eval_results` is written. Existing 141 tests stay green.

---

### Task 1: Storage

**Files:** Modify `src/ingestion/schemas.py`; Test `tests/test_schemas.py`.

- [ ] Add `_SIGNAL_EVAL_DDL` and register it in `create_clickhouse_schema`:

```python
_SIGNAL_EVAL_DDL = """
CREATE TABLE IF NOT EXISTS {db}.signal_eval_results
(
    bucket_size LowCardinality(String),
    feature LowCardinality(String),
    horizon LowCardinality(String),
    method LowCardinality(String),
    value Float64,
    n UInt32,
    tstat Float64,
    insufficient UInt8,
    detail String,
    computed_at DateTime64(3)
)
ENGINE = ReplacingMergeTree
ORDER BY (bucket_size, feature, horizon, method)
"""
```

- [ ] Test columns/engine; run `python -m src.ingestion.schemas`.

---

### Task 2: Alignment fix

**Files:** Modify `src/backtesting/data.py`; Test `tests/test_backtesting_data.py`.

- [ ] Add public `BUCKET_SECONDS = {"5m": 300, "1h": 3600, "24h": 86400}`.
- [ ] In `load_sentiment`, after setting the UTC index, shift it to bucket end:
  `df.index = df.index + pd.Timedelta(seconds=BUCKET_SECONDS[bucket])`.
- [ ] Test: returned index equals the `bucket_start` values plus the bucket size (proves closed-bucket alignment); existing sentiment-momentum tests still pass.

---

### Task 3: Pure stats + assembly (`src/research/signal_eval.py`)

**Files:** Create `src/research/signal_eval.py`; Test `tests/test_signal_eval.py`.

**Interfaces:**
- `rank_ic(x, y) -> float | None`
- `pooled_ic(panel) -> dict` where `panel` has columns `ts, x, y` (per-ts cross-sectional Spearman, then mean/std/IR/t)
- `block_bootstrap_p(series, block=1, seed=0, n_boot=2000) -> float` (standard centered bootstrap test)
- `quantile_spread(values, fwd) -> dict` (`n`, per-quantile means, monotonic, spread, welch t, insufficient)
- `load_sentiment_features(client, symbol, bucket) -> DataFrame` (5 metric columns, indexed at bucket end)
- `assemble_panel(client, symbol, bucket, horizons) -> DataFrame` (adds `fwd_{h}`, `mom_24h`; skips missing exact timestamps)
- `FEATURES = ("weighted_score", "mean_score", "velocity", "engagement_ratio", "post_count")`

- [ ] Implement; tests for perfect ±1 / zero-variance (`None`), bootstrap determinism at fixed seed, quantile monotonic spread.

---

### Task 4: Evaluate + CLI

**Files:** Modify `src/research/signal_eval.py`; Test `tests/test_signal_eval.py`.

**Interfaces:**
- `evaluate(client, buckets=("1h","24h"), horizons=(1,4,24), features=FEATURES, symbols=None) -> list[dict]`
  emits method rows `coverage`, `ic_pooled`, `ic_ts`, `quantile`, `divergence` with `{bucket_size, feature, horizon, method, value, n, tstat, insufficient, detail}`.
  Block length for the pooled-IC bootstrap = `max(1, ceil(horizon_hours / bucket_hours))`.
- `run_evaluation(client, ...)` — evaluates and inserts into `signal_eval_results`.
- `main(argv)` — `--bucket 1h,24h --horizons 1h,4h,24h --symbols all --no-store`.

- [ ] Tests: `evaluate` on live data returns documented keys and marks small samples `insufficient`; `run_evaluation` persists and re-running collapses (no duplicates).

---

### Task 5: Agent signal-state fix

**Files:** Modify `src/research/collectors.py`; Test `tests/test_collectors.py`.

- [ ] Add `_params_for_symbol(client, symbol, strategy="mean_reversion")` → `get_tuned_params` else `MR_DEFAULTS`.
- [ ] `_z_state(close, params)` uses the passed params; `collect_risk_data` resolves per-symbol params.
- [ ] Test: helper returns a seeded tuned row when present, defaults otherwise (TEST_ isolation); `_z_state` honors a custom `window`/`z_entry`.

---

### Task 6: Dashboard "Signal" view

**Files:** Modify `src/dashboard/queries.py`, `src/dashboard/app.py`, `src/dashboard/templates/base.html`; Create `src/dashboard/templates/signal.html`; Test `tests/test_dashboard_signal.py`.

**Interfaces:**
- `queries.signal_results(client, db=None) -> list[dict]`

- [ ] Routes `GET /signal` (HTML) and `GET /api/signal` (JSON); template with a freshness/insufficient banner, coverage table, IC table (feature × horizon, n/t/flag), and quantile spreads.
- [ ] Add "Signal" to the nav with the active-state pattern.
- [ ] Tests: `/signal` 200 (empty state allowed); `/api/signal` is a list; nav link present.

---

### Task 7: Docs

**Files:** Modify `README.md`; append paper/signal sections to `.superpowers/sdd/progress.md` (gitignored).

- [ ] Add a "Signal evaluation" run line and roadmap entry; update the test count.

## Verification

- `python -m pytest tests/ -v` all green (141 existing + new).
- `python -m src.research.signal_eval --no-store` prints a real report; expect many `insufficient` flags (the honest baseline).
- `127.0.0.1:8000/signal` renders.
