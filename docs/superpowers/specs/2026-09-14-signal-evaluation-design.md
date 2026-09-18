# Sentiment Signal Evaluation Design

Date: 2026-09-14
Scope: Measure how predictive the social/news sentiment feed is of forward crypto returns, over the 1h and 24h buckets, with a reusable evaluator. Outputs: CLI report, persisted `signal_eval_results`, and a dashboard "Signal" view. Also fixes two related correctness issues (sentiment lookahead, agent signal-state params).

Authorized by user decisions this session:
- Buckets: **1h + 24h** (5m excluded as too noisy).
- Forward horizons: **1h, 4h, 24h**.
- Outputs: **CLI report + DB table + dashboard tab**.
- Related fixes: **included** (sentiment alignment + agent signal-state params).

## Problem & honest framing

Current data is thin: `sentiment_metrics` has **90 rows / 11 tickers for 1h** (since 2026-09-02) and **38 rows for 24h** (since 2026-08-25); sources are stocktwits (1,368 posts), news (310), reddit (400, mostly blocked). This is enough to build and run the harness, **not** enough for a confident verdict. The deliverable is a reusable evaluator that produces an evidence-based baseline and sharpens as data accrues — every result carries its sample size and an `insufficient` flag; no causal claims. It is the measurement layer that later lets us decide whether to pay for Twitter/X.

## Central correctness point: no lookahead

A metric row `(bucket_size, bucket_start, ticker)` is computed from posts spanning `[bucket_start, bucket_start + size)` and is therefore only **knowable at `bucket_end = bucket_start + size`**. All evaluation aligns signals at `bucket_end` and computes forward returns *after* it.

- `fwd_ret_h(t) = close(t + h) / close(t) - 1` using the symbol's 1h OHLCV close, with `t = bucket_end`.
- OHLCV is looked up by exact hourly timestamp; a missing bar at `t` or `t+h` drops that observation (counted as skipped).
- **Existing bug this fixes:** `load_sentiment` indexes at `bucket_start`, and `sentiment_momentum.signals` (`src/backtesting/strategies/sentiment_momentum.py:18`) forward-fills onto the price index — so at price bar `H` it can see sentiment from the not-yet-closed `[H, H+1h)` bucket (a one-bucket leak). `load_sentiment` is changed to index at **bucket end**.

## Features evaluated

From `sentiment_metrics`: `weighted_score` (primary — the strategy's input), `mean_score`, `velocity`, `engagement_ratio`, `post_count`. Also a raw-post check: Spearman of `sentiment_posts.sentiment_score` vs the symbol's forward return (sanity/attribution of the embedding scorer).

## Metrics

1. **Coverage** — per source and per bucket: row counts, distinct tickers, date range, posts/day, scored fraction (§ data-quality baseline).
2. **Pooled cross-sectional IC** (headline) — at each `bucket_end`, Spearman-rank the feature across tickers and correlate with rank of forward returns; the mean and std of that per-time IC series give `IC`, `IR = mean/std`, and `t ≈ IR·√n`. Signature via a **seeded circular block bootstrap** (fixed seed, block length = horizon/period) because overlap induces autocorrelation.
3. **Per-ticker time-series IC** — Spearman over time per ticker, then mean IC, % positive, and count across tickers.
4. **Quantile event study** — pool observations, split the feature into quantiles (deciles → quintiles → terciles depending on n), report mean forward return per bucket, monotonicity, and top-minus-bottom spread with a block-bootstrap p-value.
5. **Sentiment vs momentum** — corr(feature, trailing 24h return) and forward returns conditional on agree/disagree.

Small-sample discipline: every row reports `n`; results below thresholds (`n_periods < 20` for IC, `< 30`/quantile) are flagged `insufficient` and rendered as such.

## Storage

New `signal_eval_results` table (`src/ingestion/schemas.py`), ReplacingMergeTree keyed so re-running with more data replaces prior rows:

```
bucket_size LowCardinality(String),
feature LowCardinality(String),
horizon LowCardinality(String),     -- '1h' | '4h' | '24h'
method LowCardinality(String),      -- 'coverage' | 'ic_pooled' | 'ic_ts' | 'quantile' | 'divergence'
value Float64,                      -- headline number (IC, spread, ...)
n UInt32,
tstat Float64,                      -- 0 when not applicable
insufficient UInt8,
detail String,                      -- JSON blob: per-ticker ICs, quantile means, skipped counts
computed_at DateTime64(3)
ORDER BY (bucket_size, feature, horizon, method)
```

## Architecture: `src/research/signal_eval.py`

- **Pure functions** (no I/O, unit-tested): `rank_ic(x, y)`, `pooled_ic(panel)`, `block_bootstrap_p(series, block, seed)`, `quantile_spread(values, fwd, q)`, `coverage_stats(...)`.
- **Data assembly**: `load_features(ch_client, ticker, bucket)` (all metric columns indexed at `bucket_end`), reuse `load_ohlcv` for closes; exact-timestamp forward-return join.
- `evaluate(ch_client, buckets, horizons, features, symbols)` → list of result dicts; `run_evaluation(...)` persists to `signal_eval_results`.
- `main(argv)` CLI: `python -m src.research.signal_eval [--bucket 1h,24h] [--horizons 1h,4h,24h] [--symbols all] [--no-store]` prints coverage, pooled IC (with n/t/insufficient), per-ticker IC summary, and quantile spreads.

## Dashboard

- `queries.signal_results(client, db=None)` (latest rows) + `queries.signal_coverage(client, db=None)`.
- Routes `GET /signal` (HTML) and `GET /api/signal` (JSON).
- `templates/signal.html`: prominent data-freshness / "insufficient sample" banner; coverage cards; IC table (feature × horizon × bucket, showing n, IC, t, flag); quantile spread table. Enable the new "Signal" nav link (`base.html`).
- Nav active-state extended for `signal`.

## Related fixes

1. **Sentiment lookahead** — `src/backtesting/data.py:load_sentiment` indexes at `bucket_end` (add a `_BUCKET_SECONDS` delta map). Verified by test; `sentiment_momentum` then sees only closed buckets. (Changes PRELIM backtest numbers — expected and correct.)
2. **Agent signal states** — `src/research/collectors.py:collect_risk_data` currently computes z-states with `MR_DEFAULTS`. It will resolve per-symbol params via `meta_learning.params.get_tuned_params`, falling back to `MR_DEFAULTS`. Extract a small testable helper (`_params_for_symbol`) so it can be unit-tested without touching production tuned rows.

## Testing (`tests/test_signal_eval.py`, plus additions)

- Pure functions: `rank_ic` on known arrays (perfect ±1, zero-variance → 0/None), `quantile_spread` monotonic case, `block_bootstrap_p` deterministic for a fixed seed and ~uniform for random noise.
- Alignment: `load_sentiment` index equals `bucket_start + bucket`; `sentiment_momentum` never uses a bucket not yet closed.
- Schema: `signal_eval_results` columns/engine.
- Integration: `evaluate` on live data runs, returns rows with the documented keys, and marks expected `insufficient` when small; `run_evaluation` persists and re-running collapses (no dupes).
- Dashboard: `/signal` renders 200 (empty state allowed); `/api/signal` shape.
- Collectors: `_params_for_symbol` returns tuned params when a row exists, defaults otherwise (TEST_ isolation).

## Non-goals

- No Twitter/X integration (a later decision gated on these results).
- No strategy roster changes beyond the alignment fix; no re-tuning; no feedback loop.
- No 5m bucket; no causal/inference claims beyond sample-limited description.
- No order-of-magnitude compute: 1h/24h over ~12 tickers is trivial.

## Guardrails honored

- No lookahead: signals at `bucket_end`, forward returns strictly after.
- Descriptive only; sample sizes and uncertainty reported, not hidden.
- Read-only over market/sentiment data; the only write is `signal_eval_results`.
- Tuning stays IS-only; evaluation is on OOS/live data and never tunes.
