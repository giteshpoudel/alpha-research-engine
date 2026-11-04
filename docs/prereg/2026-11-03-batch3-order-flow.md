# Pre-registration: strategy batch 3, trade order flow (2026-11-03)

Written and committed **before** any batch-3 result is computed. No order-flow feature has
been computed on this data for this purpose. The only earlier look was a September probe
(commit 18e2e47, `src/meta_learning/microstructure.py`) on about 1,000 live trades,
predicting the next short bucket; it was too small to mean anything. Scoping: idea 7 in `docs/2026-11-03-strategy-scoping.md`.
Trials added: 16 (running total 365).

## Question

Does recent **taker order flow** (who is crossing the spread: buyers or sellers) predict
BTC, ETH and SOL returns over the next 4 or 12 hours, by enough to trade after costs?

This is the first test that uses information the batch-2 models never had. Batch 2 used
only price and volume bars and found nothing (AUC 0.50 and 0.54).

## Data

- **Order flow:** Kraken USD trades for BTC, ETH, SOL from the resumable backfill plus the
  live collector (`market_trades`, exchange `kraken:USD`). Every trade carries price, size
  and taker side. Kraken is used because it is several times more active than Binance.US
  (e.g. ~35,000 SOL trades a day vs ~2,000), so its flow is a better read of the market.
- **Prices and execution:** Binance.US USDT 1h bars (`ohlcv`), the venue we trade on.
- **Window:** 2026-08-07 00:00 UTC (first full day after a 24h warm-up) → 2026-11-03
  00:00 UTC. About 88 days.
- **Not used:** order-book depth. We only have it since 2026-11-02 (the older
  `order_book_snapshots` table has one sample per ~15 minutes, too sparse). Book features
  become a separate forward-only test once 28+ days are recorded.
- **Known data issue:** Kraken SOL trade counts in 2026-08-06 → 08-18 are about a third of
  later days (12k vs 35k/day). Features are ratios, so they are kept; the result table
  also reports the test period alone, which doesn't touch those days.

## Features (computed at each hourly close t, from trades with timestamp < t)

Signed volume: buy-taker notional minus sell-taker notional.

1. `ofi_1h`, `ofi_4h`, `ofi_12h`: signed volume ÷ total volume over the last 1, 4, 12
   hours. Range −1…+1.
2. `ofi_z_1h`, `ofi_z_4h`, `ofi_z_12h`: the same, z-scored against the coin's own trailing
   7 days of hourly values (needs ≥ 72 hours of history; earlier hours are skipped).
3. `intensity`: trade count in the last hour ÷ median hourly count of the trailing 7 days.
   Signed by the sign of `ofi_1h`, so heavy buying is positive and heavy selling negative.
4. `large_ofi_4h`: order-flow imbalance over the last 4 hours using only trades above the
   coin's trailing-7-day 95th-percentile trade size.

## Target

Return from the **open of the next hour (t+1)** to the open of hour t+1+H, H ∈ {4, 12},
on Binance.US. The one-hour delay matches how we execute (decide on the close, fill at the
next open).

## Split

- **Selection (in-sample):** 2026-08-07 → 2026-10-01 (55 days).
- **Test (out-of-sample):** 2026-10-01 → 2026-11-03 (33 days). Run **once**.
- **Forward confirmation:** 2026-11-03 → 2027-01-01, from the live collector, re-run on or
  after 2027-01-05 with everything frozen.

## Part A: is there a signal? (16 trials)

- Grid: 8 features × 2 horizons = 16 (feature, H) pairs.
- Measure: Spearman rank correlation (IC) between feature and target, pooled over the
  three coins. For significance, use **non-overlapping** samples (every H-th hour), so
  overlapping returns don't inflate the count; t-stat = IC × √(n − 2) / √(1 − IC²).
- **Selection:** in the selection period, pick the single pair with the largest |t|.
  Its sign (momentum if positive, reversal if negative) is frozen.
- **Pass (test period, selected pair only):** IC has the same sign as in selection,
  |IC| ≥ 0.03, and t ≥ 2.0 on non-overlapping samples. All 16 pairs are reported for
  information, but only the selected one counts.

## Part B: can it be traded? (only if Part A passes)

- **Rule:** long or flat, spot only. At each hourly close, enter a coin when the selected
  feature's z-score (against its trailing 7 days) is beyond +1 in the frozen direction
  (beyond −1 for a reversal signal, meaning heavy selling → buy). Hold exactly H hours,
  then exit; no new entry in that coin while holding. Weight 1/3 per coin.
- **Costs:** per side BTC 5, ETH 5, SOL 6 bps, ×3 on the most volatile 10% of hours, as in
  batches 1–2.
- **Placebo:** 1,000 runs with the same number of entries per coin at random hours and
  the same hold length.
- **Pass to watch:** test-period net return > 0, still > 0 with costs ×2, and above the
  80th percentile of the placebo runs.
- **Cap:** 33 days can't support the candidate tier (drawdown and underwater rules need
  months). The best possible outcome now is **watch**, then a shadow forward run. Candidate
  needs the forward confirmation above to pass the same rules.

## Decision

- Part A fails → **retire** trade-flow-only features at 4–12h for now; record the ICs.
- Part A passes, Part B fails → **retire as a strategy**, but keep the feature as an
  input for later work (e.g. a breakout entry filter in a future pre-registered batch).
- Both pass → **watch**: add a shadow run in chorus-howl with these exact settings.

## Things decided in advance

- No parameter, window or feature changes after seeing any result. Bugs found after
  running are fixed and both results are reported.
- If the data turns out to have a gap of more than 6 hours in any coin in the test
  period, those hours are dropped and the count is reported. No other exclusions.
- Code: `src/research/batch3.py`. Results: `docs/2026-11-03-batch3-results.md` (named by
  the day it runs) and a row in `docs/research-index.json`.

## Why this could fail (stated beforehand)

- 33 test days is a short sample; a real but small effect could miss t ≥ 2.
- Order-flow effects found in papers are strongest at seconds to minutes, where our costs
  (5–6 bps per side) are too large. Effects at 4–12 hours are weaker and less studied.
- Kraken flow may not represent the whole market.
