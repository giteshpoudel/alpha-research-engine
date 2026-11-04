# Batch 3 results: trade order flow (2026-11-03)

Pre-registration: `docs/prereg/2026-11-03-batch3-order-flow.md` (commit 88d678d).
Code: `src/research/batch3.py` (committed before running as 8aa0261). Raw output:
`data/batch3_results.json`. Trials: 16 (running total 365).

## Outcome: retire

Taker order flow on Kraken did not predict BTC, ETH or SOL returns 4 or 12 hours ahead.
Part A failed, so Part B (the trading rule) was not run, as pre-registered.

| | Feature | Horizon | IC | t | Samples |
|---|---|---|---|---|---|
| Selection (Aug 7 – Oct 1) | `ofi_z_1h`, reversal | 12h | −0.070 | −1.23 | 309 |
| Test (Oct 1 – Nov 3) | same | 12h | +0.002 | +0.03 | 195 |

Pass rule: same sign, |IC| ≥ 0.03 and t ≥ 2 in the test period. The test IC is zero with
the wrong sign. Even the best of 16 pairs in the selection period only reached t = −1.23,
so there was no clear signal to begin with.

## All 16 pairs (reported for information; only the selected pair counts)

IC is a rank correlation on non-overlapping samples (every 4th or 12th hour), pooled over
the three coins.

| Feature | H | Selection IC | t | Test IC | t |
|---|---|---|---|---|---|
| ofi_1h | 4 | +0.009 | +0.26 | −0.041 | −1.01 |
| ofi_1h | 12 | −0.056 | −0.99 | +0.004 | +0.05 |
| ofi_4h | 4 | −0.029 | −0.89 | −0.072 | −1.75 |
| ofi_4h | 12 | −0.033 | −0.60 | −0.021 | −0.29 |
| ofi_12h | 4 | −0.015 | −0.46 | +0.058 | +1.41 |
| ofi_12h | 12 | +0.048 | +0.85 | +0.054 | +0.75 |
| ofi_z_1h | 4 | +0.013 | +0.40 | −0.046 | −1.12 |
| ofi_z_1h | 12 | −0.070 | −1.23 | +0.002 | +0.03 |
| ofi_z_4h | 4 | −0.032 | −0.98 | −0.081 | −1.98 |
| ofi_z_4h | 12 | −0.035 | −0.61 | −0.028 | −0.39 |
| ofi_z_12h | 4 | −0.008 | −0.24 | +0.047 | +1.14 |
| ofi_z_12h | 12 | +0.068 | +1.18 | +0.067 | +0.93 |
| intensity | 4 | +0.006 | +0.20 | −0.022 | −0.54 |
| intensity | 12 | −0.018 | −0.31 | −0.051 | −0.70 |
| large_ofi_4h | 4 | −0.036 | −1.09 | −0.070 | −1.70 |
| large_ofi_4h | 12 | −0.056 | −0.98 | −0.022 | −0.31 |

## Observation (not a result)

The three 4-hour-imbalance features at the 4h horizon (`ofi_4h`, `ofi_z_4h`,
`large_ofi_4h`) are negative in both periods: heavy buying over 4 hours was followed by
slightly weaker returns. None is significant in selection (|t| ≤ 1.09); in the test they
reach t −1.70 to −1.98, but that is after looking at 16 pairs, and the three features are
near-copies of each other. This is not evidence of an edge. If it is worth anything, it
can only be checked on data nobody has seen: a new pre-registration, tested on the live
collector's Nov 2026 – Jan 2027 tape only.

## Data checks

- Trades in the test period: BTC 4,258,307; ETH 1,585,714; SOL 1,176,691 (Kraken).
- Gaps of more than 6 hours with no trades in the test period: none.

## Bugs fixed during the run

The first run joined no rows: the price bars come back with UTC timezone attached and the
trade hours without one, so every sample was empty (n = 0, no IC computed, no result
seen). Fixed by converting the bars to naive UTC after computing the volatility flags.
Nothing else changed.

## What this means

- Three ML/flow batches now agree: with our costs and horizons, short-term prediction
  from price, volume or trade flow isn't there to be found in 2025–26 data.
- The breakout family (trend-following on hourly bars) is still the only repeated
  survivor. The research effort goes to its forward tests and to execution costs.
- Order-book depth features stay untested until about 28 days of book data exist
  (early December).
