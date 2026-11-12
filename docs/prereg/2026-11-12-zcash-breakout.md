# Pre-registration: the breakout rule on Zcash (2026-11-12)

Written and committed **before** the breakout rule has been run on Zcash. Judged by the
three-tier framework (`docs/prereg/2026-11-03-decision-framework.md`), with an extra
hindsight guard (below). Trials added: 1 (running total 407).

## Question

Does the frozen hourly breakout rule work on Zcash (ZEC)? ZEC was picked because the user
noticed its large moves over the past year (+86% over 12 months on Binance.US, a −72% fall
from a peak in between, 42 days with a move over 10%). Picking a coin **because** it
recently moved a lot is hindsight: a trend-following rule will almost surely look good on
a coin chosen for having trended. The test is built so that this can't carry the result.

## Data

- **Prices: Coinbase ZEC-USD hourly**, 2022-01-01 → 2026-11-01. Binance.US ZEC/USDT
  history is not usable before 2026: median traded value $14–120 per hour in 2022–25, 17–42%
  of hours with zero volume, and closes more than 2% away from Coinbase in 370–1,853 hours a
  year (checked 2026-11-12, before any strategy run). Coinbase has no empty hours and
  $16K–1.5M traded per median hour.
- **Trading venue (for costs): Binance.US ZEC/USDT**, which is liquid now: $1.4M traded in
  the last 24 hours, median spread 1.7 bps over one minute of quotes, and a $25 order fills
  within 0.4 bps of the best price (measured 2026-11-12).

## Rule (frozen, no tuning)

Exactly the new-coins rule (`docs/prereg/2026-11-04-breakout-new-coins.md`):
`vol_breakout` with **N = 48, vratio = 1.0, k = 3.0, T = 48**, vol-sized at
min(1, 0.80 / annualized 720-hour realized volatility), fixed at entry; decided on an
hourly close, filled at the next open. One coin, so it runs at full weight (not 1/4).

## Costs per side

Formula from the new-coins test: 2 bps fee + half the median spread + 3 bps slippage,
rounded up = 6 bps. **Used: 8 bps**, because the spread comes from one minute of quotes,
not days of book recordings. ×3 on top-decile volatility hours (same stress flags as all
batches).

## Evaluation

- **Primary window:** 2025-01-01 → 2026-11-01, tiered by the framework against buy-and-hold
  ZEC (retire if return ≤ 0, Sharpe < 0.5, or return with costs ×2 ≤ 0; candidate needs
  drawdown ≤ 15%, ≤ half of buy-and-hold's, and ≤ 182 days underwater; watch otherwise).
- **Hindsight guard (binding, stricter than the new-coins test):** 2022-01-01 → 2025-01-01
  is before the rally that got ZEC noticed. If the rule's return there is ≤ 0, or ≤ 0 with
  costs ×2, the result is **retire**, whatever 2025–26 shows. (The new-coins test only
  capped at watch; here the coin was chosen by looking at its recent prices, so the
  earlier period has to stand on its own.)
- **Reported for information only:** each calendar year's return, entries, win rate,
  exposure, and buy-and-hold per year, so it's visible whether the result comes from one
  rally.

## Decision

- Retire → no further work on ZEC.
- Watch or candidate → a shadow run in chorus-howl on ZEC/USDT with exactly these settings,
  under the same forward rules as the other watch-tier runs (≥ 3 months of active shadow and
  ≥ 15 entries before promotion; retire at 20% ideal drawdown). Live use would need the
  user's approval of a pilot-limit change.

## Things decided in advance

- No change to the coin, data source, settings, costs or windows after seeing any result.
  Bugs found after running are fixed and both results are reported.
- Code: `src/research/zcash_breakout.py`. Results: `docs/2026-11-12-zcash-breakout-results.md`
  and a row in `docs/research-index.json`.
