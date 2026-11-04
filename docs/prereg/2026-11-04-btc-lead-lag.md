# Pre-registration: BTC leads, altcoins follow (2026-11-04)

Written and committed **before** this rule has been run on any data. Judged by the
three-tier framework (`docs/prereg/2026-11-03-decision-framework.md`). Trials added: 4
(running total 406).

## Question

When BTC makes a sudden large hourly move up, do the altcoins that haven't moved yet catch
up over the next hours? Every earlier test used a coin's own past prices, plus funding,
order flow or ML on them. This one trades a coin on **another coin's** move. The idea
(price discovery happens in BTC first, then spreads to the altcoins) is widely discussed for
crypto, but we haven't tested it.

## Coins

- Leader: BTC (signal only, not traded).
- Traded: ETH, SOL, XRP, DOGE, LINK, ADA, AVAX, BNB, LTC. These are all Binance.US USDT
  pairs with hourly history in our database; XRP only from 2023-07-14, and before that it
  is treated as not tradeable.

## Rule

At each hourly close t:

1. **BTC shock:** BTC's return over hour t is > z × σ, where σ is the standard deviation
   of BTC's hourly returns over the previous 720 hours (t excluded).
2. **Laggards:** every traded coin whose return over the same hour is < 0.5 × BTC's.
3. Buy each laggard at the next open, at weight 1/9 (nine fixed slots, so at most fully
   invested). Hold H hours, then sell at the open. A new shock while a coin is held
   restarts its H-hour timer. A coin that isn't a laggard isn't bought.

Grid: z ∈ {2, 3} × H ∈ {3, 12} → 4 trials.

## Costs per side

ETH 5, SOL 6, XRP 5, DOGE 7, LINK 8, ADA 8, AVAX 9, BNB 8, LTC 8 bps (the costs used
for each coin in earlier tests), ×3 on top-decile volatility hours (same stress flags as
all batches).

## Selection and evaluation

- **Selection:** 2022-01-01 → 2025-01-01. The setting with the highest net Sharpe is chosen.
  If that best setting's return in this period is ≤ 0, the idea is retired without looking
  at 2025–26.
- **Test:** 2025-01-01 → 2026-11-01 with the chosen setting only. Tier by the framework
  against equal-weight buy-and-hold of the nine traded coins: retire if return ≤ 0,
  Sharpe < 0.5 or return with costs ×2 ≤ 0; candidate needs drawdown ≤ 15%, ≤ half of
  buy-and-hold's, and ≤ 182 days underwater; watch otherwise.
- Also reported: number of shocks, trades, win rate, average trade and the test-period
  results of all four settings (for information only; the decision uses the chosen one).

## Decision

- Retire → no further work on this idea.
- Watch or candidate → a shadow run in chorus-howl with exactly the chosen setting.

## Things decided in advance

- No change to coins, thresholds (0.5 laggard ratio, 720-hour σ), grid, costs or windows
  after seeing any result. Bugs found after running are fixed and both results are
  reported.
- Code: `src/research/btc_lead_lag.py`. Results: `docs/2026-11-04-btc-lead-lag-results.md`
  and a row in `docs/research-index.json`.
