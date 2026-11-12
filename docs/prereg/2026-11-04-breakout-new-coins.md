# Pre-registration: the breakout rule on new coins (2026-11-04)

Written and committed **before** the breakout rule has been run on any of these coins.
Judged by the three-tier framework (`docs/prereg/2026-11-03-decision-framework.md`).
Trials added: 4 (running total 399).

## Question

Is the hourly volatility breakout a general effect, or did it only work on the six coins
it was designed and tuned on? Running it **unchanged** on coins it has never seen is the
cleanest out-of-sample test we can do: no parameter was fit to them.

## Coins

ADA, AVAX, BNB, LTC: every Binance.US USDT pair with full hourly history since 2022 that
isn't already in the breakout family.

- **Excluded before results: DOT.** It trades about $800 an hour on Binance.US with a
  ~32 bps spread (measured 2026-11-04), so it isn't tradeable at our size and costs.
- Not considered: MATIC, PEPE, SHIB (no Binance.US hourly history in our database).

## Rule (frozen, no tuning)

- Entry/exit: `vol_breakout` in `src/research/intraday_strategies.py` with one setting
  for all four coins: **N = 48, vratio = 1.0, k = 3.0, T = 48**. This is the most common
  value of each parameter across the six existing frozen selections (k = 3.0 in 5 of 6,
  vratio = 1.0 in 4 of 6, N = 48 in 3 of 6, T = 48 in 3 of 6). It equals LINK's selection.
- Sizing: as the vol-sized breakout, (1/4) × min(1, 0.80 / annualized 720-hour realized
  volatility), fixed at entry.
- Decided on an hourly close, filled at the next open.

## Costs per side

Formula: 2 bps taker fee + half the median spread + 3 bps slippage, rounded up. It
reproduces the existing costs (DOGE 7, SOL 6, LINK 7–8). Spreads from our book recordings
since 2026-11-02 (ADA 5.8, AVAX 7.0, LTC 5.5 bps median) and a live quote for BNB
(4.1 bps):

| ADA | AVAX | BNB | LTC |
|---|---|---|---|
| 8 bps | 9 bps | 8 bps | 8 bps |

×3 on the top-decile volatility hours (same stress flags as all batches).

## Evaluation

- **Primary window:** 2025-01-01 → 2026-11-01 (the same out-of-sample window as every
  other test), equal-weight portfolio of the four coins. Tier by the framework, against
  buy-and-hold of the same four coins.
- **Consistency check:** 2022-01-01 → 2025-01-01 is also out-of-sample for these coins
  (nothing was fit to them), but the parameter values were chosen on other coins in that
  period. Rule: if the portfolio's 2022–24 return is ≤ 0, the result is capped at **watch**.
- **Per coin:** a coin goes on the shadow list if its own 2025–26 net return is > 0 and
  still > 0 with costs ×2, **and** the portfolio reaches at least watch.

## Decision

- Portfolio retire → the breakout is likely specific to the original coins; no new
  shadow runs.
- Portfolio watch or candidate → a shadow run in chorus-howl for the coins on the shadow
  list, with exactly these settings. Live use still needs the forward checks.

## Things decided in advance

- No change to coins, settings, costs or windows after seeing any result. Bugs found after
  running are fixed and both results are reported.
- Code: `src/research/breakout_new_coins.py`. Results:
  `docs/2026-11-04-breakout-new-coins-results.md` and a row in `docs/research-index.json`.

## Note added 2026-11-12: internet outage

From 2026-11-09 06:21 to 2026-11-12 05:25 UTC the Mac running the shadow runs had no
internet. All five shadow runs tripped their kill switch on stale market data at 06:25 UTC
and stayed paused until the user approved clearing them on 2026-11-12 22:27 UTC (about 3.7
days). Nothing was held, so nothing was sold. How it's handled, decided after the fact and
on the cautious side:

- The trip isn't counted as a "kill-switch trip caused by the strategy or the loop", which
  is the wording above; trips for any other reason still count.
- The paused days don't count toward the minimum shadow period (3 weeks, or 3 months for
  watch-tier runs). Entries the ideal backtest takes while a run is paused have no shadow
  fill to compare against.
- Order-book data from those 3 days is missing and can't be refilled; the order-book test
  waits about 3 days longer.
