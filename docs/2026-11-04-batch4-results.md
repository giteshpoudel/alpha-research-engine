# Batch 4 results: diversifiers (2026-11-04)

Pre-registration: `docs/prereg/2026-11-04-batch4.md` (commit 72bd7b3). Code:
`src/research/batch4.py` (committed before running as 38876ea). Raw output:
`data/research/batch4.json`. Trials: 30 (running total 395).

## Outcome: all three retired

| Strategy | OOS return | Sharpe | Max drawdown | Entries | Tier |
|---|---|---|---|---|---|
| 7. Buy the dip in an uptrend | −2.6% | −0.19 | −7.6% | 20 | retire |
| 8. Buy futures panic (funding) | −10.1% | −0.02 | −38.3% | 109 | retire |
| 9. Passive market making | −$0.91 on $60 (−0.5%) | – | – | 85 fills | retire |

Buy-and-hold of the six coins over the same window (2025-01-01 → 2026-11-01): −41.6%,
max drawdown −66.5%.

## 7. Buy the dip in an uptrend

- Final setting: 50-day trend filter, 3-day drop of 1.5 normal moves, hold 7 days.
- In-sample walk-forward: +3.2%, Sharpe 0.32, positive in 3 of 8 quarters. Weak before it
  ever reached out-of-sample data.
- Out of sample it was almost never invested (4% average exposure): in a falling market
  the trend filter is mostly off. The 20 trades it did take lost 2.6% in total. Per coin,
  LINK and ETH gained and SOL, DOGE and XRP lost.
- It sat out the 2025-10-10 crash (0% vs buy-and-hold −12.8%), as a trend filter should.

## 8. Buy futures panic (funding)

- Final setting: 3-day average funding below its 10th percentile of 90 days, hold 7 days.
- In-sample walk-forward: +58.5%, Sharpe 1.44, positive in 5 of 8 quarters, almost all
  from 2024 (one quarter +25.6%).
- Out of sample: −10.1% with a −38.3% drawdown over 109 entries. Low funding in
  2025–26 often meant the market was right to be bearish. This is the classic pattern of
  an idea that worked in one bull market and not after.

## 9. Passive market making

- Every setting lost money in the selection period (Sep 29 – Oct 16), from −13% to
  −55% of capital. The tighter the quotes, the more fills and the bigger the loss:
  4,929 fills at ±10 bps lost $32.85 on $60 of capital.
- Pre-registered pick (best of the six, ±40 bps, 3 lots): −$0.91 in the test period
  (−$1.06 with the extra 2 bps per fill), profitable on 1 of 3 coins (XRP +$0.06,
  DOGE −$0.19, LINK −$0.79).
- Why: the conservative fill rule only fills us when the price trades **through** our
  quote, which is exactly when the price is moving against us (adverse selection).
  On these thin pairs that cost is larger than the spread we earn, even at 0% fees.
  A real market maker escapes this with speed and queue position, which we don't have.

## Bugs fixed during the run

The first run of strategy 9 matched no trades: trade timestamps came back in microseconds
and were converted as if nanoseconds, so every window was empty (0 fills, no result seen).
Fixed by converting to seconds independently of the unit (commit after 38876ea).
Strategies 7 and 8 were unaffected and are reported from the first run (identical on the
second).

## What this means

- Twelve ideas over four batches; the hourly breakout family is still the only one that
  survives out of sample. Dip-buying and funding signals both depend on a rising market,
  and 2025–26 wasn't one.
- Market making on Binance.US needs order-book data and fast quote updates to avoid
  adverse selection. It isn't worth a shadow run on this evidence.
- The trial count is now 395. Each further batch raises the bar for what counts as real.
