# Results: maker-first vs market orders, live (2026-11-04)

Pre-registration: `docs/prereg/2026-11-03-maker-execution.md`. Run: chorus-howl
`exec_test --mode live --run exec2`, 60 round trips of ~$12 on BTC, ETH and SOL
(2026-11-03 19:00 → 2026-11-04 10:09 PST, paused 22:27 → 08:06 for a bug fix, below).
The earlier run `exec` was voided (two copies ran at once) and is not counted.

## Outcome: keep market orders

| Policy | Coin | Sides | Mean shortfall (bps) | Median | Done ≤ 75 s | Filled as maker | Fell back to market |
|---|---|---|---|---|---|---|---|
| maker-first | all | 60 | **+2.89** | +0.27 | 100% | 87% | 13% |
| maker-first | BTC | 20 | +3.09 | +2.46 | 100% | 85% | 15% |
| maker-first | ETH | 20 | +2.40 | +0.45 | 100% | 85% | 15% |
| maker-first | SOL | 20 | +3.19 | −0.26 | 100% | 90% | 10% |
| market | all | 60 | **+2.39** | +2.16 | 100% | 0% | – |
| market | BTC | 20 | +1.97 | +2.01 | 100% | – | – |
| market | ETH | 20 | +2.10 | +2.11 | 100% | – | – |
| market | SOL | 20 | +3.10 | +3.03 | 100% | – | – |

Shortfall is the fill price versus the mid price when the order was decided, in bps,
fees included (positive = cost).

| Rule | Needed | Result | Pass |
|---|---|---|---|
| Maker-first saves per side | ≥ 1.0 bps | −0.51 bps (costs more) | no |
| No coin worse by > 1 bps | – | BTC worse by 1.12 bps | no |
| 90% done within 75 s | ≥ 90% | 100% | yes |

## Why maker-first lost on average

- **Usually it is cheaper.** The median maker-first side cost +0.27 bps against +2.16 for
  market orders. When the resting order fills (52 of 60 sides), the average cost is
  +1.87 bps.
- **When it misses, it costs a lot.** In 8 of 60 sides the price moved away, the order
  waited 60 s and then fell back to a market order at a worse price: +9.6 bps on average,
  up to +18.5. Those 13% of sides wipe out the savings.
- **It would be worse for the strategies.** These orders were placed at random times.
  Breakout entries happen when the price is already running up, exactly when a resting buy
  is most likely to miss (the limitation noted in the pre-registration).

## Bug during the run

At cycle 29 the run stopped itself: each market-order sell rounded down to the lot size and
left ~$0.78 of BTC behind; leftovers grew to $3.90 BTC + $1.14 ETH, counted as two open
positions and blocked a SOL buy. Fixed in chorus-howl c99d2de (sells take the run's whole
holding, own fills only); the run resumed from cycle 29 and finished with leftovers under
one lot ($0.78 BTC, $0.22 ETH, $0.07 SOL). The fix doesn't change any measurement.

## Cost of the test

Realized P&L over 60 round trips: −$0.13 (fees and price moves). Final balance check: ok.

## Next

Strategies keep market orders. Execution cost assumptions in research (2 bps fee plus
spread and slippage) stay as they are; measured market-order cost here is ~2.4 bps per side
on these three coins.
