# Results: the breakout rule on new coins (2026-11-04)

Pre-registration: `docs/prereg/2026-11-04-breakout-new-coins.md` (commit 233c760). Code:
`src/research/breakout_new_coins.py` (committed before running as 7cff8bc). Raw output:
`data/research/breakout_new_coins.json`. Trials: 4 (running total 399). Ran once, no bugs.

## Outcome: watch. Shadow list: ADA, BNB

The frozen rule (N 48, vratio 1.0, k 3.0, T 48, vol-sized), never tuned on these coins:

| Window | Return | Sharpe | Max DD | Costs ×2 | Buy-and-hold | B&H max DD |
|---|---|---|---|---|---|---|
| 2025-01 → 2026-11 (primary) | **+18.3%** | 0.62 | −25.2% | +0.5% | −50.5% | −72.4% |
| 2022-01 → 2025-01 (check) | −1.6% | 0.06 | −28.4% | −25.5% | −11.0% | −75.5% |

- Retire tests passed: return > 0, Sharpe 0.62 ≥ 0.5, still positive with costs doubled
  (+0.5%, barely).
- Not candidate: drawdown 25.2% is over the 15% budget, and underwater 386 days is over
  182. The 2022–24 return is negative, which by itself would also have capped it at watch.

## Per coin (each at full weight)

| Coin | 2025–26 | Costs ×2 | Max DD | Entries | 2022–24 |
|---|---|---|---|---|---|
| ADA | +39.1% | +18.6% | −28.7% | 94 | +92.4% |
| BNB | +32.1% | +13.9% | −26.2% | 86 | −30.1% |
| LTC | +2.1% | −14.1% | −39.0% | 85 | −72.0% |
| AVAX | −10.1% | −24.4% | −39.9% | 96 | +88.0% |

Shadow list rule (2025–26 return > 0 and > 0 with costs ×2): **ADA and BNB**.

## How to read this

- **For the breakout being general:** on four coins it never saw, it made +18% while the
  coins lost half their value, with about a third of buy-and-hold's drawdown. That is the
  same shape as the original six coins (breakout positive, buy-and-hold deeply negative).
- **Against:** the coin-level results flip between periods. AVAX was +88% in 2022–24 and
  −10% after; BNB the reverse. Only ADA was positive in both. Costs ×2 leave almost
  nothing. So the effect is real enough to pass the bar, but weak and noisy per coin. It
  looks like a modest market-wide tendency for strong hourly moves to continue, not a
  reliable per-coin edge.
- **Practical meaning:** this supports running the breakout across many coins at small
  size (diversification across coins is what makes a noisy edge usable), rather than
  betting heavily on any single coin.

## Next (as pre-registered)

Shadow run in chorus-howl for ADA and BNB with exactly these settings. Forward data
decides promotion, as for the other breakout runs.
