# Results: the breakout rule on Zcash (2026-11-12)

Pre-registration: `docs/prereg/2026-11-12-zcash-breakout.md` (commit 4d2f45c). Code:
`src/research/zcash_breakout.py` (committed before running as 0f82f20; a test's float
comparison was fixed in 3589067, no code change). Raw output: `data/research/zcash_breakout.json`.
Trials: 1 (running total 407). Ran once, no bugs.

## Outcome: retire (hindsight guard failed)

The frozen rule (N 48, vratio 1.0, k 3.0, T 48, vol-sized), Coinbase ZEC-USD hourly prices,
8 bps per side:

| Window | Return | Sharpe | Max DD | Costs ×2 | Entries | Win rate | Buy-and-hold | B&H max DD |
|---|---|---|---|---|---|---|---|---|
| 2025-01 → 2026-11 (primary) | **+135.4%** | 1.34 | −26.6% | +99.3% | 97 | 40% | +2,124% | −73.1% |
| 2022-01 → 2025-01 (guard) | **−61.6%** | −0.85 | −72.2% | −70.1% | 142 | 32% | −61.6% | −92.2% |

- 2025–26 alone would have been **watch**: it passed the retire tests (return > 0, Sharpe
  1.34, +99% with costs doubled), but its drawdown (26.6%) is over the 15% candidate budget.
- The pre-registered hindsight guard requires 2022–24 to be positive, and with costs ×2.
  It lost 61.6%, so the result is **retire**.

## By year (information only)

| Year | Rule | Sharpe | Max DD | Entries | Win rate | Buy-and-hold |
|---|---|---|---|---|---|---|
| 2022 | −15.7% | −0.47 | −34.0% | 42 | 38% | −74.6% |
| 2023 | −54.8% | −2.61 | −58.1% | 53 | 23% | −27.7% |
| 2024 | +0.8% | 0.21 | −32.6% | 47 | 38% | +108.7% |
| 2025 | +34.3% | 0.93 | −26.6% | 49 | 39% | +808.6% |
| 2026 (to Nov 1) | +75.2% | 1.81 | −24.8% | 48 | 42% | +144.8% |

## How to read this

- **This is the hindsight trap the guard was built for.** ZEC was picked because of its
  last year, and the rule looks excellent there. On the three years before anyone was
  looking, the same rule lost 62%, about as much as just holding the coin.
- **2023 shows the failure mode.** ZEC fell 28% that year in sharp bursts that reversed.
  The rule bought 53 breakouts, and 77% of them lost: each spike looked like the start of a
  trend and then faded. Big moves are only profitable to a trend rule when they keep going.
- **It also misses most of a rally.** In 2025 ZEC rose ~9× and the rule made +34%, because
  it's in the market only ~12% of the time and exits on pullbacks.
- **What it says about the breakout in general:** the rule works on a coin while that coin
  is trending, and loses when the coin chops. Volatility alone isn't the edge; persistence
  is, and we can't tell in advance which coins will have it.

## Next

No shadow run. Retired.
