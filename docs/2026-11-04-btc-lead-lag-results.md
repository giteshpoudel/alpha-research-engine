# Results: BTC leads, altcoins follow (2026-11-04)

Pre-registration: `docs/prereg/2026-11-04-btc-lead-lag.md` (commit 3933143). Code:
`src/research/btc_lead_lag.py` (committed before running as 4388819). Raw output:
`data/research/btc_lead_lag.json`. Trials: 4 (running total 406). Ran once, no bugs.

## Outcome: retire

Chosen in 2022–24 (highest net Sharpe): **shock z = 2, hold 12 hours**.

| Period | Return | Sharpe | Max DD | Costs ×2 | BTC shocks | Trades | Win rate | Avg trade |
|---|---|---|---|---|---|---|---|---|
| 2022–24 (selection) | +3.0% | 0.14 | −17.9% | −15.9% | 704 | 970 | 48% | +0.06% |
| 2025–26 (test) | **+9.0%** | **0.46** | −11.9% | **−1.9%** | 423 | 482 | 46% | +0.17% |

Equal-weight buy-and-hold of the nine altcoins in 2025–26: −46.3%, max drawdown −70.6%.

Retire tests: return > 0 passed; Sharpe 0.46 < 0.5 failed; return with costs ×2 −1.9% ≤ 0
failed.

## All four settings (information only)

| Setting | 2022–24 return | Sharpe | 2025–26 return | Sharpe | Costs ×2 | Avg trade |
|---|---|---|---|---|---|---|
| z 2, hold 3 h | −13.1% | −0.62 | −8.1% | −0.70 | −17.9% | −0.14% |
| **z 2, hold 12 h** | +3.0% | 0.14 | +9.0% | 0.46 | −1.9% | +0.17% |
| z 3, hold 3 h | −4.2% | −0.30 | −1.8% | −0.40 | −4.4% | −0.13% |
| z 3, hold 12 h | +2.1% | 0.13 | +3.6% | 0.42 | +0.9% | +0.27% |

## How to read this

- **No fast catch-up.** The 3-hour holds lost in both periods and at both thresholds: after
  a BTC jump, the coins that didn't move with it don't catch up within a few hours,
  certainly not by more than the ~15 bps a round trip costs.
- **A weak slow drift.** Over 12 hours the laggards did drift up a little (+0.06% to +0.27%
  per trade after costs), consistently across settings. That is a real-looking but tiny
  effect: doubling costs wipes it out, and in 2022–24 it barely beat zero.
- **Compared with buy-and-hold** (−46%) it looks good, but only because it was in the
  market a small fraction of the time; the framework requires it to earn its keep on its
  own, and it didn't.
- The market seems to price BTC's moves into the altcoins within the same hour, at least
  on hourly bars. A faster version (minutes) would need the trade tape and lower costs
  than we have.

## Next

No shadow run. Retired.
