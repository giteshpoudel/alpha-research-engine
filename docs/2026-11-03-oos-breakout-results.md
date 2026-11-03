# Exploratory OOS result: vol_breakout @ 1h (2026-11-03)

Amendment: [`docs/prereg/2026-11-03-phase2-oos-exploratory.md`](prereg/2026-11-03-phase2-oos-exploratory.md)
(committed `0b44748` before this ran). Harness: `src/research/oos_breakout.py`.
Window: 2025-01-01 → 2026-11-01 (22 months), frozen 2024 parameters, run once.

## Decision: retire (by the pre-registered rule)

Max drawdown **−22.8%** breached the 20% limit. Return and Sharpe passed. The rule was
fixed before the result, so the family is retired from live use. The shadow forward test
in chorus-howl costs nothing and keeps running as data.

## Results (equal-weight BTC/ETH/SOL, net)

| Costs | Return | Sharpe | Max DD | Trades | Per trade | Win rate |
|---|---|---|---|---|---|---|
| Base (5–6 bps/side, ×3 in stress) | **+24.5%** | **0.80** | **−22.8%** | 202 | +37.9 bps | 38% |
| Costs ×2 | +14.3% | 0.52 | −25.1% | 202 | +25.2 bps | 36% |
| Kraken fees (40 bps taker) | −31.2% | −1.12 | −42.3% | 202 | −50.4 bps | 28% |
| Measured (canary: ~2.5 bps/side) | +30.4% | 0.95 | −21.5% | 202 | +44.8 bps | 39% |
| Buy-and-hold | −29.8% | −0.06 | −65.3% | – | – | – |

Exposure-matched buy-and-hold (13% time in market): −1.2%.

- **By symbol:** ETH +43.0% (Sharpe 1.06), SOL +36.9% (0.70), BTC −8.5% (−0.28).
- **By quarter:** 2025Q1 +0.0%, Q2 +4.2%, **Q3 +17.5%**, Q4 −1.7%; 2026Q1 −7.4%, Q2 −4.1%,
  **Q3 +18.2%**, Q4 (Oct) −1.3%. Two trending quarters produced all the gain.
- **Crashes:** 2025-10-10: −0.6% vs buy-and-hold −9.8%. Binance hack: 0.0% vs +0.6%.

## What it means

- Unlike in-sample (a bull market), OOS it clearly beat buy-and-hold on risk: Sharpe 0.80 vs
  −0.06, and it sidestepped most of a −65% drawdown by being out of the market 87% of the time.
- It is a classic trend follower: 38% win rate, long flat or losing stretches, gains in rare
  strong trends. Its own drawdown is the problem, and it is concentrated in 2026H1.
- Venue fees decide it: profitable at Binance.US costs, deeply negative at Kraken's 40 bps.
- BTC parameters (N=96) didn't work OOS; ETH and SOL carried the result.
- Any refinement (e.g. drop BTC, a drawdown brake) is a new hypothesis chosen after seeing
  this data. It needs a new pre-registration and forward data; it can't be validated on
  this window again.
