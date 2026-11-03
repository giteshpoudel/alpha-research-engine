# Strategy batch 1 results (2026-11-03)

Pre-registration: [`docs/prereg/2026-11-03-batch1.md`](prereg/2026-11-03-batch1.md) (`0355509`,
committed before this ran). Harness: `src/research/batch1.py`. OOS (2025-01 → 2026-11) run once.
Trial count now 337.

## Outcome

| Strategy | OOS return | Sharpe | Max DD | Longest underwater | Costs ×2 | Tier |
|---|---|---|---|---|---|---|
| Daily trend, vol-scaled | +5.5% | 0.24 | −32.7% | 466 d | +2.2% | **retire** (Sharpe) |
| Momentum rotation | −37.9% | −0.36 | −48.7% | 648 d | −40.9% | **retire** |
| Vol-sized breakout (6 coins) | **+26.9%** | **0.90** | −20.1% | 343 d | +12.8% | **watch** |
| Trend + funding filter | +4.7% | 0.22 | −32.7% | 466 d | +1.4% | **retire** (Sharpe) |
| Buy-and-hold, 6 coins | −41.6% | −0.13 | −66.5% | – | – | – |

The vol-sized breakout would be "watch" either way: it misses the candidate checks on budget
drawdown (20.1% > 15%) and underwater time (343 > 182 days), and it was capped at watch in
advance because its signals had been seen out of sample before.

## Details

- **Trend** (final L=20 days, vol target 40%): strong in-sample (+116%, Sharpe 1.62,
  5/8 folds) but almost flat OOS. It still lost far less than holding (exposure-matched
  buy-and-hold −7.1%) and sat out the Oct 10 crash (−1.4% vs −12.8%).
- **Rotation** (final L=7 days, top 3): in-sample +331% (Sharpe 1.53), OOS −38%. Short
  lookback rotation chased SOL and LINK into their declines. Clear overfit to the 2023–24 bull run.
- **Vol-sized breakout** (final target 80%): in-sample Sharpe 1.87, max DD −11.8%, 6/8 folds.
  OOS +26.9% with contributions from ETH, SOL, XRP, DOGE; LINK slightly negative, BTC flat.
  Volatility sizing cut the OOS drawdown from −22.8% (v1) and −26.5% (alt-coin) to −20.1%
  while keeping the return.
- **Funding filter**: lowered the 2024 Sharpe for every threshold (best 1.49 vs 1.83 for
  trend alone) and did not help OOS. No evidence that funding crowding helps on this data.
- **Correlation of OOS daily returns:** 0.47–0.67 between strategies. All long-only crypto
  strategies share the market's direction, so diversification across them is limited.

## What this tells us

1. **The OOS window is a bear market** (six coins −42%, worst drop −67%). Long-only trend and
   rotation strategies whipsaw in that regime. Their in-sample results came from the 2023–24
   bull market. One regime is not a verdict on trend following in general, but these settings
   didn't survive it.
2. **The breakout family is the only idea that keeps surviving out of sample** — now three
   times (v1, alt-coin, vol-sized), each time positive, far ahead of buy-and-hold, and
   sidestepping crashes. Its weak point is drawdown, and volatility sizing helps there.
3. Per the pre-registration, the vol-sized breakout goes to shadow. A forward-test
   pre-registration should come first; it overlaps with v2 and XRP/DOGE, which already run.
