# Phase 2 results: intraday strategies (2026-11-02)

Pre-registration: [`docs/prereg/2026-11-02-phase2-intraday.md`](prereg/2026-11-02-phase2-intraday.md)
(committed `cab043e` before any result). Harness: `src/research/intraday_study.py`.
Raw output: `data/research/phase2_results.json` (git-ignored).

## Verdict

**No family passed the in-sample bar.** Per the pre-registration, nothing was run
out-of-sample and nothing goes to live trading. The 2025–2026 OOS window stays untouched.

## In-sample walk-forward (2023–2024 test folds, equal-weight BTC/ETH/SOL, net of costs)

| Family @ bar | Return | Sharpe | Max DD | Time in market | Trades | Net / trade | DSR | Folds + | Pass |
|---|---|---|---|---|---|---|---|---|---|
| vol_breakout @ 1h | **+70.7%** | **1.52** | −17.6% | 13% | 268 | **+67.9 bps** | 0.29 | 6/8 | fail |
| vol_breakout @ 15m | +36.0% | 0.98 | −14.9% | 10% | 897 | +11.4 bps | 0.00 | 5/8 | fail |
| overshoot_fade @ 1h | −1.0% | 0.04 | −17.4% | 8% | 284 | +1.7 bps | 0.00 | 3/8 | fail |
| overshoot_fade @ 15m | −42.9% | −1.76 | −48.4% | 8% | 1309 | −12.5 bps | 0.00 | 1/8 | fail |

Buy-and-hold over the same period: Sharpe **2.02–2.08** (2023–2024 was a strong bull market).
Average round-trip cost ≈ 10.7 bps; the bar was ≥ 2× that per trade.

### Closest candidate: vol_breakout @ 1h

- **Passed:** Sharpe ≥ 1.0, 6/8 positive folds, 268 trades, net per trade 6× cost.
- **Failed:**
  - **Deflated Sharpe 0.29 (needed 0.95).** After testing 240 configurations, a result this
    good has a large chance of being the luckiest draw rather than a real edge.
  - **Lower Sharpe than buy-and-hold (1.52 vs 2.02).** It earned about twice the return of
    buy-and-hold scaled to the same time in market (+70.7% vs +36.3%), but per unit of risk
    it did not beat simply holding through a bull market.
- **Crash stress:** lost 13.6% on one symbol around LUNA and 9.7% around Celsius/3AC
  (flagged; > 5%).

### Overshoot fade: rejected

Buying sharp drops lost money at 15m and was flat at 1h. It was hurt by exactly the events
it targets: the yen-carry unwind (−11.5% worst symbol) and the 2025-10-10 crash (−6.5%).
In real crashes the first flush is often not the bottom. This matches the repo's earlier
finding that intraday mean reversion fails on costs and tail events.

## What this means

- None of these four is good enough to risk money on. The best one looks like trend beta
  in a bull market, and it doesn't survive the multiple-testing correction.
- Faster bars are worse: 15m versions earn less per trade and pay more costs.
- Volatility alone is not an edge, confirmed again.

## Honest next options

1. **Shadow-track vol_breakout @ 1h, without money.** Run it in chorus-howl shadow mode as a
   prospective test. Forward data costs nothing and can't be overfit, but it is not adoption.
2. **Order-flow features** once the websocket tape has 4–8 weeks of data. This is the
   documented short-horizon edge the bar data can't capture.
3. **A new pre-registration** for any refinement (for example a regime filter on the
   breakout). The trial count carries over (240 + new trials), so the bar gets harder.
4. **The live canary still makes sense.** It measures real execution, not edge.
