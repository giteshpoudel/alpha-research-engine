# Strategy batch 2 results: machine learning (2026-11-03)

Pre-registration: [`docs/prereg/2026-11-03-batch2-ml.md`](prereg/2026-11-03-batch2-ml.md) (`04c4547`,
committed before this ran). Harness: `src/research/batch2.py`. OOS (2025-01 → 2026-11) run once.
Trial count now 349.

One fix after a crash on the first run, before any result was produced: a feature with no
values at all in a quarter's training set (funding before 2023-05) is left out of that
quarter's model, and infinite values count as missing.

## Outcome: both retired

| Strategy | OOS return | Sharpe | Max DD | Costs ×2 | Model AUC (IS / OOS) | Tier |
|---|---|---|---|---|---|---|
| Meta-labeled breakout | +0.1% | 0.05 | −13.8% | −5.8% | 0.516 / 0.497 | **retire** |
| Gradient-boosted daily model | −11.5% | 0.01 | −40.9% | −19.6% | 0.491 / 0.536 | **retire** |
| *Unfiltered vol-sized breakout (batch 1)* | *+26.9%* | *0.90* | *−20.1%* | *+12.8%* | – | *watch* |
| Buy-and-hold, 6 coins | −41.6% | −0.13 | −66.5% | – | – | – |

AUC measures how well a model ranks winners above losers: 0.5 is a coin flip, 1.0 is perfect.

## Details

- **Meta-labeling** (final: threshold 0.40, depth 3). It kept about half of the breakout
  entries. In-sample the filtered strategy looked better (Sharpe 1.80 vs about 1.87 unfiltered
  over the same 2023–24 quarters). But the model's in-sample AUC was only 0.516, so the
  filter was close to random and the in-sample Sharpe came mostly from the breakout itself.
  Out of sample the AUC was 0.497 and the filter dropped many of the winners: return fell
  from +26.9% to +0.1%. Breakout trades win about 38–39% of the time, with profits from a few
  large winners; a filter that can't tell which trades those are mostly removes profit.
  The most-used inputs were short-term volatility and ATR.
- **Daily gradient-boosted model** (final: threshold 0.50, depth 2). Predicted whether each
  coin would be up over the next 3 days. In-sample AUC 0.491, OOS 0.536 (accuracy 52.7% vs a
  47.2% base rate). That matches the published 52–54% daily accuracy, and it isn't enough:
  daily re-decisions cost 82 turns a year, and the strategy lost 11.5% with a 41% drawdown.
  Top inputs were day of week, funding and distance from the 50-day average, each with
  very small importance.

## What this tells us

1. On these features and this much data (about 740 in-sample breakout trades, about 5,800
   coin-days), gradient-boosted trees found **no usable signal**. That matches the
   literature: price-derived features alone carry very little information about crypto
   direction, and the small edge that exists at daily horizons is eaten by costs.
2. A model with AUC ≈ 0.5 can still produce a strong in-sample backtest (Sharpe 1.80). That's
   why the pre-registration, the AUC report and the one-time OOS matter.
3. Deep models (LSTM, Transformer) are not justified on this evidence: they need more signal,
   not more capacity. If ML returns, it should be with information the price doesn't
   already contain, such as order flow (batch 3) or cross-venue data.
4. The unfiltered vol-sized breakout remains the best result we have.
