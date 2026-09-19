# Feedback Loop Evaluation — Does Rolling Re-tune + Gating Add Value?

Date: 2026-09-14
Scope: Quantify the two components of the backtest↔paper feedback loop — (a) rolling walk-forward re-tune (v2 params, deployed 2026-09-18) and (b) per-symbol trailing-return gating — against the original fixed-IS params (v1, deployed 2025-01-01).

**Bottom line: the loop has not demonstrated out-of-sample value. The apparent improvement is a contamination artifact; gating trades return for drawdown; the forward window is far too short to judge.**

## 1. Tuning window contamination (critical)

The rolling re-tune at `T` tunes on `[T − 36mo − 7d embargo, T − 7d]`. With `T = 2026-09-18`, the in-sample window is **2023-09-11 → 2026-09-11**, which **overlaps the OOS window 2025-01-01 → 2026-09**. So v2 is *in-sample* on most of the period commonly used as OOS. Any v2 evaluation on 2025–2026 is not a valid out-of-sample test.

v2's genuine OOS begins **2026-09-11** (after the embargo), i.e. roughly one week of data at evaluation time.

## 2. v1 vs v2 on OOS (`2025-01-01 → 2026-09-11`)

| params | avg OOS return (12 symbols) | validity |
|---|---:|---|
| v1 (fixed IS 2022–2024) | **−0.065** | clean OOS |
| v2 (rolling IS through 2026-09-11) | **+0.713** | **contaminated** (in-sample) |

Stored walk-forward validation Sharpe (the tuner's own out-of-fold metric):

| params | mean validation Sharpe |
|---|---:|
| v1 | **+0.31** |
| v2 | **+0.08** |

v2 is **worse** on the tuner's own validation measure for most symbols (ETH 0.20→−0.93, SOL 0.94→−0.07, DOGE 0.46→−0.50, LINK 0.41→−0.27). The 3-year rolling window drops the 2022 data and fits recent regimes less well.

## 3. Gating A/B (clean OOS with v1 params, equal-weight $1 sleeves)

| gating | equal-weight return | max drawdown |
|---|---:|---:|
| off | **−0.065** | −0.251 |
| on | **−0.128** | −0.170 |

Gating **reduces return** (−0.065 → −0.128) while reducing drawdown. It is a risk control, not a return enhancer, and on this sample it does not improve absolute performance (both are negative).

## 4. Forward window since v2 deploy (`2026-09-11 → 2026-09-18`)

| params | avg return (12 symbols) |
|---|---:|
| v1 | +0.0064 |
| v2 | +0.0024 |

One week — pure noise. No conclusion possible.

## Conclusions

1. **Rolling re-tune value is unproven and currently negative on validation.** Do not present v2 as an improvement; its OOS "win" is contamination.
2. **Gating is risk-only** and reduces return on this sample.
3. **The underlying Mean Reversion edge is weak/negative over the full OOS** (v1 avg −0.065), independent of the loop.

## Recommendations

1. **Add a champion/challenger adoption gate.** At each re-tune, publish new params only if their walk-forward validation Sharpe beats the incumbent's **and** an untouched holdout check; otherwise keep the current version. This prevents degradation like v2.
2. **Never evaluate params on a window they were tuned on.** For forward evaluation, only use data strictly after the tuning embargo, and let it accumulate (target ≥ 3 months) before judging.
3. **Reconsider the rolling IS length** (e.g., keep a longer window or anchor to a fixed IS) given validation degradation.
4. **Treat gating as optional risk control**, to be enabled only if drawdown reduction is desired and its return cost is acceptable.
5. Keep the loop running for **forward** measurement, clearly labeled unproven.

## Reproduce

Numbers come from: `tuned_params` (v1/v2), `backtest_runs` OOS via the engine, and an in-memory replica of the paper executor's trailing-return gate over OOS. See `.superpowers/sdd/progress.md` for the session ledger.
