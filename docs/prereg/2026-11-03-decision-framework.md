# Pre-registration: decision framework for strategy tests (2026-11-03)

Committed before any test that uses it. It replaces the single pass/fail rule
("retire if OOS return ≤ 0, Sharpe < 0.5, or max drawdown ≥ 20%") for **tests
pre-registered after this commit**. It is not applied to earlier results: breakout v1
(`969ceab`) and the alt-coin breakout (`51f703e`) stay retired under the rule they were
registered with. Both already run in shadow, which is what the new "watch" tier would do.

## Why

The old rule mixed two questions: *is there an edge?* and *how much risk at what size?*
Its 20% drawdown limit was measured on a fully invested sleeve, so it tested an assumed
size rather than the strategy, ignored what the market did over the same period, and let
one bad stretch override the rest of the evidence. Twice it retired strategies that made
+25–28% while their coins fell 30–51%.

## Three outcomes

Each test's out-of-sample result is placed in exactly one tier.

### 1. Retire: no edge (any one fails)
- OOS net return ≤ 0;
- OOS Sharpe < 0.5;
- OOS net return ≤ 0 with costs doubled.

### 2. Candidate: edge and acceptable risk (all hold)
- Not retired.
- **Budget drawdown ≤ 15%**: max drawdown of the strategy simulated at its planned live
  sizing (fixed dollars per position, position cap) measured against the whole pilot
  budget. Example: 4 positions × $25 on a $100 budget. The fully invested sleeve
  drawdown is still reported.
- **Smaller than holding**: max drawdown ≤ half the max drawdown of equal-weight
  buy-and-hold of the same coins over the same period.
- **Recovers**: longest time below a previous equity peak ≤ 6 months.

### 3. Watch: edge, but a risk check fails
- Not retired, and not a candidate.
- Goes to shadow (paper) only, with the drawdown brake where the strategy family has one.
- It may become a candidate only through a forward test pre-registered for it.

## Next steps by tier

- **Candidate:** shadow, then small live learning trades under that strategy's
  pre-registered forward rules (template: breakout v2, ≥ 3 weeks of shadow).
- **Watch:** shadow only. A risk fix such as smaller size, a brake, or fewer coins is a
  new pre-registration and is judged on forward data.
- **Retire:** stop. It can come back only as a new hypothesis with new reasoning.

## Unchanged

- Live money is protected by the bot's own limits, not by this framework: $5 daily and
  $25 total loss stops, position caps and the kill switch. Changing those stays a
  separate decision the user approves first.
- Forward tests already registered (breakout v2, XRP/DOGE) keep their own rules,
  including retirement at a 20% ideal sleeve drawdown.
- Research discipline is unchanged: pre-registration before results, walk-forward
  selection, OOS run once, costs ×3 on stress bars, running trial count (314).
