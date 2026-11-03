# Amendment: exploratory out-of-sample check of vol_breakout @ 1h (2026-11-03)

Written and committed **before** the OOS window is evaluated. Amends
[`2026-11-02-phase2-intraday.md`](2026-11-02-phase2-intraday.md).

## What changes and why

The original pre-registration evaluates OOS only for families that pass the in-sample
bar. None passed. The closest, `vol_breakout @ 1h`, failed on Deflated Sharpe (0.29) and
on Sharpe vs buy-and-hold (1.52 vs 2.02).

Goals changed on 2026-11-03: chorus-howl may run small real-money learning trades
inside the $100 pilot limits. The research bar now decides *scaling*, not whether a
small pilot may run. A forward test of ~8 weeks gives ~15 trades; the untouched OOS
window (2025-01-01 → 2026-11-01, 22 months, including the 2025-10-10 crash and the
2026-10-29 Binance hack) gives far more evidence now. This amendment spends that window
on this one family.

Consequences:
- The OOS window is **used up** for this family. It is labelled exploratory, because the
  family failed the IS bar first.
- The trial count becomes 241 for any future Deflated Sharpe calculation.
- No other family or parameter set is evaluated on the window. Parameters are the ones
  already frozen by the pre-registered rule (best 2024 net Sharpe per symbol):
  BTC N=96 v=1.0 k=3 T=192; ETH N=48 v=1.3 k=3 T=96; SOL N=24 v=1.3 k=3 T=48.

## What is computed (harness and cost model unchanged)

Equal-weight BTC/ETH/SOL portfolio on 1h bars, one-bar lag, stressed costs as in the
original. Reported:
- net return, annualized Sharpe, max drawdown, trades, mean net return per trade,
  time in market; per symbol and per quarter;
- the same at costs ×2 and with Kraken fees (40 bps taker);
- a **measured-cost** sensitivity from the 2026-11-03 live canary: 2 bps fee, 0 slippage,
  half-spread 0.5 bps per side (reported only, not used for the decision);
- buy-and-hold Sharpe and exposure-matched buy-and-hold return;
- crash windows: 2025-10-10 and 2026-10-29.

## Decision rule (fixed now)

At base costs:
- **Retire the family** if OOS net return ≤ 0, or Sharpe < 0.5, or max drawdown ≥ 20%.
- Otherwise it is a **live-learning candidate**: it may run live at pilot size (inside
  the $100 limits, $25 total stop) while the shadow forward test continues. It is not
  adopted: scaling beyond the pilot still needs it to beat exposure-matched buy-and-hold
  OOS *and* the forward-test promotion rules in chorus-howl `docs/shadow-breakout.md`.
