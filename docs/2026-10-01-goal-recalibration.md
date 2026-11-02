# Goal recalibration: from "10%/day" to evidence-based targets (2026-10-01)

Status: decision note. Purpose: replace the original "sustained ~10%/day intraday"
premise with targets the measured evidence can actually support, and state what
"go live" should require. No code change is implied by this note except where a
section says so.

## 1. Why the original premise must change

10%/day compounded is ~10^11 %/year (a $1,000 sleeve would imply ~$10^18 in a
year). Even 1%/day is ~3,700%/year. No public evidence, and nothing we have
measured, approaches that. Keeping an unreachable target in `agent_goals` makes
the optimizer chase noise and mislabels honest flat results as failures.

## 2. What we actually measured (this repo, 2026-09/10)

| Approach | Result | Verdict |
|---|---|---|
| Funding carry (delta-neutral, HL perp vs Binance.US spot) | **+3.4% to +14.4% APY** net, one entry flip, ~1–4% DD | **Adopt / papering** |
| 5m mean-reversion scalping | breakeven ~3–5 bps/side vs realistic cost; OOS fragile | Reject |
| Maker/limit execution | worse (−10..−15 bps edge) via adverse selection | Reject |
| Cross-sectional momentum (hourly/daily) | IS Sharpe 2.1–2.9, **OOS −47%/−51%** | Reject |
| Lead-lag (BTC→alt) | same-bar artifact; negative at +1 bar | Reject |
| Hourly ML (literature: Cohen 2023) | ~0.08%/day idealised over 6 mo | Weak / regime-dependent |

The one production-grade edge is **funding carry**, and it is market-neutral and
low-vol, not a daily compounding machine.

## 3. Compounding math (why daily targets are the wrong unit)

- 10%/day → 1.10^365 ≈ **1.3×10^11** (13 trillion %).
- 1%/day → 1.01^365 ≈ **37.8×** (3,700%/yr).
- 0.05%/day → ≈ **20%/yr**; 0.03%/day → ≈ **11.6%/yr**.

Carry sits around **0.02–0.04%/day** on perp notional, and *less* on total
capital because both legs are funded (spot + perp margin). That is the honest
scale.

## 4. Proposed target framework

Track **net annualized return** (already computable per strategy) with explicit
risk, not a daily number:

| Band | Composition | Net target | Risk |
|---|---|---|---|
| Conservative | carry only, regime-guarded, no leverage | **5–12%/yr** | low; negative-funding regimes |
| Moderate | carry + selective hourly/daily directional | **8–20%/yr** | +10–30% DD |
| Aggressive | carry at 2–3× crypto-collateralized leverage | **15–30%/yr** | liquidation + funding-flip tail |

Capital-efficiency reality: carry yields are quoted on **perp notional**; annualized
on *deployed* capital (spot + margin) is roughly **half to two-thirds** of that
unless margin is reused.

## 5. Recommended changes to the goal system

1. Stop using `daily_profit_pct` to gate success. Keep it as a *monitor*, but set
   the goal to a **rolling annualized net return band** with a stated risk cap
   (e.g., target 10%/yr, hard stop if trailing-90d DD > 20%).
2. Cap any per-day goal at a sane ceiling (e.g. 0.10%/day) so `raise_goal` cannot
   drift toward the old 10%/day.
3. Score strategies by **net, risk-adjusted, multiplicity-corrected** performance
   (fees + slippage + lag inside the objective; Deflated Sharpe as the adoption
   gate) — already in place.
4. Report **return on deployed capital**, not just on notional, so leverage and
   capital efficiency are explicit.

## 6. What "go live" should require (unchanged guardrails + additions)

- A strategy must clear **prospective paper** for a meaningful window (≥1 quarter)
  after any historical validation — no adoption straight from a backtest.
- Carry: prove net APY live with the regime guard, and add a **basis/negative-
  funding risk limit** (now measurable via `perp_marks`/`basis.py`).
- Any leverage is sized so a **2σ adverse funding + basis move** cannot liquidate.
- Live-trading code stays out of the public repo (policy).

## 7. Bottom line

The credible engine is a **low-vol, market-neutral funding carry** in the
mid-single to low-double digits annually, optionally augmented by a hard-won
directional edge. The honest portfolio expectation today is **~5–15%/yr** —
good, real, and completely incompatible with "10%/day". This note is the
baseline every future claim should be measured against.

## 8. Implemented (2026-11-02)

- `src/agents/goals.py`: default daily target = daily equivalent of 10%/yr (0.0261%/day);
  `raise_goal` capped at 0.10%/day (~44%/yr); stored targets above the cap are clamped.
- Fixed `daily_profit_pct`: it averaged raw equity across whichever series had a row at
  each timestamp (14 on the hour, 3 at :15/:30/:45), which produced fake +27% "days" and
  marked the goal met daily since 2026-10-07. It now compares each active
  (strategy, symbol) series with its own equity 24h earlier and averages those returns.
