# Pre-registration: tighter exits for the breakout (2026-11-04)

Written and committed **before** any of these exits has been run. Trials added: 3
(running total 402).

## Question

Would the breakout do better by locking in gains sooner: selling as soon as an open gain
starts to shrink, instead of waiting for a wide trailing stop? Smaller, more frequent wins
are attractive, but costs (10–16 bps a round trip) and hourly noise (0.5–1%) may eat them,
and a tight exit may cut short the big moves that pay for the losers.

## What stays fixed

- **Portfolio:** the vol-sized breakout on the six original coins (BTC, ETH, SOL, XRP, DOGE,
  LINK), the broadest one in shadow. Frozen entry settings `BREAKOUT_VOL_PARAMS`
  (chorus-howl `signals.py`): N, vratio and T per coin, unchanged.
- **Sizing:** (1/6) × min(1, 0.80 / annualized 720-hour realized volatility), fixed at entry.
- **Fills:** decided on an hourly close, filled at the next open.
- **Costs per side:** BTC 5, ETH 5, SOL 6, XRP 5, DOGE 7, LINK 8 bps, ×3 on top-decile
  volatility hours (same stress flags as all batches).
- **Time limit:** every variant still exits after T hours.

## Exits compared

ATR is the N-hour average true range used by the existing rule (same `atr(h, l, c, N)`).
The peak is the highest close since entry.

| Name | Exit rule |
|---|---|
| baseline | current: close < peak − k × ATR, with each coin's frozen k (2.0 or 3.0) |
| tight15 | close < peak − 1.5 × ATR |
| tight10 | close < peak − 1.0 × ATR |
| lock | baseline stop, plus a profit lock: once peak ≥ entry close + 1 × ATR, exit when close ≤ entry close + 0.5 × (peak − entry close), i.e. half the best open gain has been given back |

The baseline is a check, not a trial: it must reproduce the existing rule exactly.

## Evaluation

- **Primary window:** 2025-01-01 → 2026-11-01 (out of sample for every frozen setting).
- **Check window:** 2022-01-01 → 2025-01-01.
- Reported for each: return, Sharpe, max drawdown, return with costs ×2, number of
  trades, win rate, average win, average loss and average hours held.

## Decision

A variant **replaces the baseline exit** only if all of these hold:

1. Primary-window Sharpe ≥ baseline + 0.10, and return > baseline's.
2. Primary-window return with costs ×2 > baseline's with costs ×2.
3. Check-window return ≥ baseline's.

If more than one passes, the one with the highest primary-window Sharpe. If none passes,
the current exit stays.

Adopting a new exit would mean a **new** shadow run with that exit beside the existing ones
(existing forward tests stay frozen), not a change to them. Live use still needs the
forward checks.

## Things decided in advance

- No change to coins, settings, the 1.0 / 1.5 / 0.5 thresholds, costs or windows after
  seeing any result. Bugs found after running are fixed and both results are reported.
- Code: `src/research/breakout_exits.py`. Results: `docs/2026-11-04-breakout-exits-results.md`
  and a row in `docs/research-index.json`.
