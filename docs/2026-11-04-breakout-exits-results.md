# Results: tighter exits for the breakout (2026-11-04)

Pre-registration: `docs/prereg/2026-11-04-breakout-exits.md` (commit 3ca52c9). Code:
`src/research/breakout_exits.py` (committed before running as d6b13a4). Raw output:
`data/research/breakout_exits.json`. Trials: 3 (running total 402). Ran once, no bugs.

## Outcome: keep the current exit

None of the three tighter exits passed. Same six-coin vol-sized breakout, same entries;
only the exit changes.

| Exit | 2025–26 return | Sharpe | Max DD | Costs ×2 | 2022–24 return |
|---|---|---|---|---|---|
| **baseline** (2–3 ATR trailing) | **+26.9%** | 0.90 | −20.1% | **+12.8%** | **+80.8%** |
| tight15 (1.5 ATR) | +27.1% | **1.13** | **−8.0%** | +10.2% | +12.1% |
| tight10 (1.0 ATR) | +12.4% | 0.66 | −9.6% | −4.1% | +1.2% |
| lock (sell after giving back half the gain) | +24.9% | 0.96 | −12.9% | +9.2% | +29.1% |

| Exit | Rule 1: Sharpe +0.10 and higher return | Rule 2: costs ×2 | Rule 3: 2022–24 | Pass |
|---|---|---|---|---|
| tight15 | yes | no | no | no |
| tight10 | no | no | no | no |
| lock | no | no | no | no |

## Trades (2025–26)

| Exit | Trades | Win rate | Avg win | Avg loss | Avg trade | Hours held |
|---|---|---|---|---|---|---|
| baseline | 541 | 38% | +4.13% | −2.03% | +0.31% | 22 |
| tight15 | 628 | 38% | +3.08% | −1.44% | +0.27% | 10 |
| tight10 | 709 | 35% | +2.50% | −1.15% | +0.13% | 6 |
| lock | 603 | **55%** | +2.20% | −2.14% | +0.25% | 14 |

Per-trade returns are net of both sides' costs, on the coin's price (not portfolio-weighted).

## How to read this

- **The profit lock does what it promises: more wins.** 55% of trades win instead of 38%.
  But the wins are half as big (+2.2% vs +4.1%) and the losses are the same size, so the
  average trade earns less. In the 2022–24 rally years it made +29% where the current exit
  made +81%: it sold strong trends early.
- **Tighter trailing stops cut losses and winners alike, and pay more costs.** At 1.0 ATR
  the average trade is +0.13%, too thin to survive doubled costs.
- **1.5 ATR is the interesting near miss.** In 2025–26 it matched the return with less
  than half the drawdown (−8% vs −20%). But it failed the two robustness checks: with
  costs doubled it earns less, and in 2022–24 it made +12% against +81%. Its 2025–26 edge
  looks like a feature of a choppy, falling market, not a better exit in general. This was
  seen after the fact, so any later test of it counts as a new trial with that caveat.
- **The general lesson:** the breakout makes its money from a minority of large winners
  (38% win rate, wins twice the size of losses). Exits that take profits sooner turn some
  of those large winners into small ones, which costs more than the extra small wins earn.

## Next

Nothing changes in shadow or live. The current exit stays.
