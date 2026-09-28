# Rapid Optimization System — Design

Date: 2026-09-28
Scope: Turn the once-daily optimizer into a **continuous, compute-hungry optimizer** that discovers and tunes strategies fast enough to reach a sustained target, then gate a later live-trading phase.

## Decisions (user)

- **Success target:** a strategy (per symbol / category) that **consistently grows ~10% per day** in paper. "Consistently" = meets/exceeds the daily goal over a rolling window of ≥7 consecutive days (paper forward), not a single lucky day.
- **Compute:** no local budget cap; instead guard against **stuck loops/deadlocks** wasting compute (timeouts, heartbeats, iteration/run ceilings, restart-on-stall).
- **Autonomy:** the agent may **add and tune new strategies autonomously for paper trading** — advanced strategies encouraged, no permission needed. (Live trading still gated.)
- **Live trading (later):** graduate to small real budget once the target is sustained; venues **Coinbase + Binance.US**.

## Architecture

1. **Continuous worker** — `optimizer --continuous` runs iterations in a loop (interval between them) under launchd `KeepAlive`. Each iteration optimizes one symbol and records its own `agent_runs` row.
2. **Watchdog / anti-waste** — per-iteration wall-clock deadline, overall `--max-runtime-seconds` (process exits and restarts fresh, bounding leaks), `--max-iterations`, and a heartbeat to `change_log`. Stalls are logged, not silently looped.
3. **Priority scheduler** — each iteration targets the neediest symbol: halted sleeve, missing params, or worst validation Sharpe; least-recently-touched breaks ties.
4. **Adoption margin** — champion/challenger must beat the incumbent by ε (not tie) to avoid churn from continuous re-proposals.
5. **Fast feedback** — on adoption, re-run the symbol's OOS backtest and re-replay its paper sleeve, and log the before/after delta.
6. **Strategy synthesis (stage 2)** — agent-authored strategies in `src/backtesting/strategies/generated/`, constrained by a sandbox (see below), registered, tuned, evaluated; paper-only.

## Guardrails

- Champion/challenger with margin; IS/OOS discipline; never tune on OOS.
- No cost cap, but step/iteration/runtime ceilings + watchdog to prevent runaway loops/deadlocks.
- Strategy sandbox (stage 2): pure `signals(close, **params)` contract; **AST scan** allows only `numpy`/`pandas`/`math` imports and forbids file/network/`exec`/`eval`/`importlib`; execution under a timeout; errors isolated; generated strategies are paper-only until they clear the gate.
- Kill switch: stop the launchd worker.
- Every adoption/switch/goal change → `change_log` + dashboard.

## Metrics / goal tracking

- `daily_profit_pct` (equal-weight paper, trailing 24h) vs the ratcheting daily target.
- **Streak tracking**: consecutive days meeting target (new `agent_goals` fields or derived) — the gate for live trading.
- Per strategy×category performance aggregation to focus the search.

## Stages

1. **Stage 1 (this):** continuous worker + watchdog + priority + adoption margin + paper re-replay on adoption.
2. **Stage 2:** autonomous strategy synthesis + sandbox + broader throughput (concurrent backtests, more trials).
3. **Stage 3:** live-trading groundwork (Coinbase + Binance.US broker interface, dry-run, sizing/risk limits, kill switch), enabled only once the 10%/day target is sustained.
