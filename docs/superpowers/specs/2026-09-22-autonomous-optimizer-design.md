# Autonomous Strategy Optimizer — Design

Date: 2026-09-22
Scope: An autonomous agent system (`src/agents/`) that uses the existing research/backtest/paper stack as tools to optimize trading strategy — choosing parameters, strategies, and *which symbols* to trade — under strict guardrails, working toward a daily profitability goal.

Authorized by user decisions:
- **Planner/executor are product-side API calls**: planner = Moonshot **kimi-k3**, executor/builder = **DeepSeek API** (`DEEPSEEK_API_KEY`).
- **Symbol classification**: rule-based + overrides; **price buckets** `$0–2`, `$2–20`, `$20+`; separate **meme / high pump-and-dump risk** category.
- **Autonomy**: full autonomy for backtesting and paper trading; agent sets and raises a daily profit goal over iterations.
- **Guardrails**: IS/OOS-safe + champion/challenger; agent may make **minor** code fixes autonomously, **major** changes require user approval; resource/data additions (e.g. Twitter) are **requested from the user with justification**.
- **Universe**: agent can add/remove symbols (any coin tradable on a US exchange), targeting meme/penny coins for fast intraday trading.

## Vision

An optimizer that iterates: observe the feedback loop (backtest + paper + signal + health) → hypothesize improvements (params, symbol subset, strategy, categories) → run experiments via tools → adopt only what wins the champion/challenger gate → measure against a daily goal → repeat. Over time it should discover which symbol clusters and strategy styles (intraday scalp on meme/penny coins vs swings on majors) are profitable, and focus trading there.

## Architecture: `src/agents/`

- `symbols.py` — price bucketing + category/risk classification (rule-based + overrides). **Phase 1.**
- `tools.py` — deterministic tool surface over existing modules (no LLM inside): `run_backtest`, `fit`/`publish`, `validate_params`, `compare`, `signal_eval`, `paper_summary`, `system_health`, `classify_universe`, `backfill_symbol`.
- `multi_llm.py` — model clients: planner (Moonshot kimi-k3, OpenAI-compatible) and executor (DeepSeek API, OpenAI-compatible); reuse `research/llm.py` patterns.
- `planner.py` — the planner agent: given state + goal, produces a structured improvement proposal (hypothesis, changes, experiment plan, expected effect).
- `executor.py` — the executor agent: turns a proposal into concrete tool calls (and, where safe, minor code/config changes), collects evidence.
- `optimizer.py` — the autonomous loop: state assembly → plan → execute → evidence → adopt/reject → record → update goal. Controls loop count per run.
- `requests.py` — resource/change requests to the user, with justification and expected impact.

## Storage (ClickHouse)

| Table | Purpose |
|---|---|
| `symbol_metadata` | per symbol: price, `price_bucket`, `category` (major/alt/meme), `ann_vol`, `pump_risk`, `updated_at` |
| `strategy_proposals` | hypothesis, proposed changes (JSON), evidence (IS/OOS metrics), decision, model, created_at |
| `agent_runs` | one row per optimizer run: goal, loops, proposals made, adopted count, summary |
| `agent_goals` | daily target (profit %) and progress |
| `agent_requests` | requests to the user (data/API/major change) with justification + status |

## Guardrails (non-negotiable)

1. **No tuning on OOS**; every adoption goes through champion/challenger on the same folds.
2. All tools deterministic and reproducible; every experiment logged with provenance.
3. Autonomy boundary: backtest/paper experimentation is free; **minor code fixes** autonomous; **major** architecture/guardrail changes → `agent_requests` for user approval.
4. Resource requests (new data/API) require user action; the agent must justify expected profit impact.

## Phases

- **Phase 1 (this):** symbol classification + `symbol_metadata` + tests.
- **Phase 2:** deterministic tool surface + multi-model clients (kimi-k3 planner, DeepSeek executor).
- **Phase 3:** proposal workflow + tables + champion/challenger integration + dashboard "Proposals" view.
- **Phase 4:** autonomous loop + daily goal + scheduler + loop-count control.
- **Phase 5:** symbol universe add/remove + backfill + meme/penny intraday strategy candidates.
- **Phase 6:** resource-request mechanism + minor/major code-change policy.

## Open items (later)

- Exact DeepSeek API model id for the executor.
- Daily-goal definition (equal-weight paper portfolio % vs count of profitable transactions) and starting target.
- Loop cadence and compute budget (Colima 8 GiB).
- How aggressively to expand the symbol universe (data/backfill cost).
