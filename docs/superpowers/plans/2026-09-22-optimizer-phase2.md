# Optimizer Phase 2 — Tools, State, Tracing, Cost

Goal: deterministic foundation for the autonomous optimizer — Pydantic-typed tool registry over the existing stack, multi-model clients (kimi-k3 planner / DeepSeek executor) with fallback, and run/step tracing with token+cost accounting.

Spec: `docs/superpowers/specs/2026-09-22-autonomous-optimizer-design.md`

## Constraints
- Deterministic tools only (no LLM inside handlers); every call traced.
- No live LLM in tests (mock HTTP).
- DB-backed trace/cost; simple fallback counters + budget caps. No OTel/task queue/async rewrite.
- No tuning on OOS (champion/challenger stays Phase 3).

## Tasks
1. **Storage** — `agent_runs(run_id, kind, goal_json, status, started_at, ended_at, loops, tokens_in, tokens_out, cost_usd, model_planner, model_executor, summary)`, `agent_steps(run_id, step_idx, tool, args_json, result_json, latency_ms, error, tokens_in, tokens_out, cost_usd, created_at)`.
2. **state.py** — Pydantic models: `OptimizerState`, `ToolResult`, per-tool arg models; add `pydantic>=2` to requirements.
3. **tools.py** — `Tool`/`TOOL_REGISTRY`/`invoke()` (Pydantic-validate args, time, capture errors); handlers wrap runner/tuner/compare/signal_eval/collectors/health/symbols.
4. **tracing.py** — `Tracer.run()` context manager (insert `agent_runs`, finalize on exit), `Run.step()` (insert `agent_steps`), `MODEL_COSTS` (env/placeholder; $ = 0 until priced), `Budget`/`BudgetExceeded`.
5. **multi_llm.py** — `chat(role, system, user) -> LLMResult{text, model, tokens, cost, degraded}`; planner Moonshot kimi-k3 → DeepSeek → Ollama; executor DeepSeek → kimi-k3 → Ollama; per-provider failure cooldown.
6. **tests + docs** — registry/validation/errors; tracing rows + cost + budget; multi_llm env selection + fallback + cooldown with mocked HTTP; README/ledger.

## Verification
- `pytest tests/ -v` green.
- Dry `Tracer.run` + tool steps produce rows; mocked LLM chat shows tokens/cost and fallback.
