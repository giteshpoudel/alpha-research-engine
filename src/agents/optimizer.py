"""Autonomous strategy optimizer loop.

Observe -> plan (planner LLM) -> build (executor LLM) -> propose ->
champion/challenger gate -> record, working toward a daily profit goal. Bounded
by a step/cost ``Budget`` and per-provider circuit breakers, and fully traced.

Usage:
    python -m src.agents.optimizer --loops 2
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

from src.agents import goals, proposals, tools
from src.agents.multi_llm import chat as default_chat
from src.agents.state import OptimizerState
from src.agents.tracing import Budget, Tracer
from src.ingestion.schemas import get_clickhouse_client

PARAM_BOUNDS = {"window": (12, 72), "z_entry": (-3.5, -1.0), "z_exit": (-0.5, 0.5)}

_PLANNER_SYSTEM = (
    "You are the planning half of an autonomous crypto strategy optimizer for a "
    "research/paper-trading system. You optimize the mean-reversion strategy's "
    "parameters per symbol. The harness enforces out-of-sample discipline for you.\n"
    "Respond with STRICT JSON only, no prose. Choose ONE action:\n"
    '1) parameter change: {"action":"propose_params","symbol":"<TICKER>",'
    '"params":{...},"hypothesis":"<one sentence>","analysis":"<one or two sentences>"}\n'
    '   (mean_reversion params: window 12-72, z_entry -3.5..-1.0, z_exit -0.5..0.5)\n'
    '2) switch a symbol to a different strategy: {"action":"switch_strategy",'
    '"symbol":"<TICKER>","strategy":"mean_reversion"|"momentum"|"breakout",'
    '"hypothesis":"<one sentence>","analysis":"<one or two sentences>"}\n'
    '3) ask the human for data/API or a major change: {"action":"request",'
    '"kind":"data"|"api"|"change","title":"<short>",'
    '"justification":"<why it should improve profit>","expected_impact":"<estimate>"}\n'
    '4) do nothing: {"action":"none"}\n'
    "Match strategy style to the symbol: momentum/breakout for trending majors, "
    "mean_reversion for choppy/penny/meme names. Prefer propose_params for small "
    "tweaks; use switch_strategy when a different approach fits better. Use request "
    "ONLY when new data/APIs or a major change is the real blocker."
)

_EXECUTOR_SYSTEM = (
    "You are the executor half of an autonomous crypto strategy optimizer. Given "
    "a planner's JSON proposal, return the final concrete proposal as STRICT JSON "
    'only: {"action","symbol","params":{"window","z_entry","z_exit"},'
    '"hypothesis","analysis"}. Tighten or veto inconsistent numbers; otherwise echo '
    "it. Allowed ranges: window 12-72, z_entry -3.5..-1.0, z_exit -0.5..0.5."
)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_params(params) -> bool:
    if not isinstance(params, dict):
        return False
    for key, (low, high) in PARAM_BOUNDS.items():
        try:
            value = float(params.get(key))
        except (TypeError, ValueError):
            return False
        if not low <= value <= high:
            return False
    return True


def _observe(ch) -> dict:
    from src.backtesting.strategies.registry import STRATEGY_NAMES
    from src.meta_learning.assignments import list_assignments

    return {
        "paper": tools.invoke("paper_summary", {}, ch=ch).data,
        "health": tools.invoke("system_health", {}, ch=ch).data,
        "symbols": tools.invoke("classify_universe", {}, ch=ch).data,
        "strategies": list(STRATEGY_NAMES),
        "assignments": list_assignments(ch),
    }


def _planner_prompt(observed: dict, state: OptimizerState) -> tuple[str, str]:
    user = json.dumps({"goal_profit_pct": state.goal_profit_pct, "observed": observed},
                      default=str)
    return _PLANNER_SYSTEM, user


def _llm(chat_fn, role: str, system: str, user: str, run, budget: Budget):
    result = chat_fn(role, system, user)
    run.step(f"llm_{role}", {"system": system, "user": user},
             {"model": result.model, "text": result.text[:1000]},
             tokens_in=result.tokens_in, tokens_out=result.tokens_out, model=result.model)
    budget.steps += 1
    budget.cost_usd += result.cost_usd
    return result


def _execute_plan(plan: dict | None, ch, run, strategy: str, publish_adopted: bool) -> dict:
    if not plan:
        run.step("no_action", {"plan": plan}, {"action": "none"})
        return {"action": "none"}
    if plan.get("action") == "request":
        from src.agents import requests as requests_mod
        request_id = requests_mod.create_request(
            ch, plan.get("title", ""), plan.get("justification", ""),
            kind=plan.get("kind", "data"), expected_impact=plan.get("expected_impact", ""),
            model=plan.get("model", ""), run_id=run.run_id)
        run.step("request", {"request_id": request_id, "title": plan.get("title", "")},
                 {"action": "request"})
        return {"action": "request", "request_id": request_id}
    if plan.get("action") == "switch_strategy":
        from src.backtesting.strategies.registry import STRATEGY_NAMES
        symbol, strategy = plan.get("symbol"), plan.get("strategy")
        if symbol and strategy in STRATEGY_NAMES:
            change = {"strategy": strategy, "symbol": symbol,
                      "params": plan.get("params") or {}}
            proposal_id = proposals.create_proposal(
                ch, plan.get("hypothesis") or plan.get("analysis", ""), change,
                kind="strategy", symbol=symbol, model=plan.get("model", ""),
                run_id=run.run_id)
            row = proposals.evaluate_proposal(ch, proposal_id,
                                              publish_adopted=publish_adopted,
                                              run_id=run.run_id)
            run.step("proposal", {"proposal_id": proposal_id, "change": change},
                     {"decision": row["decision"], "evidence": row["evidence"]})
            return {"action": "proposal", "kind": "strategy",
                    "proposal_id": proposal_id, "decision": row["decision"]}
        run.step("no_action", {"plan": plan}, {"action": "none"})
        return {"action": "none"}
    if plan.get("action") != "propose_params" or not validate_params(plan.get("params")):
        run.step("no_action", {"plan": plan}, {"action": "none"})
        return {"action": "none"}
    change = {"strategy": strategy, "symbol": plan["symbol"], "params": plan["params"]}
    proposal_id = proposals.create_proposal(
        ch, plan.get("hypothesis", ""), change, symbol=plan["symbol"],
        model=plan.get("model", ""), run_id=run.run_id)
    row = proposals.evaluate_proposal(ch, proposal_id, publish_adopted=publish_adopted,
                                      run_id=run.run_id)
    run.step("proposal", {"proposal_id": proposal_id, "change": change},
             {"decision": row["decision"], "evidence": row["evidence"]})
    return {"action": "proposal", "proposal_id": proposal_id, "decision": row["decision"]}


def run_optimizer(ch=None, loops: int = 1, strategy: str = "mean_reversion",
                  publish_adopted: bool = True, chat_fn=None, max_cost_usd: float = 0.0,
                  max_steps: int = 0, planner_model: str = "",
                  executor_model: str = "", goal_pct: float | None = None,
                  day=None) -> dict:
    ch = ch or get_clickhouse_client()
    chat_fn = chat_fn or default_chat
    day = day or datetime.now(timezone.utc).date()
    goal = goals.ensure_goal(ch, day, goal_pct) if goal_pct else goals.ensure_goal(ch, day)
    achieved = goals.daily_profit_pct(ch)
    state = OptimizerState(goal_profit_pct=float(goal["target_profit_pct"]), max_loops=loops)
    budget = Budget(max_cost_usd=max_cost_usd, max_steps=max_steps)

    summary = {"loops": 0, "proposals": [], "requests": [], "adopted": 0,
               "goal": goal, "achieved_pct": achieved, "run_id": ""}
    with Tracer(ch).run(kind="optimizer",
                        goal={"target_profit_pct": float(goal["target_profit_pct"]),
                              "achieved_pct": achieved},
                        model_planner=planner_model,
                        model_executor=executor_model) as run:
        summary["run_id"] = run.run_id
        observed = _observe(ch)
        run.step("observe", {}, {"health": observed["health"]})
        budget.steps += 1

        for _ in range(loops):
            budget.check()
            system, user = _planner_prompt(observed, state)
            plan = _extract_json(_llm(chat_fn, "planner", system, user, run, budget).text)
            if plan:
                built = _extract_json(
                    _llm(chat_fn, "executor", _EXECUTOR_SYSTEM, json.dumps(plan),
                         run, budget).text)
                if built:
                    built.setdefault("model", executor_model)
                    plan = built
                else:
                    plan.setdefault("model", planner_model)
            outcome = _execute_plan(plan, ch, run, strategy, publish_adopted)
            state.loops += 1
            summary["loops"] += 1
            if outcome["action"] == "proposal":
                summary["proposals"].append(outcome)
                if outcome["decision"] == "adopted":
                    summary["adopted"] += 1
            elif outcome["action"] == "request":
                summary["requests"].append(outcome)

        run.loops = state.loops
        achieved_now = goals.daily_profit_pct(ch)
        goal_row = goals.update_goal(ch, day, achieved_now or 0.0)
        if achieved_now is not None and achieved_now >= float(goal_row["target_profit_pct"]):
            goal_row = goals.raise_goal(ch, day)
        summary["goal"] = goal_row
        summary["achieved_pct"] = achieved_now
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Autonomous strategy optimizer")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--strategy", default="mean_reversion")
    parser.add_argument("--max-cost", type=float, default=0.0, dest="max_cost")
    parser.add_argument("--max-steps", type=int, default=20, dest="max_steps")
    parser.add_argument("--no-publish", action="store_true",
                        help="evaluate proposals but do not publish adopted params")
    parser.add_argument("--observe-only", action="store_true",
                        help="gather state and exit (no LLM, no cost)")
    args = parser.parse_args(argv)

    if args.observe_only:
        observed = _observe(get_clickhouse_client())
        print(json.dumps({"health": observed["health"], "n_symbols": len(observed["symbols"])},
                         indent=2, default=str))
        return
    result = run_optimizer(loops=args.loops, strategy=args.strategy,
                           publish_adopted=not args.no_publish,
                           max_cost_usd=args.max_cost, max_steps=args.max_steps)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
