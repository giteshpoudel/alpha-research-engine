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
import time
from datetime import datetime, timezone

from src.agents import goals, proposals, tools
from src.agents.multi_llm import chat as default_chat
from src.agents.state import OptimizerState
from src.agents.tracing import Budget, Tracer
from src.ingestion.schemas import database_name, get_clickhouse_client

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
    '"symbol":"<TICKER>","strategy":"<name>",'
    '"hypothesis":"<one sentence>","analysis":"<one or two sentences>"}\n'
    '3) add a NEW strategy (paper-only, sandboxed): {"action":"add_strategy",'
    '"name":"<lowercase_snake>","code":"<python source>","symbol":"<TICKER>",'
    '"description":"<one sentence>"}\n'
    '   The code must define signals(close, **params) -> (entries, exits) boolean '
    'pandas Series, plus DEFAULTS dict (and optional SEARCH(trial) -> params dict). '
    'Imports allowed: numpy, pandas, math only. No while loops, no reflection.\n'
    '4) ask the human for data/API or a major change: {"action":"request",'
    '"kind":"data"|"api"|"change","title":"<short>",'
    '"justification":"<why it should improve profit>","expected_impact":"<estimate>"}\n'
    '5) do nothing: {"action":"none"}\n'
    "Strategies available: the built-ins (mean_reversion, momentum, breakout) plus "
    "any generated ones in \"strategies\". Match style to the symbol (momentum/"
    "breakout for trending majors; mean_reversion for choppy/penny/meme names). "
    "Prefer propose_params for small tweaks; switch_strategy when another existing "
    "strategy fits better; add_strategy when a genuinely new idea is warranted. "
    "Use request ONLY when new data/APIs or a major change is the real blocker."
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


def _validation_by_symbol(ch) -> dict[tuple[str, str], float]:
    rows = ch.query(
        f"SELECT symbol, strategy, argMax(validation_sharpe, valid_from) AS v "
        f"FROM {database_name()}.tuned_params FINAL GROUP BY symbol, strategy"
    ).result_rows
    return {(symbol, strategy): float(v) for symbol, strategy, v in rows}


def _focus_symbol(ch, observed: dict, avoid: set | None = None) -> tuple[str | None, str | None]:
    """Neediest symbol first: halted sleeve, then missing params, then lowest validation."""
    assignments = observed.get("assignments") or {}
    paper = {s["symbol"]: s for s in (observed.get("paper") or {}).get("sleeves", [])}
    valuation = _validation_by_symbol(ch)
    scored = []
    for symbol, strategy in assignments.items():
        halted = not paper.get(symbol, {}).get("enabled", True)
        value = valuation.get((symbol, strategy))
        score = -1.0 if halted else (0.0 if value is None else value)
        scored.append((score, symbol, strategy))
    scored.sort(key=lambda t: (t[0], t[1]))
    avoid = avoid or set()
    for score, symbol, strategy in scored:
        if symbol not in avoid:
            return symbol, strategy
    return (scored[0][1], scored[0][2]) if scored else (None, None)


def _tune_iteration(ch, run, strategy: str, symbol: str, n_trials: int,
                    publish_adopted: bool) -> dict:
    """Rapid deterministic Optuna tuning for one symbol, then champion/challenger."""
    fitted = tools.invoke("fit_params",
                          {"strategy": strategy, "symbol": symbol, "n_trials": n_trials},
                          ch=ch)
    if not fitted.ok or not fitted.data:
        run.step("tune", {"symbol": symbol, "strategy": strategy},
                 {"error": fitted.error})
        return {"action": "none"}
    change = {"strategy": strategy, "symbol": symbol, "params": fitted.data["params"]}
    proposal_id = proposals.create_proposal(
        ch, f"rapid-tune {strategy} {symbol}", change, kind="params",
        symbol=symbol, model="optuna", run_id=run.run_id)
    row = proposals.evaluate_proposal(ch, proposal_id, publish_adopted=publish_adopted,
                                      run_id=run.run_id)
    run.step("tune_proposal", {"proposal_id": proposal_id, "change": change},
             {"decision": row["decision"], "evidence": row["evidence"]})
    return {"action": "proposal", "kind": "params",
            "proposal_id": proposal_id, "decision": row["decision"]}


def _planner_prompt(observed: dict, state: OptimizerState,
                    focus: str | None = None) -> tuple[str, str]:
    user = json.dumps({"goal_profit_pct": state.goal_profit_pct,
                       "focus_symbol": focus, "observed": observed}, default=str)
    return _PLANNER_SYSTEM, user


def _llm(chat_fn, role: str, system: str, user: str, run, budget: Budget):
    result = chat_fn(role, system, user)
    run.step(f"llm_{role}", {"system": system, "user": user},
             {"model": result.model, "text": result.text[:1000]},
             tokens_in=result.tokens_in, tokens_out=result.tokens_out, model=result.model)
    budget.steps += 1
    budget.cost_usd += result.cost_usd
    return result


def _execute_plan(plan: dict | None, ch, run, strategy: str, publish_adopted: bool,
                  focus: str | None = None) -> dict:
    if not plan:
        run.step("no_action", {"plan": plan}, {"action": "none"})
        return {"action": "none"}
    if plan.get("action") == "add_strategy":
        from src.agents.strategy_gen import register_generated_strategy
        registered = register_generated_strategy(
            ch, plan.get("name", ""), plan.get("code", ""),
            plan.get("description", ""), plan.get("model", ""))
        run.step("add_strategy", {"name": plan.get("name")}, registered)
        if not registered.get("registered"):
            return {"action": "none", "errors": registered.get("errors")}
        symbol = plan.get("symbol") or focus
        if not symbol:
            return {"action": "registered", "name": registered["name"]}
        change = {"strategy": registered["name"], "symbol": symbol, "params": {}}
        proposal_id = proposals.create_proposal(
            ch, f"assign new strategy {registered['name']} to {symbol}", change,
            kind="strategy", symbol=symbol, model=plan.get("model", ""), run_id=run.run_id)
        row = proposals.evaluate_proposal(ch, proposal_id, publish_adopted=publish_adopted,
                                          run_id=run.run_id)
        run.step("proposal", {"proposal_id": proposal_id, "change": change},
                 {"decision": row["decision"], "evidence": row["evidence"]})
        return {"action": "proposal", "kind": "strategy", "proposal_id": proposal_id,
                "decision": row["decision"]}
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
                  day=None, focus_symbol: str | None = None, tune_trials: int = 0,
                  avoid: set | None = None) -> dict:
    ch = ch or get_clickhouse_client()
    chat_fn = chat_fn or default_chat
    day = day or datetime.now(timezone.utc).date()
    goal = goals.ensure_goal(ch, day, goal_pct) if goal_pct else goals.ensure_goal(ch, day)
    achieved = goals.daily_profit_pct(ch)
    state = OptimizerState(goal_profit_pct=float(goal["target_profit_pct"]), max_loops=loops)
    budget = Budget(max_cost_usd=max_cost_usd, max_steps=max_steps)
    assignments = {}

    summary = {"loops": 0, "proposals": [], "requests": [], "adopted": 0,
               "goal": goal, "achieved_pct": achieved, "run_id": "", "focus": None,
               "mode": "tune" if tune_trials else "llm"}
    with Tracer(ch).run(kind="optimizer",
                        goal={"target_profit_pct": float(goal["target_profit_pct"]),
                              "achieved_pct": achieved},
                        model_planner=planner_model,
                        model_executor=executor_model) as run:
        summary["run_id"] = run.run_id
        observed = _observe(ch)
        assignments = observed.get("assignments") or {}
        run.step("observe", {}, {"health": observed["health"]})
        budget.steps += 1

        focus, focus_strategy = focus_symbol, None
        if focus is None or tune_trials:
            focus, focus_strategy = _focus_symbol(ch, observed, avoid)
        summary["focus"] = focus
        if focus and not focus_strategy:
            focus_strategy = assignments.get(focus, strategy)

        if tune_trials and focus:
            budget.check()
            outcome = _tune_iteration(ch, run, focus_strategy, focus, tune_trials,
                                      publish_adopted)
            state.loops += 1
            summary["loops"] += 1
            if outcome["action"] == "proposal":
                summary["proposals"].append(outcome)
                if outcome["decision"] == "adopted":
                    summary["adopted"] += 1
        else:
            for _ in range(loops):
                budget.check()
                system, user = _planner_prompt(observed, state, focus)
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
                plan_symbol = plan.get("symbol") if plan else None
                plan_strategy = assignments.get(plan_symbol, strategy) if plan_symbol else strategy
                outcome = _execute_plan(plan, ch, run, plan_strategy, publish_adopted,
                                        focus=focus)
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


def run_continuous(ch=None, interval: float = 300.0, max_iterations: int = 0,
                   max_runtime_seconds: float = 86400.0, llm_every: int = 5,
                   tune_trials: int = 20, **kwargs) -> dict:
    """Rapid optimizer loop with a watchdog (bounds runaway loops/deadlocks).

    Most iterations are fast deterministic Optuna tuning; every ``llm_every``
    iterations the planner LLM runs for bigger-picture actions (switch/request).
    Exits after ``max_iterations`` or ``max_runtime_seconds`` (launchd restarts
    it fresh), so a stuck process can't spin forever.
    """
    # Do NOT create the client here: a transient DB outage at startup would
    # crash the worker. It is created (and any failure caught) per iteration.
    started = time.monotonic()
    recent: list[str] = []
    iterations = proposals_made = adopted = 0
    while True:
        if max_iterations and iterations >= max_iterations:
            break
        if max_runtime_seconds and (time.monotonic() - started) >= max_runtime_seconds:
            break
        use_llm = bool(llm_every) and iterations % llm_every == 0
        t0 = time.monotonic()
        try:
            result = run_optimizer(ch=ch, loops=1, focus_symbol=None,
                                   tune_trials=0 if use_llm else tune_trials,
                                   avoid=set(recent), **kwargs)
            proposals_made += len(result.get("proposals", []))
            adopted += result.get("adopted", 0)
            focus = result.get("focus")
            print(f"optimizer[{iterations}] {'llm' if use_llm else 'tune'} focus={focus} "
                  f"proposals={len(result.get('proposals', []))} adopted={result.get('adopted')} "
                  f"({time.monotonic() - t0:.1f}s)")
            if focus:
                recent = (recent + [focus])[-5:]
        except Exception as exc:
            print(f"optimizer[{iterations}] iteration failed: {exc}")
        iterations += 1
        if interval:
            time.sleep(interval)
    return {"iterations": iterations, "proposals": proposals_made, "adopted": adopted}


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
    parser.add_argument("--continuous", action="store_true",
                        help="run the rapid optimizer loop until limits")
    parser.add_argument("--interval", type=float, default=300.0,
                        help="seconds between iterations in continuous mode")
    parser.add_argument("--max-iterations", type=int, default=0, dest="max_iterations")
    parser.add_argument("--max-runtime-seconds", type=float, default=86400.0,
                        dest="max_runtime")
    parser.add_argument("--llm-every", type=int, default=5, dest="llm_every",
                        help="run the planner LLM every Nth continuous iteration")
    parser.add_argument("--tune-trials", type=int, default=20, dest="tune_trials",
                        help="Optuna trials per rapid-tune iteration")
    args = parser.parse_args(argv)

    if args.observe_only:
        observed = _observe(get_clickhouse_client())
        print(json.dumps({"health": observed["health"], "n_symbols": len(observed["symbols"])},
                         indent=2, default=str))
        return
    if args.continuous:
        result = run_continuous(
            interval=args.interval, max_iterations=args.max_iterations,
            max_runtime_seconds=args.max_runtime, llm_every=args.llm_every,
            tune_trials=args.tune_trials, strategy=args.strategy,
            publish_adopted=not args.no_publish, max_cost_usd=args.max_cost,
            max_steps=args.max_steps)
        print(json.dumps(result, indent=2, default=str))
        return
    result = run_optimizer(loops=args.loops, strategy=args.strategy,
                           publish_adopted=not args.no_publish,
                           max_cost_usd=args.max_cost, max_steps=args.max_steps)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
