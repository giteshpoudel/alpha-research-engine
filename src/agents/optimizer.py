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

from src.agents import goals, proposals, scoring, tools
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
    "Strategies available: everything listed in \"strategies\" (built-ins plus any "
    "generated). Match style to the symbol (trend/momentum for trending majors; "
    "mean-reversion styles for choppy/penny/meme names). "
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

    # Offer both the 1h base strategies and their 5m intraday variants.
    strategies = list(STRATEGY_NAMES) + [f"{s}@5m" for s in STRATEGY_NAMES]
    return {
        "paper": tools.invoke("paper_summary", {}, ch=ch).data,
        "health": tools.invoke("system_health", {}, ch=ch).data,
        "symbols": tools.invoke("classify_universe", {}, ch=ch).data,
        "strategies": strategies,
        "assignments": list_assignments(ch),
    }


def _validation_by_symbol(ch) -> dict[tuple[str, str], float]:
    rows = ch.query(
        f"SELECT symbol, strategy, argMax(validation_sharpe, valid_from) AS v "
        f"FROM {database_name()}.tuned_params FINAL GROUP BY symbol, strategy"
    ).result_rows
    return {(symbol, strategy): float(v) for symbol, strategy, v in rows}


def _priority_pairs(ch, observed: dict,
                    avoid: set | None = None) -> list[tuple[str, str]]:
    """(symbol, assigned strategy) most-needy first; ``avoid`` moved to the back.

    Neediness: halted sleeve < missing params < lowest validation Sharpe.
    """
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
    return ([(s, st) for _, s, st in scored if s not in avoid]
            + [(s, st) for _, s, st in scored if s in avoid])


def _focus_symbol(ch, observed: dict, avoid: set | None = None) -> tuple[str | None, str | None]:
    pairs = _priority_pairs(ch, observed, avoid)
    return pairs[0] if pairs else (None, None)


def run_parallel_round(ch=None, round_size: int = 6, n_trials: int = 20,
                       workers: int | None = None) -> dict:
    """Tune the neediest ``round_size`` symbols in parallel (all cores)."""
    from src.agents import governor
    from src.meta_learning import batch as batch_mod

    ch = ch or get_clickhouse_client()
    observed = _observe(ch)
    pairs = _priority_pairs(ch, observed)[:round_size]
    combos = [(strategy, symbol) for symbol, strategy in pairs]
    if not combos:
        return {"combos": 0, "adopted": 0}
    with governor.heavy("optimizer_round") as acquired:
        if not acquired:
            return {"combos": 0, "adopted": 0, "skipped": "compute lock busy"}
        return batch_mod.run_batch(ch, combos=combos, n_trials=n_trials, workers=workers)


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
    from src.meta_learning import selection
    if not selection.passes_dsr(fitted.data.get("dsr")):
        run.step("dsr_reject", {"symbol": symbol, "strategy": strategy,
                                "dsr": fitted.data.get("dsr")},
                 {"action": "none", "reason": "below MIN_DSR_PROB"})
        return {"action": "none", "reason": "dsr_gate"}
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


def _planner_prompt(observed: dict, state: OptimizerState, focus: str | None = None,
                    prefer_add_strategy: bool = False) -> tuple[str, str]:
    payload = {"goal_profit_pct": state.goal_profit_pct,
               "focus_symbol": focus, "observed": observed}
    if prefer_add_strategy:
        payload["directive"] = ("This iteration, prioritize action 'add_strategy' if you "
                                "can justify a genuinely new idea; otherwise propose_params "
                                "or none.")
    return _PLANNER_SYSTEM, json.dumps(payload, default=str)


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
        from src.backtesting.strategies.registry import STRATEGY_NAMES, base_name
        symbol, strategy = plan.get("symbol"), plan.get("strategy")
        if symbol and base_name(strategy) in STRATEGY_NAMES:
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
                  avoid: set | None = None, prefer_add_strategy: bool = False) -> dict:
    ch = ch or get_clickhouse_client()
    if chat_fn is None:
        from src.agents.breakers import BreakerStore
        from src.agents.multi_llm import Cooldown
        cooldown = Cooldown(store=BreakerStore(ch))
        chat_fn = (lambda role, system, user:
                   default_chat(role, system, user, cooldown=cooldown))
    started_at = time.monotonic()
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

        if tune_trials and focus and not prefer_add_strategy:
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
                system, user = _planner_prompt(observed, state, focus, prefer_add_strategy)
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

    scoring.score_run(ch, summary["run_id"], adopted=summary["adopted"],
                      proposals=len(summary["proposals"]),
                      requests=len(summary["requests"]),
                      tokens_in=run.tokens_in, tokens_out=run.tokens_out,
                      cost_usd=run.cost_usd,
                      duration_ms=(time.monotonic() - started_at) * 1000, status="done")
    return summary


def run_continuous(ch=None, interval: float = 300.0, max_iterations: int = 0,
                   max_runtime_seconds: float = 86400.0, llm_every: int = 5,
                   synth_every: int = 10, synth_count: int = 2, round_size: int = 6,
                   tune_trials: int = 20, **kwargs) -> dict:
    """Rapid optimizer loop with a watchdog (bounds runaway loops/deadlocks).

    Non-LLM iterations tune a *batch* of needy symbols in parallel (all cores);
    every ``synth_every`` iterations it synthesizes new strategies; every
    ``llm_every`` the planner LLM runs for switch/request actions. Exits after
    ``max_iterations`` or ``max_runtime_seconds`` (launchd restarts it fresh).
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
        synth = bool(synth_every) and iterations > 0 and iterations % synth_every == 0
        use_llm = synth or (bool(llm_every) and iterations % llm_every == 0)
        mode = "synth" if synth else ("llm" if use_llm else
                                      ("round" if round_size > 1 else "tune"))
        t0 = time.monotonic()
        try:
            client = ch or get_clickhouse_client()
            if synth:
                from src.agents.strategy_synth import synthesize_batch
                names = synthesize_batch(client, synth_count, chat_fn=kwargs.get("chat_fn"))
                print(f"optimizer[{iterations}] synth registered={names}")
            if use_llm:
                result = run_optimizer(ch=client, loops=1, focus_symbol=None, tune_trials=0,
                                       avoid=set(recent), prefer_add_strategy=synth, **kwargs)
                proposals_made += len(result.get("proposals", []))
                adopted += result.get("adopted", 0)
                focus = result.get("focus")
            elif round_size > 1:
                rres = run_parallel_round(ch=client, round_size=round_size,
                                          n_trials=tune_trials)
                proposals_made += rres.get("combos", 0)
                adopted += rres.get("adopted", 0)
                focus = rres.get("skipped") or f"round({rres.get('combos', 0)})"
            else:
                result = run_optimizer(ch=client, loops=1, focus_symbol=None,
                                       tune_trials=tune_trials, avoid=set(recent), **kwargs)
                proposals_made += len(result.get("proposals", []))
                adopted += result.get("adopted", 0)
                focus = result.get("focus")
            print(f"optimizer[{iterations}] {mode} focus={focus} "
                  f"(total proposals={proposals_made} adopted={adopted}) "
                  f"({time.monotonic() - t0:.1f}s)")
            if isinstance(focus, str) and focus and not focus.startswith("round"):
                recent = (recent + [focus])[-5:]
        except Exception as exc:
            print(f"optimizer[{iterations}] iteration failed: {exc}")
        iterations += 1
        if iterations % 10 == 0:
            try:
                regression = scoring.detect_regression(ch or get_clickhouse_client())
                if regression:
                    print(f"optimizer regression: {regression}")
            except Exception:
                pass
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
    parser.add_argument("--synth-every", type=int, default=10, dest="synth_every",
                        help="every Nth iteration, prioritize new-strategy synthesis")
    parser.add_argument("--synth-count", type=int, default=2, dest="synth_count",
                        help="candidate strategies to generate per synth iteration")
    parser.add_argument("--round-size", type=int, default=6, dest="round_size",
                        help="symbols tuned in parallel per non-LLM round")
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
            synth_every=args.synth_every, synth_count=args.synth_count,
            round_size=args.round_size,
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
