"""Deterministic tool registry for the autonomous optimizer.

Tools wrap existing modules (backtest/tune/compare/signal/paper/health/symbols).
Handlers are pure orchestration — no LLM inside — so a run is reproducible and
every call can be traced. The planner chooses tools by name; ``invoke``
validates arguments via the tool's Pydantic model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.agents.state import (
    AddSymbolArgs,
    BacktestArgs,
    ClassifyArgs,
    CompareArgs,
    CreateRequestArgs,
    EmptyArgs,
    RegisterStrategyArgs,
    FitArgs,
    SignalArgs,
    SymbolArg,
    ToolResult,
    ValidateArgs,
)
from src.ingestion.schemas import get_clickhouse_client


SCOPE_ORDER = ("read", "write", "admin")


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, Any], Any]
    scope: str = "read"


TOOL_REGISTRY: dict[str, Tool] = {}

# Capability boundary: read = no mutation, write = mutates state, admin = code.
TOOL_SCOPES: dict[str, str] = {
    "run_backtest": "write",
    "fit_params": "read",
    "validate_params": "read",
    "compare_variants": "read",
    "evaluate_signal": "read",
    "paper_summary": "read",
    "system_health": "read",
    "classify_universe": "read",
    "list_universe": "read",
    "add_symbol": "write",
    "remove_symbol": "write",
    "create_request": "write",
    "list_requests": "read",
    "register_strategy": "admin",
}


def register(name: str, description: str, args_model: type[BaseModel]):
    def deco(fn):
        TOOL_REGISTRY[name] = Tool(name, description, args_model, fn,
                                   scope=TOOL_SCOPES.get(name, "read"))
        return fn
    return deco


def _client(ch):
    return ch or get_clickhouse_client()


def _tool_cooldown(ch):
    from src.agents.breakers import BreakerStore
    from src.agents.multi_llm import Cooldown
    # Threshold is generous: legitimate per-symbol failures (e.g. no data)
    # shouldn't trip it; 5 fast failures mean something systemic (e.g. DB down).
    return Cooldown(threshold=5, seconds=60.0, store=BreakerStore(ch))


def describe() -> list[dict]:
    return [{"name": t.name, "scope": t.scope, "description": t.description}
            for t in TOOL_REGISTRY.values()]


@register("run_backtest", "Run backtest(s) for a strategy and store results.", BacktestArgs)
def _run_backtest(args: BacktestArgs, ch) -> list[dict]:
    from src.backtesting.runner import run_backtest
    from src.ingestion.tickers import TICKER_ALIASES

    ch = _client(ch)
    symbols = tuple(TICKER_ALIASES) if args.symbols in ([], ["all"]) else tuple(args.symbols)
    return [{"symbol": s, "run_id": run_backtest(ch, args.strategy, s, args.window)}
            for s in symbols]


@register("fit_params", "Walk-forward tune params (no publish).", FitArgs)
def _fit_params(args: FitArgs, ch) -> dict:
    from src.meta_learning.tuner import fit

    record, _ = fit(_client(ch), args.strategy, args.symbol, n_trials=args.n_trials)
    return {"params": record["params"], "validation_sharpe": record["validation_sharpe"],
            "folds": record["folds"]}


@register("validate_params", "Validation Sharpe of fixed params over the IS folds.", ValidateArgs)
def _validate_params(args: ValidateArgs, ch) -> dict:
    from src.meta_learning.tuner import validate_params, walk_forward_folds

    value = validate_params(_client(ch), args.strategy, args.symbol, args.params,
                            walk_forward_folds())
    return {"validation_sharpe": value}


@register("compare_variants", "Static vs tuned OOS comparison for one symbol.", CompareArgs)
def _compare_variants(args: CompareArgs, ch) -> dict:
    from src.backtesting.strategies.mean_reversion import MR_DEFAULTS
    from src.meta_learning.compare import _execute_variant, _tuned_params

    ch = _client(ch)
    tuned = _tuned_params(ch, args.strategy, args.symbol)
    out = {}
    for name, params in (("static", dict(MR_DEFAULTS)), ("tuned", tuned)):
        _, _, result = _execute_variant(ch, args.strategy, args.symbol, params, 0.001)
        out[name] = {"total_return": result.total_return, "sharpe": result.sharpe,
                     "max_drawdown": result.max_drawdown}
    return out


@register("evaluate_signal", "Sentiment signal evaluation; returns significant ICs.", SignalArgs)
def _evaluate_signal(args: SignalArgs, ch) -> dict:
    from src.research.signal_eval import evaluate

    rows = evaluate(_client(ch), buckets=tuple(args.buckets),
                    horizons=tuple(args.horizons),
                    symbols=tuple(args.symbols) or None)
    significant = [r for r in rows if r["method"] == "ic_pooled" and not r["insufficient"]]
    significant.sort(key=lambda r: -abs(r["value"]))
    return {"n_rows": len(rows), "top_ic": significant[:5]}


@register("paper_summary", "Paper-trading sleeves, halts, param version.", EmptyArgs)
def _paper_summary(args: EmptyArgs, ch) -> dict:
    from src.research.collectors import collect_portfolio_data
    return collect_portfolio_data(_client(ch))


@register("system_health", "Latest data timestamp per pipeline stage.", EmptyArgs)
def _system_health(args: EmptyArgs, ch) -> dict:
    from src.dashboard.queries import system_health
    health = system_health(_client(ch))
    return {k: (str(v) if v is not None else None) for k, v in health.items()}


@register("classify_universe", "Price bucket / category / risk per symbol.", ClassifyArgs)
def _classify_universe(args: ClassifyArgs, ch) -> list[dict]:
    from src.agents.symbols import classify_universe
    return classify_universe(_client(ch), symbols=tuple(args.symbols) or None)


@register("list_universe", "Tradable universe membership and aliases.", EmptyArgs)
def _list_universe(args: EmptyArgs, ch) -> list[dict]:
    from src.ingestion.universe import list_universe
    return list_universe(_client(ch))


@register("add_symbol", "Add a symbol to the universe and backfill its data.", AddSymbolArgs)
def _add_symbol(args: AddSymbolArgs, ch) -> dict:
    from src.ingestion.universe import add_symbol
    return add_symbol(_client(ch), args.symbol, aliases=args.aliases or None,
                      backfill=args.backfill)


@register("remove_symbol", "Disable a symbol in the universe.", SymbolArg)
def _remove_symbol(args: SymbolArg, ch) -> dict:
    from src.ingestion.universe import remove_symbol
    return remove_symbol(_client(ch), args.symbol)


@register("create_request", "File a request to the human (data/API/major change) with justification.", CreateRequestArgs)
def _create_request(args: CreateRequestArgs, ch) -> dict:
    from src.agents.requests import create_request
    return {"request_id": create_request(
        _client(ch), args.title, args.justification, kind=args.kind,
        expected_impact=args.expected_impact)}


@register("list_requests", "List optimizer requests to the human.", EmptyArgs)
def _list_requests(args: EmptyArgs, ch) -> list[dict]:
    from src.agents.requests import list_requests
    return list_requests(_client(ch))


@register("register_strategy", "Sandbox-validate and register an agent-generated strategy (paper-only).", RegisterStrategyArgs)
def _register_strategy(args: RegisterStrategyArgs, ch) -> dict:
    from src.agents.strategy_gen import register_generated_strategy
    return register_generated_strategy(_client(ch), args.name, args.code,
                                       args.description, args.model)


def invoke(name: str, args: dict | None = None, ch=None,
           allow: tuple[str, ...] | list[str] | None = None) -> ToolResult:
    """Validate args, enforce scope, and run a tool; never raises.

    ``allow`` is the caller's capability set (default: all scopes). A tool whose
    scope isn't allowed is refused before any work happens. Repeated failures
    open a short breaker for that tool.
    """
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return ToolResult(ok=False, error=f"unknown tool: {name}")
    if allow is not None and tool.scope not in set(allow):
        return ToolResult(ok=False, error=f"scope denied: '{tool.scope}' requires "
                                          f"one of {sorted(set(allow))}")
    try:
        parsed = tool.args_model(**(args or {}))
    except ValidationError as exc:
        return ToolResult(ok=False, error=f"invalid args: {exc}")

    breaker = _tool_cooldown(ch) if ch is not None else None
    key = f"tool:{name}"
    if breaker is not None and breaker.should_skip(key):
        return ToolResult(ok=False, error=f"circuit open for {name}")

    start = time.perf_counter()
    try:
        data = tool.handler(parsed, ch)
        if breaker is not None:
            breaker.record_success(key)
        return ToolResult(ok=True, data=data,
                          latency_ms=(time.perf_counter() - start) * 1000)
    except Exception as exc:  # tools are isolated: a failure never kills the run
        if breaker is not None:
            breaker.record_failure(key)
        return ToolResult(ok=False, error=str(exc),
                          latency_ms=(time.perf_counter() - start) * 1000)
