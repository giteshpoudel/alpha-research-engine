"""Generate candidate strategies with the executor LLM; sandbox-register the valid ones.

The LLM proposes code; ``register_generated_strategy`` enforces the AST allowlist
and subprocess smoke test, so only safe, contract-conforming strategies enter the
registry (paper-only).
"""

from __future__ import annotations

import os
import re
import time

from src.agents.multi_llm import chat as default_chat
from src.agents.strategy_gen import register_generated_strategy

_SYSTEM = (
    "You are a quant researcher. Write ONE new crypto trading strategy as Python. "
    "Respond with ONLY the Python code (no prose, no markdown fences). "
    "It must define a module-level DEFAULTS dict and "
    "signals(close, **params) -> (entries, exits), where entries/exits are boolean "
    "pandas Series, plus optional SEARCH(trial) -> params dict for Optuna. "
    "Imports allowed: pandas, numpy, math. Long-only. No loops, no I/O."
)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    starts = [i for i in (code.find("import "), code.find("from "),
                          code.find("DEFAULTS"), code.find("def ")) if i >= 0]
    return code[min(starts):].strip() if starts else code.strip()


def max_untested() -> int:
    """Synthesis pauses once this many generated strategies await tuning."""
    return int(os.environ.get("SYNTH_MAX_UNTESTED", "24"))


def synthesize_batch(ch, n: int = 2, chat_fn=None, model: str = "",
                     prefix: str = "gen") -> list[str]:
    """Ask the executor for ``n`` strategy ideas; register the ones that pass."""
    from src.meta_learning.performance import untested_generated

    # Don't outrun the tuner: pause synthesis while untested strategies pile up.
    try:
        if len(untested_generated(ch)) >= max_untested():
            return []
    except Exception:
        pass
    chat_fn = chat_fn or default_chat
    registered: list[str] = []
    for i in range(n):
        try:
            result = chat_fn("executor", _SYSTEM,
                             f"Strategy idea #{i + 1}: be distinct and simple.")
            code = _extract_code(result.text)
            name = f"{prefix}{int(time.time())}{i}"
            outcome = register_generated_strategy(
                ch, name, code, description="agent-synthesized",
                model=getattr(result, "model", model))
            if outcome.get("registered"):
                registered.append(name)
        except Exception:
            continue
    return registered
