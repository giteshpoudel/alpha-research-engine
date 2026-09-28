"""Registry of tunable signal strategies: name -> (signals fn, defaults, search space).

Keeps the tuner, paper runner, and optimizer strategy-agnostic: they dispatch by
name instead of hardcoding mean reversion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from src.backtesting.strategies import breakout, mean_reversion, momentum

GENERATED_DIR = Path(__file__).parent / "generated"

# name -> (signals function, default params)
SIGNAL_FUNCS: dict[str, tuple[Callable, dict]] = {
    "mean_reversion": (mean_reversion.signals, mean_reversion.MR_DEFAULTS),
    "momentum": (momentum.signals, momentum.MOMENTUM_DEFAULTS),
    "breakout": (breakout.signals, breakout.BREAKOUT_DEFAULTS),
}

# name -> optuna suggestion callable
SEARCH_SPACES: dict[str, Callable] = {
    "mean_reversion": lambda t: {
        "window": t.suggest_int("window", 12, 72),
        "z_entry": t.suggest_float("z_entry", -3.5, -1.0),
        "z_exit": t.suggest_float("z_exit", -0.5, 0.5),
    },
    "momentum": lambda t: {
        "window": t.suggest_int("window", 6, 96),
        "threshold": t.suggest_float("threshold", -0.02, 0.04),
    },
    "breakout": lambda t: {
        "entry_window": t.suggest_int("entry_window", 12, 72),
        "exit_window": t.suggest_int("exit_window", 6, 48),
    },
}

# Mutable list so importers that bound it keep seeing newly loaded strategies.
STRATEGY_NAMES: list[str] = list(SIGNAL_FUNCS)


def defaults(strategy: str) -> dict:
    return dict(SIGNAL_FUNCS[strategy][1])


def signals(strategy: str, close, params: dict):
    fn = SIGNAL_FUNCS[strategy][0]
    return fn(close, **params)


_GENERATED_LOADED: set[str] = set()


def load_generated() -> list[str]:
    """Load agent-generated strategies from ``strategies/generated/*.py``.

    These modules passed the sandbox (AST + subprocess smoke test) at
    registration time. Registered in place so existing importers see them;
    removed files are pruned back out.
    """
    current: set[str] = set()
    if GENERATED_DIR.is_dir():
        for path in sorted(GENERATED_DIR.glob("*.py")):
            name = path.stem
            if name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"generated_{name}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if not callable(getattr(module, "signals", None)):
                    continue
                SIGNAL_FUNCS[name] = (module.signals, dict(getattr(module, "DEFAULTS", {})))
                search = getattr(module, "SEARCH", None)
                if callable(search):
                    SEARCH_SPACES[name] = search
                current.add(name)
            except Exception:
                continue
    for stale in _GENERATED_LOADED - current:
        SIGNAL_FUNCS.pop(stale, None)
        SEARCH_SPACES.pop(stale, None)
    _GENERATED_LOADED.clear()
    _GENERATED_LOADED.update(current)
    STRATEGY_NAMES[:] = list(SIGNAL_FUNCS)
    return sorted(current)


load_generated()
