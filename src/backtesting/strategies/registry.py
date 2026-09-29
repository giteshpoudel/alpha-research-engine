"""Registry of tunable signal strategies: name -> (signals fn, defaults, search space).

Keeps the tuner, paper runner, and optimizer strategy-agnostic: they dispatch by
name instead of hardcoding mean reversion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from src.backtesting.strategies import (
    bollinger,
    breakout,
    ema_cross,
    macd,
    mean_reversion,
    momentum,
    rsi_reversion,
)

GENERATED_DIR = Path(__file__).parent / "generated"

# name -> (signals function, default params)
SIGNAL_FUNCS: dict[str, tuple[Callable, dict]] = {
    "mean_reversion": (mean_reversion.signals, mean_reversion.MR_DEFAULTS),
    "momentum": (momentum.signals, momentum.MOMENTUM_DEFAULTS),
    "breakout": (breakout.signals, breakout.BREAKOUT_DEFAULTS),
    "rsi_reversion": (rsi_reversion.signals, rsi_reversion.RSI_DEFAULTS),
    "bollinger": (bollinger.signals, bollinger.BOLLINGER_DEFAULTS),
    "ema_cross": (ema_cross.signals, ema_cross.EMA_CROSS_DEFAULTS),
    "macd": (macd.signals, macd.MACD_DEFAULTS),
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
    "rsi_reversion": lambda t: {
        "window": t.suggest_int("window", 7, 30),
        "oversold": t.suggest_float("oversold", 15.0, 35.0),
        "overbought": t.suggest_float("overbought", 50.0, 70.0),
    },
    "bollinger": lambda t: {
        "window": t.suggest_int("window", 10, 60),
        "k": t.suggest_float("k", 1.0, 3.0),
    },
    "ema_cross": lambda t: {
        "fast": t.suggest_int("fast", 5, 30),
        "slow": t.suggest_int("slow", 20, 80),
    },
    "macd": lambda t: {
        "fast": t.suggest_int("fast", 5, 20),
        "slow": t.suggest_int("slow", 20, 60),
        "signal": t.suggest_int("signal", 5, 15),
    },
}

# Mutable list so importers that bound it keep seeing newly loaded strategies.
# Coarse param grids for fast vectorized sweeps (much faster than Optuna for
# grid-style strategies; used by tuner.fit_grid / batch --method grid).
GRIDS: dict[str, list[dict]] = {
    "mean_reversion": [{"window": w, "z_entry": ze, "z_exit": 0.0}
                       for w in (12, 24, 36, 48, 72)
                       for ze in (-1.0, -1.5, -2.0, -2.5, -3.0)],
    "momentum": [{"window": w, "threshold": 0.0} for w in (6, 12, 24, 48, 72)],
    "breakout": [{"entry_window": e, "exit_window": max(6, e // 2)}
                 for e in (12, 24, 48, 72)],
    "rsi_reversion": [{"window": w, "oversold": o, "overbought": 60.0}
                      for w in (7, 14, 21) for o in (20.0, 30.0)],
    "bollinger": [{"window": w, "k": k} for w in (10, 20, 40) for k in (1.5, 2.0, 2.5)],
    "ema_cross": [{"fast": f, "slow": s} for f in (5, 10, 20) for s in (30, 50, 80)],
    "macd": [{"fast": f, "slow": s, "signal": 9}
             for f in (8, 12, 16) for s in (21, 26, 35)],
    # cross-asset lead-lag (BTC leads); special-cased in the tuner, not a
    # single-series SIGNAL_FUNCS entry so paper/assignments can't dispatch to it.
    "lead_lag": [{"lead_window": w, "threshold": th, "exit_threshold": 0.0}
                 for w in (3, 6, 12, 24) for th in (-0.005, 0.0, 0.005)],
}

STRATEGY_NAMES: list[str] = list(SIGNAL_FUNCS)


def grid(strategy: str) -> list[dict]:
    return GRIDS.get(base_name(strategy), [])


# --- interval namespacing: "mean_reversion@5m" -> ("mean_reversion", "5m") ---
DEFAULT_INTERVAL = "1h"


def normalize(strategy: str) -> tuple[str, str]:
    """(base strategy, interval), stripping a TEST_ prefix and @interval suffix."""
    name = strategy.removeprefix("TEST_")
    if "@" in name:
        base, interval = name.split("@", 1)
        return base, interval
    return name, DEFAULT_INTERVAL


def base_name(strategy: str) -> str:
    return normalize(strategy)[0]


def interval_of(strategy: str) -> str:
    return normalize(strategy)[1]


def qualify(base: str, interval: str) -> str:
    return base if interval == DEFAULT_INTERVAL else f"{base}@{interval}"


def defaults(strategy: str) -> dict:
    return dict(SIGNAL_FUNCS[base_name(strategy)][1])


def signals(strategy: str, close, params: dict):
    fn = SIGNAL_FUNCS[base_name(strategy)][0]
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
