"""Registry of tunable signal strategies: name -> (signals fn, defaults, search space).

Keeps the tuner, paper runner, and optimizer strategy-agnostic: they dispatch by
name instead of hardcoding mean reversion.
"""

from __future__ import annotations

from typing import Callable

from src.backtesting.strategies import breakout, mean_reversion, momentum

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

STRATEGY_NAMES: tuple[str, ...] = tuple(SIGNAL_FUNCS)


def defaults(strategy: str) -> dict:
    return dict(SIGNAL_FUNCS[strategy][1])


def signals(strategy: str, close, params: dict):
    fn = SIGNAL_FUNCS[strategy][0]
    return fn(close, **params)
