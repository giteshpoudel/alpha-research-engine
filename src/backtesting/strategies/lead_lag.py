"""Cross-asset lead-lag: BTC leads the majors/alts (momentum transmission).

Multi-series strategy (needs a leader series), so like ``funding_arb`` it is
special-cased in the tuner rather than living in ``registry.SIGNAL_FUNCS`` --
that keeps single-series consumers (paper, assignments) from ever dispatching
to it. Long the target while the leader's trailing return is positive; flat
once it turns negative. Adverse execution is the same for any strategy.
"""

from __future__ import annotations

import pandas as pd

DEFAULTS = {"lead_window": 6, "threshold": 0.0, "exit_threshold": 0.0}


def search(trial) -> dict:
    return {
        "lead_window": trial.suggest_int("lead_window", 3, 48),
        "threshold": trial.suggest_float("threshold", -0.005, 0.01),
        "exit_threshold": trial.suggest_float("exit_threshold", -0.01, 0.0),
    }


def signals(target_close: pd.Series, leader_close: pd.Series,
            lead_window: int = 6, threshold: float = 0.0,
            exit_threshold: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Entries when the leader's ``lead_window`` return exceeds ``threshold``;
    exits when it falls below ``exit_threshold``. Returns boolean Series."""
    lead_ret = leader_close.reindex(target_close.index).pct_change(lead_window)
    entries = (lead_ret > threshold).fillna(False).astype(bool)
    exits = (lead_ret < exit_threshold).fillna(False).astype(bool)
    return entries, exits
