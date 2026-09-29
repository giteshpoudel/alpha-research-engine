import math
from datetime import datetime, timezone

import pandas as pd

from src.backtesting.strategies import lead_lag, registry


def test_lead_lag_signals_follow_leader():
    leader = pd.Series([100.0, 100, 100, 110, 110, 100])
    entries, exits = lead_lag.signals(leader, leader, lead_window=1, threshold=0.0)
    assert entries.iloc[3]           # leader +10%
    assert exits.iloc[5]             # leader -9%
    assert not (entries & exits).any()
    assert entries.dtype == bool and exits.dtype == bool


def test_lead_lag_grid_registered():
    assert registry.grid("lead_lag")
    assert registry.grid("lead_lag@5m") == registry.grid("lead_lag")


def test_lead_lag_tuner_loads_leader_and_runs():
    from src.ingestion.schemas import get_clickhouse_client
    from src.meta_learning.tuner import _load_train_data, _run

    ch = get_clickhouse_client()
    df = _load_train_data(ch, "lead_lag@5m", "DOGE",
                          datetime(2025, 7, 1, tzinfo=timezone.utc),
                          datetime(2025, 7, 2, tzinfo=timezone.utc))
    assert "leader_close" in df.columns
    assert len(df) > 0
    result = _run(df, "lead_lag@5m", dict(lead_lag.DEFAULTS), fee=0.0002)
    assert math.isfinite(result.sharpe)
