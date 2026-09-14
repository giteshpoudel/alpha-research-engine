from datetime import datetime, timezone

import pytest

from src.meta_learning.tuner import TUNABLE_STRATEGIES, tune, walk_forward_folds
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_walk_forward_folds_structure():
    folds = walk_forward_folds()
    assert len(folds) == 8
    for train_start, train_end, val_start, val_end in folds:
        assert train_end == val_start
        assert (train_end - train_start).days in (365, 366)
        assert (val_end - val_start).days in (89, 90, 91, 92)
        # tripwire: everything inside IS (IS end candle is 2024-12-31 23:00)
        assert train_start >= datetime(2022, 1, 1, tzinfo=timezone.utc)
        assert val_end <= datetime(2025, 1, 1, tzinfo=timezone.utc)
    # validate windows never overlap
    val_windows = [(f[2], f[3]) for f in folds]
    for (a_start, a_end), (b_start, b_end) in zip(val_windows, val_windows[1:]):
        assert a_end <= b_start


def test_tune_stores_tuned_params():
    ch = get_clickhouse_client()
    try:
        result = tune(ch, "TEST_mean_reversion", "BTC", n_trials=5, seed=42)
        assert set(result) == {"params", "train_sharpe", "validation_sharpe", "folds"}
        assert result["folds"] == 8
        params = result["params"]
        assert 12 <= params["window"] <= 72
        assert -3.5 <= params["z_entry"] <= -1.0
        assert -0.5 <= params["z_exit"] <= 0.5
        rows = ch.query(
            f"SELECT params_json, folds FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = 'TEST_mean_reversion' AND symbol = 'BTC'"
        ).result_rows
        assert len(rows) == 1
        assert rows[0][1] == 8
        # re-tune replaces, never duplicates
        tune(ch, "TEST_mean_reversion", "BTC", n_trials=5, seed=42)
        rows2 = ch.query(
            f"SELECT count() FROM {database_name()}.tuned_params FINAL "
            "WHERE strategy = 'TEST_mean_reversion' AND symbol = 'BTC'"
        ).result_rows
        assert rows2[0][0] == 1
    finally:
        ch.command(
            f"ALTER TABLE {database_name()}.tuned_params DELETE WHERE strategy = 'TEST_mean_reversion'",
            settings={"mutations_sync": 1},
        )


def test_tune_funding_arb_skips_dataless_folds():
    ch = get_clickhouse_client()
    try:
        result = tune(ch, "TEST_funding_arb", "BTC", n_trials=3, seed=42)
        assert result["folds"] < 8  # early folds skipped (no funding data before 2023-05-12)
        assert result["folds"] >= 1
        assert "threshold" in result["params"]
        assert 1e-5 <= result["params"]["threshold"] <= 1e-3
    finally:
        ch.command(
            f"ALTER TABLE {database_name()}.tuned_params DELETE WHERE strategy = 'TEST_funding_arb'",
            settings={"mutations_sync": 1},
        )
