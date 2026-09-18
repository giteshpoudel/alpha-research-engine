from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research import signal_eval as se


# ---------- pure statistics (no DB) ----------


def test_rank_ic_perfect_and_undefined():
    assert se.rank_ic([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert se.rank_ic([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert se.rank_ic([1, 1, 1], [1, 2, 3]) is None  # zero variance
    assert se.rank_ic([1, 2], [1, 2]) is None  # n < 3


def test_pooled_ic_perfect_cross_sections():
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    panel = pd.DataFrame({
        "ts": [ts] * 3 + [ts.replace(hour=1)] * 3,
        "x": [1, 2, 3, 1, 2, 3],
        "y": [0.1, 0.2, 0.3, 0.2, 0.4, 0.6],
    })
    out = se.pooled_ic(panel)
    assert out["n"] == 2
    assert out["ic"] == pytest.approx(1.0)


def test_block_bootstrap_deterministic_and_noise_not_significant():
    series = [0.05, 0.1, -0.02, 0.08, 0.03, 0.06, 0.01]
    assert (se.block_bootstrap_p(series, block=2, seed=42, n_boot=800)
            == se.block_bootstrap_p(series, block=2, seed=42, n_boot=800))
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 60)
    assert se.block_bootstrap_p(noise, block=1, seed=1, n_boot=800) > 0.2


def test_quantile_spread_monotonic_and_small_sample_flag():
    out = se.quantile_spread(list(range(60)), [i * 0.01 for i in range(60)])
    assert out["insufficient"] is False
    assert out["monotonic"] is True
    assert out["spread"] > 0
    assert se.quantile_spread([1, 2, 3], [0.1, 0.2, 0.3])["insufficient"] is True


# ---------- integration ----------


@pytest.fixture(scope="module")
def ch_client():
    return get_clickhouse_client()


def test_load_sentiment_features_aligned_to_bucket_end(ch_client):
    df = se.load_sentiment_features(ch_client, "BTC", "1h")
    assert not df.empty
    assert {"weighted_score", "mean_score", "velocity", "engagement_ratio",
            "post_count"} <= set(df.columns)
    assert df.index.tz is not None
    assert all(ts.minute == 0 and ts.second == 0 for ts in df.index)


def test_assemble_panel_has_forward_returns(ch_client):
    panel = se.assemble_panel(ch_client, "BTC", "1h", horizons=(1, 4))
    assert not panel.empty
    assert {"ts", "ticker", "fwd_1", "fwd_4", "mom_24h"} <= set(panel.columns)
    assert set(panel["ticker"]) == {"BTC"}


def test_evaluate_and_store_idempotent(ch_client):
    rows = se.evaluate(ch_client, buckets=("1h",), horizons=(1,),
                       features=("weighted_score",), symbols=("BTC",))
    methods = {r["method"] for r in rows}
    assert {"coverage", "ic_pooled", "ic_ts", "quantile", "divergence"} <= methods
    for row in rows:
        assert set(row) == {"bucket_size", "feature", "horizon", "method", "value",
                            "n", "tstat", "insufficient", "detail"}

    def stored_count():
        return ch_client.query(
            f"SELECT count() FROM {database_name()}.signal_eval_results FINAL "
            "WHERE bucket_size = '1h' AND feature = 'weighted_score'"
        ).result_rows[0][0]

    before = stored_count()
    se.run_evaluation(ch_client, buckets=("1h",), horizons=(1,),
                      features=("weighted_score",), symbols=("BTC",))
    se.run_evaluation(ch_client, buckets=("1h",), horizons=(1,),
                      features=("weighted_score",), symbols=("BTC",))
    assert stored_count() == before  # Replace keys collapse → no duplicates
    assert before >= 4  # ic_pooled, ic_ts, quantile, divergence
