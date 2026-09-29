import pandas as pd

from src.meta_learning.microstructure import order_flow_predictiveness


def _trades(rows):
    return pd.DataFrame(rows, columns=["ts", "price", "qty", "is_buyer_maker"])


def test_obi_predicts_next_bucket_return():
    # Bucket 1: pure buying at 100; bucket 2: price rises to 101.
    rows = [("2025-01-01 00:00:10", 100.0, 5.0, 0),
            ("2025-01-01 00:00:40", 100.0, 3.0, 0),
            ("2025-01-01 00:01:10", 101.0, 5.0, 1),
            ("2025-01-01 00:02:10", 101.0, 5.0, 1),
            ("2025-01-01 00:03:10", 102.0, 5.0, 1)]
    stats = order_flow_predictiveness(_trades(rows), horizon=1, bucket="1min")
    assert stats["n_buckets"] >= 2
    assert stats["corr"] is not None and stats["corr"] > 0


def test_empty_trades_is_safe():
    assert order_flow_predictiveness(_trades([])) == {"n_buckets": 0, "corr": None}
