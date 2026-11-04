import numpy as np
import pandas as pd

from src.research import breakout_new_coins as bn


def _df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.003, "low": c * 0.997, "close": c, "volume": 1.0},
                        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


def test_weights_do_not_use_the_future():
    a = _df()
    b = a.copy()
    b.iloc[2000:, :4] *= 1.5
    np.testing.assert_array_equal(bn.weights(a, 4)[:2000], bn.weights(b, 4)[:2000])


def test_weight_is_capped_at_a_quarter():
    assert bn.weights(_df(), 4).max() <= 0.25 + 1e-12
