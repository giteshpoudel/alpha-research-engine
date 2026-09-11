def test_vectorbt_importable():
    import vectorbt as vbt
    assert vbt.__version__


def test_pandas_numpy_available():
    import numpy as np
    import pandas as pd
    s = pd.Series([1.0, 2.0, 3.0])
    assert float(np.nanmean(s)) == 2.0
