import numpy as np
import pandas as pd

from src.research import batch3 as b3


def _flow(hours=300, seed=0):
    rng = np.random.default_rng(seed)
    H, NB = hours, b3.NBINS
    cb = rng.poisson(0.05, (H, NB)).astype(float)
    tb = cb * 10 ** np.linspace(-2, 8, NB)[None, :]
    sb = tb * rng.uniform(-1, 1, (H, 1))
    return {"hours": pd.date_range("2026-08-06", periods=H, freq="1h"), "cnt_bins": cb,
            "signed_bins": sb, "tot_bins": tb}


def test_features_use_only_past_and_current_hour():
    f1 = b3.features(_flow())
    fl = _flow()
    fl["signed_bins"][200:] *= -1          # change the future from hour 200 on
    fl["cnt_bins"][200:] += 5
    fl["tot_bins"][200:] *= 3
    f2 = b3.features(fl)
    pd.testing.assert_frame_equal(f1.iloc[:200], f2.iloc[:200])


def test_ofi_range_and_warmup():
    f = b3.features(_flow())
    assert f["ofi_1h"].dropna().between(-1, 1).all()
    assert f["ofi_z_4h"].iloc[:b3.MIN_HIST].isna().all()


def test_target_starts_at_next_open():
    opens = pd.Series(np.arange(1.0, 21.0))
    t = b3.targets(opens, 4)
    assert t.iloc[0] == opens.iloc[5] / opens.iloc[1] - 1


def test_entries_do_not_overlap_and_follow_direction():
    z = pd.Series([0, 2, 2, 2, 2, 2, -2, 0, 2, 0], index=pd.date_range("2026-10-01", periods=10, freq="1h"))
    e = b3.entries_from_signal(z, 1.0, 4, z.index[0], z.index[-1] + pd.Timedelta(hours=1))
    assert e == [1, 5]          # the hold from 5 covers hours 5-8
    assert b3.entries_from_signal(z, -1.0, 4, z.index[0], z.index[-1] + pd.Timedelta(hours=1)) == [6]


def test_gap_hours():
    c = pd.Series([5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4], index=pd.date_range("2026-10-01", periods=12, freq="1h"))
    assert len(b3.gap_hours(c)) == 7
