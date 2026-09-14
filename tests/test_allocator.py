from datetime import datetime, timedelta, timezone

from src.meta_learning.allocator import allocator_windows, pick_strategy

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_pick_strategy_argmax():
    assert pick_strategy({"mean_reversion": 0.05, "funding_arb": 0.12}) == "funding_arb"
    assert pick_strategy({"mean_reversion": 0.2, "funding_arb": -0.1}) == "mean_reversion"


def test_pick_strategy_defaults_on_empty_history():
    assert pick_strategy({"mean_reversion": None, "funding_arb": None}) == "mean_reversion"
    assert pick_strategy({"mean_reversion": 0.0, "funding_arb": 0.0}) == "mean_reversion"


def test_allocator_windows():
    end = T0 + timedelta(days=20)
    windows = allocator_windows(T0, end, rebalance_days=7)
    assert windows[0] == (T0, T0 + timedelta(days=7))
    assert windows[-1][1] == end  # last window clamped
    for (_, a_end), (b_start, _) in zip(windows, windows[1:]):
        assert a_end == b_start  # contiguous, no gaps
