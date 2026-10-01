import pytest

from src.agents import governor


def test_heavy_lock_is_exclusive():
    with governor.heavy("first") as acquired:
        if not acquired:
            pytest.skip("compute lock held by a live batch/optimizer job")
        assert acquired is True
        with governor.heavy("second") as nested:
            assert nested is False  # already held
    with governor.heavy("third") as again:
        assert again is True  # released
