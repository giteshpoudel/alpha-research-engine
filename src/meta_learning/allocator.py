"""Causal trailing-performance strategy picker for the comparison simulation.

Picks depend only on data strictly before the rebalance point (no lookahead).
"""

from __future__ import annotations

from datetime import datetime, timedelta


def pick_strategy(trailing_returns: dict[str, float | None],
                  default: str = "mean_reversion") -> str:
    """Argmax over trailing returns; default when history is empty or all zero."""
    best, best_ret = default, None
    for strategy, ret in trailing_returns.items():
        if ret is not None and ret != 0.0 and (best_ret is None or ret > best_ret):
            best, best_ret = strategy, ret
    return best


def allocator_windows(oos_start: datetime, oos_end: datetime,
                      rebalance_days: int = 7, trailing_days: int = 30) -> list[tuple[datetime, datetime]]:
    """Contiguous (start, end) rebalance windows covering [oos_start, oos_end)."""
    windows = []
    cursor = oos_start
    while cursor < oos_end:
        end = min(cursor + timedelta(days=rebalance_days), oos_end)
        windows.append((cursor, end))
        cursor = end
    return windows
