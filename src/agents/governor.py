"""A single heavy-compute lock.

Batch tuning (8 workers) and the optimizer's parallel rounds would otherwise
contend for every core. ``heavy`` is a non-blocking exclusive lock: the first
holder runs; an overlapping launch is told to skip instead of piling on.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_LOCK_DIR = Path("data")
_LOCK_PATH = _LOCK_DIR / ".heavy.lock"


@contextmanager
def heavy(label: str = "heavy"):
    """Yield True if we acquired the compute lock, False if it's busy."""
    _LOCK_DIR.mkdir(exist_ok=True)
    handle = open(_LOCK_PATH, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        handle.write(f"{label} pid={os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        handle.flush()
        yield True
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()
