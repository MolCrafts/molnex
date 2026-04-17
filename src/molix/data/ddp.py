"""Opt-in DDP helpers for workflow-driven cache materialization.

The caching APIs in :mod:`molix.data.cache` are process-local and have no
built-in rank coordination. For distributed launches, workflows typically
want rank 0 to build the cache while other ranks wait for it to appear. This
module provides the two primitives that pattern needs:

``rank()``            — read the current rank from the ``RANK`` env var.
``wait_for_ready()``  — block until a cache file exists.

Recommended use::

    from molix.data.ddp import rank, wait_for_ready
    from molix.data.cache import cache, is_ready

    if rank() == 0 and not is_ready(sink):
        cache(pipeline, source, sink=sink, fit_source=train)
    else:
        wait_for_ready(sink)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from molix.data.cache import is_ready


__all__ = ["rank", "wait_for_ready"]


def rank() -> int:
    """Read the distributed rank from ``$RANK`` (default 0, malformed → 0)."""
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def wait_for_ready(
    sink: str | Path,
    *,
    timeout: float = 600.0,
    poll_interval: float = 0.5,
) -> None:
    """Block until :func:`is_ready` returns True for *sink*, or raise.

    Args:
        sink: Cache file the caller is waiting on.
        timeout: Maximum seconds to wait before raising
            :class:`TimeoutError`.
        poll_interval: Seconds between readiness probes.
    """
    sink = Path(sink)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_ready(sink):
            return
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting {timeout:.0f}s for cache at {sink}. "
        "cache() must be driven from rank 0 (e.g. a prepare_data stage) "
        "before workers start."
    )
