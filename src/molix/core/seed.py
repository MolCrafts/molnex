"""Reproducibility helpers: global seeding + DataLoader worker seeding.

A fresh run is only reproducible if *every* RNG that influences it is
seeded from one source: Python's ``random``, NumPy, torch (CPU + CUDA),
the train DataLoader's shuffle generator, and each worker process. This
module centralises that so a training script does one call::

    from molix.core.seed import seed_everything
    seed_everything(42, deterministic=True)

``deterministic=True`` additionally forces deterministic cuDNN/cuBLAS
kernels (``torch.use_deterministic_algorithms``) — slower, but required
to produce bit-comparable reference checkpoints across runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "make_worker_init_fn", "make_generator"]


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python / NumPy / torch (CPU + CUDA) from a single value.

    Args:
        seed: The base seed.
        deterministic: When ``True``, also enable deterministic algorithms
            (``torch.use_deterministic_algorithms(True)``, cuDNN
            deterministic, ``CUBLAS_WORKSPACE_CONFIG``) so repeated runs are
            bit-reproducible. Costs throughput; use for reference runs.

    Returns:
        The seed, for logging / echoing back.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before the first CUDA call to honour
        # deterministic matmul; setting it here covers the common case where
        # seed_everything() runs at program start.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU :class:`torch.Generator` seeded with ``seed``.

    Pass to ``DataLoader(generator=...)`` so shuffle order is reproducible
    and independent of the global RNG's consumption elsewhere.
    """
    return torch.Generator().manual_seed(seed)


class _WorkerInit:
    """Picklable ``worker_init_fn`` seeding each DataLoader worker uniquely.

    Must be a top-level class, not a closure: ``spawn`` / ``forkserver`` start
    methods pickle the ``worker_init_fn`` to send it to workers, and a local
    closure is unpicklable (``Can't pickle local object`` on Python 3.14's
    default forkserver). Each worker gets ``base_seed + worker_id`` across
    Python / NumPy / torch — reproducible yet decorrelated between workers.
    """

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)


def make_worker_init_fn(base_seed: int) -> "_WorkerInit":
    """Build a picklable ``worker_init_fn`` seeding each worker from ``base_seed``."""
    return _WorkerInit(base_seed)
