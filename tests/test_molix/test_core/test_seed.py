"""Tests for molix.core.seed reproducibility helpers."""

from __future__ import annotations

import pickle

from molix.core.seed import make_generator, make_worker_init_fn, seed_everything


def test_worker_init_fn_is_picklable():
    """worker_init_fn must pickle for spawn/forkserver DataLoader workers.

    Regression: a local-closure implementation raised
    ``Can't pickle local object`` under Python 3.14's default forkserver,
    breaking any DataModule with num_workers > 0.
    """
    fn = make_worker_init_fn(42)
    restored = pickle.loads(pickle.dumps(fn))
    assert restored.base_seed == 42
    # callable and side-effect-free to invoke
    restored(0)


def test_seed_everything_returns_seed():
    assert seed_everything(123) == 123


def test_make_generator_is_reproducible():
    import torch

    g1 = make_generator(7)
    g2 = make_generator(7)
    a = torch.randint(0, 1_000_000, (5,), generator=g1)
    b = torch.randint(0, 1_000_000, (5,), generator=g2)
    assert torch.equal(a, b)
