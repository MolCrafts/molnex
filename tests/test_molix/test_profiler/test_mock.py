"""Tests for :mod:`molix.profiler.mock` — the ``seed`` reproducibility promise.

Both generators document ``seed`` as making their output reproducible. That
must cover the *drawn shapes* (atom / edge counts), not only the tensor
values: any golden built on mock data is otherwise unstable within a single
process.
"""

from __future__ import annotations

from tensordict import TensorDict

from molix.profiler import MockBatch, MockSource


def _shapes(batch: TensorDict) -> tuple[tuple[int, ...], ...]:
    """Shape signature of one mock batch.

    Args:
        batch: Nested batch from :meth:`MockBatch.__call__`.

    Returns:
        Shapes of ``atoms.Z`` ``(N,)``, ``atoms.pos`` ``(N, 3)``,
        ``edges.edge_index`` ``(E, 2)`` and ``graphs.num_atoms`` ``(B,)``.
    """
    return (
        tuple(batch["atoms", "Z"].shape),
        tuple(batch["atoms", "pos"].shape),
        tuple(batch["edges", "edge_index"].shape),
        tuple(batch["graphs", "num_atoms"].shape),
    )


class TestMockSource:
    """``seed`` fixes the per-sample atom counts, not just the tensor values."""

    def test_seed_makes_atom_counts_reproducible(self):
        """Two same-seed sources built in one process yield identical atom counts."""
        first = MockSource(n_samples=8, n_atoms=(5, 20), seed=0)
        second = MockSource(n_samples=8, n_atoms=(5, 20), seed=0)

        assert [len(first[i]["Z"]) for i in range(8)] == [len(second[i]["Z"]) for i in range(8)]


class TestMockBatch:
    """``seed`` fixes the drawn shapes of every call, not just the tensor values."""

    def test_seed_makes_shapes_reproducible(self):
        """Two same-seed factories emit the same shape sequence over repeated calls."""
        first = MockBatch(n_atoms=(8, 32), n_edges=(16, 64), seed=0)
        second = MockBatch(n_atoms=(8, 32), n_edges=(16, 64), seed=0)

        assert [_shapes(first()) for _ in range(4)] == [_shapes(second()) for _ in range(4)]
