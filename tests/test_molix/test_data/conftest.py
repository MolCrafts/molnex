"""Shared fixtures for the ``molix.data`` unit tests.

Currently holds one builder: the degenerate ``n_edges == n_atoms``
geometry used by ``test_cache.py``, ``test_collate_packed.py`` and
``test_dataset.py`` to pin the packed-cache bucket-identity contract.
The three modules assert three consequences of the *same* contract, so
the geometry lives here rather than being copied (and drifting) three
times.
"""

from __future__ import annotations

import torch


def equal_count_samples(n: int = 3) -> list[dict]:
    """Build *n* samples with ``n_edges == n_atoms == 2`` in every sample.

    A 2-atom molecule with a single bidirectional pair (``NeighborList``'s
    default ``symmetry=True`` → ``E = 2 x n_pairs``) is the smallest
    physically real case where the per-sample edge count collides with the
    atom count. Nothing in the leading dim then distinguishes a per-atom
    key from a per-edge key, so the packed schema must decide bucket
    membership by key role, not by numerology.

    Every value is hand-built (no RNG), so the fixture is bit-identical
    across runs and independent of global torch seed state.

    Args:
        n: Number of samples.

    Returns:
        Flat sample dicts with per-atom ``Z`` ``(2,)`` / ``pos`` ``(2, 3)``,
        per-edge ``edge_index`` ``(2, 2)`` / ``edge_diff`` ``(2, 3)`` /
        ``edge_dist`` ``(2,)``, and graph-level ``targets.U0`` ``(1,)``
        carrying the sample's identity ``float(i)``.
    """
    samples: list[dict] = []
    for i in range(n):
        # Atom 1 sits at +1 A along x from atom 0; the pair is stored in both
        # directions, so edge_diff = pos[target] - pos[source] flips sign.
        samples.append(
            {
                "Z": torch.tensor([1, 6], dtype=torch.long),
                "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "edge_diff": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                "edge_dist": torch.tensor([1.0, 1.0]),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples
