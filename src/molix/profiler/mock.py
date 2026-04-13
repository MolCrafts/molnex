"""Synthetic data generators for isolated module profiling.

Provides two generators:

- :class:`MockBatch` — callable that produces a :class:`~molix.data.types.GraphBatch`
  with configurable (or random) atom / edge / graph counts.  Used with
  :class:`~molix.profiler.module.ModuleProfiler`.

- :class:`MockSource` — implements the
  :class:`~molix.data.source.DataSource` protocol and returns plain
  ``{"Z": ..., "pos": ...}`` sample dicts.  Used with
  :class:`~molix.profiler.task.TaskProfiler` and
  :class:`~molix.profiler.dataloader.DataLoaderProfiler`.

Example::

    from molix.profiler.mock import MockBatch, MockSource

    # Fixed shape — same batch every call
    factory = MockBatch(n_atoms=64, n_edges=512, n_graphs=8)
    batch = factory()   # -> GraphBatch

    # Variable shape — drawn from a range on each call
    factory = MockBatch(n_atoms=(30, 100), n_edges=(100, 600), n_graphs=(2, 8))
    batch = factory()

    # Mock DataSource
    source = MockSource(n_samples=500, n_atoms=(5, 20))
    sample = source[0]  # -> {"Z": tensor, "pos": tensor}
"""

from __future__ import annotations

import random
from typing import Union

import torch

from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData

# Type alias: int means fixed; tuple[int, int] means sample from [lo, hi]
_IntOrRange = Union[int, tuple[int, int]]


def _resolve(value: _IntOrRange) -> int:
    """Sample a concrete integer from a fixed value or (lo, hi) range.

    Args:
        value: Either a fixed ``int`` or a ``(lo, hi)`` inclusive range.

    Returns:
        A concrete integer.
    """
    if isinstance(value, int):
        return value
    lo, hi = value
    return random.randint(lo, hi)


# ---------------------------------------------------------------------------
# MockBatch
# ---------------------------------------------------------------------------


class MockBatch:
    """Callable factory that generates synthetic :class:`~molix.data.types.GraphBatch` instances.

    Useful for profiling model forward/backward passes without a real dataset.
    Shapes can be fixed or randomised per call to stress-test variable-size inputs.

    Args:
        n_atoms: Total atom count, or ``(lo, hi)`` range sampled per call.
        n_edges: Total edge count, or ``(lo, hi)`` range sampled per call.
        n_graphs: Number of graphs in the batch, or ``(lo, hi)`` range.
        atomic_numbers: Number of distinct element types (controls ``Z`` range).
        device: Device for generated tensors.
        seed: Optional seed for reproducibility.  If ``None``, each call is random.

    Example::

        factory = MockBatch(n_atoms=64, n_edges=512, n_graphs=8)
        batch = factory()   # produces a GraphBatch

        # Variable sizes — good for stress testing
        factory = MockBatch(n_atoms=(32, 128), n_edges=(128, 1024), n_graphs=(2, 16))
        for _ in range(10):
            batch = factory()   # different shape each time
    """

    def __init__(
        self,
        n_atoms: _IntOrRange = 32,
        n_edges: _IntOrRange = 128,
        n_graphs: _IntOrRange = 4,
        *,
        atomic_numbers: int = 10,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.n_atoms = n_atoms
        self.n_edges = n_edges
        self.n_graphs = n_graphs
        self.atomic_numbers = atomic_numbers
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self._torch_gen = torch.Generator(device=self.device)
        if seed is not None:
            self._torch_gen.manual_seed(seed)

    def __call__(self) -> GraphBatch:
        """Generate a fresh :class:`~molix.data.types.GraphBatch`.

        Returns:
            A ``GraphBatch`` with random tensor values and the configured shape.
        """
        n_a = _resolve(self.n_atoms)
        n_e = _resolve(self.n_edges)
        n_g = _resolve(self.n_graphs)

        dev = self.device
        gen = self._torch_gen

        # --- Atom data ---
        Z = torch.randint(1, self.atomic_numbers + 1, (n_a,), device=dev, generator=gen)
        pos = torch.randn(n_a, 3, device=dev, generator=gen)
        # Distribute atoms across graphs (roughly equal split)
        batch_vec = torch.zeros(n_a, dtype=torch.long, device=dev)
        if n_g > 1 and n_a > 0:
            boundaries = sorted(
                self._rng.sample(range(1, n_a), min(n_g - 1, n_a - 1))
            )
            for graph_idx, start in enumerate(boundaries):
                batch_vec[start:] = graph_idx + 1

        atoms = AtomData({"Z": Z, "pos": pos, "batch": batch_vec}, batch_size=[n_a])

        # --- Edge data ---
        if n_e > 0 and n_a > 1:
            # Random edges (source, target) within [0, n_a)
            src = torch.randint(0, n_a, (n_e,), device=dev, generator=gen)
            dst = torch.randint(0, n_a, (n_e,), device=dev, generator=gen)
            edge_index = torch.stack([src, dst], dim=1)  # (E, 2)
            bond_diff = torch.randn(n_e, 3, device=dev, generator=gen)
            bond_dist = torch.rand(n_e, device=dev, generator=gen) * 5.0
        else:
            edge_index = torch.zeros(0, 2, dtype=torch.long, device=dev)
            bond_diff = torch.zeros(0, 3, device=dev)
            bond_dist = torch.zeros(0, device=dev)
            n_e = 0

        edges = EdgeData(
            {"edge_index": edge_index, "bond_diff": bond_diff, "bond_dist": bond_dist},
            batch_size=[n_e],
        )

        # --- Graph data ---
        num_atoms_per_graph = torch.bincount(batch_vec, minlength=n_g).long()
        graphs = GraphData({"num_atoms": num_atoms_per_graph}, batch_size=[n_g])

        return GraphBatch({"atoms": atoms, "edges": edges, "graphs": graphs}, batch_size=[])

    def describe(self) -> str:
        """Return a human-readable description of the batch shape configuration.

        Returns:
            Description string.
        """
        def _fmt(v: _IntOrRange) -> str:
            return str(v) if isinstance(v, int) else f"{v[0]}–{v[1]}"

        return (
            f"MockBatch(n_atoms={_fmt(self.n_atoms)}, "
            f"n_edges={_fmt(self.n_edges)}, "
            f"n_graphs={_fmt(self.n_graphs)}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# MockSource
# ---------------------------------------------------------------------------


class MockSource:
    """Synthetic :class:`~molix.data.source.DataSource` returning random molecule samples.

    Each sample is a ``dict`` with at minimum ``Z`` (atomic numbers) and
    ``pos`` (Cartesian positions).  Suitable for profiling pipeline tasks
    that accept raw sample dicts.

    Args:
        n_samples: Number of samples in the source.
        n_atoms: Fixed atom count or ``(lo, hi)`` range per sample.
        atomic_numbers: Number of distinct element types (controls ``Z`` range).
        seed: Random seed for reproducibility.

    Example::

        source = MockSource(n_samples=500, n_atoms=(5, 20))
        sample = source[0]   # {"Z": tensor(N,), "pos": tensor(N, 3)}
        len(source)          # 500
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_atoms: _IntOrRange = (5, 20),
        *,
        atomic_numbers: int = 10,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.atomic_numbers = atomic_numbers
        # Pre-generate atom counts for each sample so source_id is stable
        rng = random.Random(seed)
        self._atom_counts: list[int] = [
            _resolve(n_atoms) if not isinstance(n_atoms, int) else n_atoms
            for _ in range(n_samples)
        ]
        # Per-sample generator seeds for reproducible, independent samples
        self._seeds: list[int] = [rng.randint(0, 2**31) for _ in range(n_samples)]

    @property
    def source_id(self) -> str:
        """Stable identifier for cache key computation."""
        return f"mock:{self.n_samples}:{self.n_atoms}:{self.atomic_numbers}"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Return a synthetic sample dict for index ``idx``.

        Args:
            idx: Sample index in ``[0, n_samples)``.

        Returns:
            Dict with keys ``"Z"`` ``(N,)`` and ``"pos"`` ``(N, 3)``.
        """
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for MockSource(n={self.n_samples})")
        n = self._atom_counts[idx]
        gen = torch.Generator().manual_seed(self._seeds[idx])
        Z = torch.randint(1, self.atomic_numbers + 1, (n,), generator=gen)
        pos = torch.randn(n, 3, generator=gen)
        return {"Z": Z, "pos": pos}

    def describe(self) -> str:
        """Return a human-readable description of this source.

        Returns:
            Description string.
        """
        return f"MockSource(n_samples={self.n_samples}, n_atoms={self.n_atoms})"
