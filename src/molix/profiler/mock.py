"""Synthetic data generators for isolated module profiling.

Provides two generators:

- :class:`MockBatch` — produces a nested ``TensorDict`` batch
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
    batch = factory()   # -> TensorDict

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
import torch.nn as nn
from tensordict import TensorDict

# Type alias: int means fixed; tuple[int, int] means sample from [lo, hi]
_IntOrRange = Union[int, tuple[int, int]]


def _resolve(value: _IntOrRange, rng: random.Random) -> int:
    """Sample a concrete integer from a fixed value or (lo, hi) range.

    Args:
        value: Either a fixed ``int`` or a ``(lo, hi)`` inclusive range.
        rng: Caller-owned generator to draw from. Passed explicitly rather
            than read off the ``random`` module so a seeded owner
            (:class:`MockBatch`, :class:`MockSource`) really is reproducible.

    Returns:
        A concrete integer.
    """
    if isinstance(value, int):
        return value
    lo, hi = value
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# MockBatch
# ---------------------------------------------------------------------------


class MockBatch:
    """Callable factory that generates synthetic ``TensorDict`` instances.

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
        batch = factory()   # produces a TensorDict

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

    def __call__(self) -> TensorDict:
        """Generate a fresh ``TensorDict``.

        Returns:
            A ``TensorDict`` with random tensor values and the configured shape.
        """
        n_a = _resolve(self.n_atoms, self._rng)
        n_e = _resolve(self.n_edges, self._rng)
        n_g = _resolve(self.n_graphs, self._rng)

        dev = self.device
        gen = self._torch_gen

        # --- Atom data ---
        Z = torch.randint(1, self.atomic_numbers + 1, (n_a,), device=dev, generator=gen)
        pos = torch.randn(n_a, 3, device=dev, generator=gen)
        # Distribute atoms across graphs (roughly equal split)
        batch_vec = torch.zeros(n_a, dtype=torch.long, device=dev)
        if n_g > 1 and n_a > 0:
            boundaries = sorted(self._rng.sample(range(1, n_a), min(n_g - 1, n_a - 1)))
            for graph_idx, start in enumerate(boundaries):
                batch_vec[start:] = graph_idx + 1

        atoms = TensorDict({"Z": Z, "pos": pos, "batch": batch_vec}, batch_size=[n_a])

        # --- Edge data ---
        if n_e > 0 and n_a > 1:
            # Random edges (source, target) within [0, n_a)
            src = torch.randint(0, n_a, (n_e,), device=dev, generator=gen)
            dst = torch.randint(0, n_a, (n_e,), device=dev, generator=gen)
            edge_index = torch.stack([src, dst], dim=1)  # (E, 2)
            edge_diff = torch.randn(n_e, 3, device=dev, generator=gen)
            edge_dist = torch.rand(n_e, device=dev, generator=gen) * 5.0
        else:
            edge_index = torch.zeros(0, 2, dtype=torch.long, device=dev)
            edge_diff = torch.zeros(0, 3, device=dev)
            edge_dist = torch.zeros(0, device=dev)
            n_e = 0

        edges = TensorDict(
            {"edge_index": edge_index, "edge_diff": edge_diff, "edge_dist": edge_dist},
            batch_size=[n_e],
        )

        # --- Graph data ---
        num_atoms_per_graph = torch.bincount(batch_vec, minlength=n_g).long()
        graphs = TensorDict({"num_atoms": num_atoms_per_graph}, batch_size=[n_g])

        return TensorDict({"atoms": atoms, "edges": edges, "graphs": graphs}, batch_size=[])

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
        self._atom_counts: list[int] = [_resolve(n_atoms, rng) for _ in range(n_samples)]
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


class MockModel(nn.Module):
    """MolNex encoder-protocol model with negligible compute.

    Mirrors the molzoo encoder contract — ``forward(td: TensorDict) ->
    TensorDict`` reads ``atoms.pos`` and writes per-layer node features
    ``(N, 1, n_features)`` under ``atoms.node_features`` — but does only a
    single scalar multiply, so a Trainer loop wrapped around it spends
    essentially all its time in framework machinery (Step dispatch, hooks,
    ``batch_to``, TrainState writes), not model FLOPs. That is exactly what
    :class:`~molix.profiler.trainer.TrainerProfiler` needs to isolate the
    Trainer's own per-step overhead.

    One scalar :class:`~torch.nn.Parameter` keeps the autograd graph real so
    ``backward`` and the optimizer step exercise their normal paths.

    Args:
        n_features: Width of the emitted ``node_features`` feature axis.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.n_features = n_features

    def forward(self, batch: TensorDict) -> TensorDict:
        """Write ``atoms.node_features`` ``(N, 1, n_features)`` and return *batch*.

        Args:
            batch: Post-collate nested batch with ``atoms.pos`` ``(N, 3)``.

        Returns:
            The same batch, with ``node_features`` written under ``atoms``.
        """
        pos = batch["atoms", "pos"]
        val = (pos.sum(dim=-1, keepdim=True) * self.w.sum()).unsqueeze(1)
        batch["atoms", "node_features"] = val.expand(-1, 1, self.n_features)
        return batch


def mock_node_feature_loss(predictions: TensorDict, batch: TensorDict) -> torch.Tensor:
    """Sum of ``atoms.node_features`` — the default loss for :class:`MockModel`.

    Args:
        predictions: Batch returned by :class:`MockModel` (carries
            ``atoms.node_features``).
        batch: The input batch (unused; present for the loss-fn signature).

    Returns:
        A scalar loss whose backward touches the model's parameter.
    """
    return predictions["atoms", "node_features"].sum()
