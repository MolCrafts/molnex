"""Per-sample max atom/edge counts for batch-padding capacity."""

from __future__ import annotations

from typing import Any

import torch

from molix.data.task import DatasetTask


class CapacityStats(DatasetTask):
    """Record the largest single-sample atom and edge counts seen.

    With these two numbers, any batch of size ``B`` is bounded by
    ``B * max_atoms`` atoms and ``B * max_edges`` edges — enough to size
    fixed-shape batch padding for ``torch.compile``.

    Place after tasks that change atom/edge counts (e.g. ``NeighborList``).

    Usage::

        pipe = Pipeline(...).add(NeighborList(...)).add(CapacityStats()).build()
        cache(pipe, source, sink=sink, fit_source=train_subset)
        ds = MmapDataset(sink)
        pad_a, pad_e = CapacityStats.pad_sizes_from_state(
            ds.get_task_state("capacity_stats"), batch_size=256
        )
    """

    def __init__(self) -> None:
        self.max_atoms: int = 0
        self.max_edges: int = 0

    @property
    def task_id(self) -> str:
        return "capacity_stats"

    # -- DatasetTask contract -----------------------------------------------

    def fit(self, samples: list[dict]) -> None:
        max_a = 0
        max_e = 0
        for sample in samples:
            max_a = max(max_a, int(sample["Z"].shape[0]))
            ei = sample.get("edge_index")
            if ei is not None:
                max_e = max(max_e, int(ei.shape[0]))
        self.max_atoms = max_a
        self.max_edges = max_e

    def execute(self, data: dict) -> dict:
        return data

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_atoms": torch.tensor(self.max_atoms, dtype=torch.long),
            "max_edges": torch.tensor(self.max_edges, dtype=torch.long),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.max_atoms = int(state["max_atoms"].item())
        self.max_edges = int(state["max_edges"].item())

    # -- Helper -------------------------------------------------------------

    @staticmethod
    def pad_sizes_from_state(
        state: dict[str, Any], batch_size: int
    ) -> tuple[int, int]:
        """``(batch_size * max_atoms, batch_size * max_edges)``."""
        max_a = int(state["max_atoms"].item())
        max_e = int(state["max_edges"].item())
        return batch_size * max_a, batch_size * max_e
