"""End-to-end DataLoader smoke test with multiprocessing workers.

On Python 3.14+ the default POSIX start method is ``forkserver``, which
requires every DataLoader worker argument (dataset, collate_fn, ...) to
be picklable. This file is the regression test for two related bugs that
broke real training runs:

* DataModule._make_collate_fn returned a local closure (unpicklable)
* MmapDataset.__getstate__ dropped subclass-added attributes

Test flow mirrors the data path of train_allegro_qm9.py: pipeline -> mmap
dataset -> split -> DataModule with workers > 0 -> iterate one batch.
"""

from __future__ import annotations

import torch

from molix.data.collate import TargetSchema
from molix.data.datamodule import DataModule
from molix.data.dataset import MmapDataset
from molix.data.pipeline import Pipeline
from molix.data.source import InMemorySource
from molix.data.task import SampleTask
from molix.data.types import GraphBatch


class FakeNeighborList(SampleTask):
    """Linear-chain neighbor list, no compiled deps."""

    @property
    def task_id(self) -> str:
        return "fake_nlist"

    def execute(self, data: dict) -> dict:
        n = int(data["Z"].shape[0])
        if n < 2:
            edge_index = torch.zeros(0, 2, dtype=torch.long)
            diff = torch.zeros(0, 3)
            dist = torch.zeros(0)
        else:
            src = torch.arange(n - 1)
            dst = src + 1
            edge_index = torch.stack([src, dst], dim=1).long()
            diff = data["pos"][dst] - data["pos"][src]
            dist = diff.norm(dim=-1)
        return {**data, "edge_index": edge_index, "bond_diff": diff.float(),
                "bond_dist": dist.float()}


def _raw_samples(n: int = 16) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6, 1, 1], dtype=torch.long),
            "pos": torch.randn(4, 3),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


def test_mmap_dataset_with_num_workers_4(tmp_path):
    """Reproduces the exact failure mode of train_allegro_qm9.py:
    forkserver pickling + multi-worker DataLoader on a real mmap-backed
    dataset that has been split into train/val subsets.
    """
    raw = _raw_samples(16)
    src = InMemorySource(raw)
    spec = Pipeline("e2e").add(FakeNeighborList()).build()
    sink = tmp_path / "prepared"
    spec.materialize(src, sink=sink)
    full = MmapDataset.from_cache(sink)
    train, val = full.split(ratio=0.75, seed=0)
    assert len(train) == 12 and len(val) == 4

    schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
    dm = DataModule(
        train, val,
        target_schema=schema,
        batch_size=4,
        num_workers=4,           # triggers forkserver path
        pin_memory=False,        # avoid CUDA on CI
        prefetch_factor=2,
    )

    seen_batches = 0
    for batch in dm.train_dataloader():
        assert isinstance(batch, GraphBatch)
        assert batch["atoms", "Z"].shape[0] == 4 * 4   # 4 mols * 4 atoms each
        assert batch["graphs", "U0"].shape == (4,)
        assert batch["edges", "edge_index"].shape[1] == 2
        seen_batches += 1
    assert seen_batches == 3   # 12 / 4


def test_collate_picklable_with_batch_tasks(tmp_path):
    """Collate fn includes batch_tasks closure path; must still pickle."""
    import pickle

    raw = _raw_samples(8)
    src = InMemorySource(raw)
    spec = Pipeline("p").add(FakeNeighborList()).build()
    sink = tmp_path / "prepared"
    spec.materialize(src, sink=sink)
    ds = MmapDataset.from_cache(sink)
    train, val = ds.split(ratio=0.5)

    schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
    dm = DataModule(train, val, target_schema=schema, batch_size=2,
                    num_workers=0, pin_memory=False)

    fn = dm._make_collate_fn()
    pickle.loads(pickle.dumps(fn))   # must not raise
