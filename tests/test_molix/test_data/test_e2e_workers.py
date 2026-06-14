"""End-to-end DataLoader smoke test with multiprocessing workers.

On Python 3.14+ the default POSIX start method is ``forkserver``, which
requires every DataLoader worker argument (dataset, collate_fn, ...) to be
picklable. This file is the regression test for two related bugs that broke
real training runs:

* DataModule._make_collate_fn returned a local closure (unpicklable)
* MmapDataset must round-trip through pickle intact

Test flow mirrors the data path of train_allegro_qm9.py:
pipeline → PipelineSpec.cache() → MmapDataset → Subset →
DataModule(num_workers>0) → iterate.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.data.collate import TargetSchema
from molix.data.datamodule import DataModule
from molix.data.pipeline import Pipeline
from molix.data.source import InMemorySource
from molix.data.task import SampleTask


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
        return {
            **data,
            "edge_index": edge_index,
            "bond_diff": diff.float(),
            "bond_dist": dist.float(),
        }


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
    """Reproduces the training data path: cache() → dataset → split →
    DataModule with forkserver + multiple workers → iterate one epoch."""
    src = InMemorySource(_raw_samples(16))
    pipe = Pipeline("e2e").add(FakeNeighborList()).build()
    dag = pipe.cache(src, base_dir=tmp_path)

    full = dag.dataset(mmap=True)
    train, val = full.split(ratio=0.75, seed=0)
    assert len(train) == 12 and len(val) == 4

    schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
    dm = DataModule(
        train,
        val,
        target_schema=schema,
        batch_size=4,
        num_workers=4,  # triggers forkserver path
        pin_memory=False,  # avoid CUDA on CI
        prefetch_factor=2,
    )

    seen_batches = 0
    for batch in dm.train_dataloader():
        assert isinstance(batch, TensorDict)
        assert batch["atoms", "Z"].shape[0] == 4 * 4  # 4 mols × 4 atoms each
        assert batch["graphs", "U0"].shape == (4,)
        assert batch["edges", "edge_index"].shape[1] == 2
        seen_batches += 1
    assert seen_batches == 3  # 12 / 4


def test_collate_picklable_with_batch_nodes(tmp_path):
    """Collate fn wraps batch_nodes; must still pickle."""
    import pickle

    src = InMemorySource(_raw_samples(8))
    pipe = Pipeline("p").add(FakeNeighborList()).build()
    dag = pipe.cache(src, base_dir=tmp_path)

    ds = dag.dataset(mmap=True)
    train, val = ds.split(ratio=0.5)

    schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
    dm = DataModule(train, val, target_schema=schema, batch_size=2, num_workers=0, pin_memory=False)

    fn = dm._make_collate_fn()
    pickle.loads(pickle.dumps(fn))  # must not raise
