"""DataLoader worker arguments must pickle.

On Python 3.14+ the default POSIX start method is ``forkserver``, which
pickles every DataLoader worker argument (dataset, collate_fn, ...).
``DataModule._make_collate_fn`` once returned a local closure, which broke
real training runs; the dataset side is pinned in ``test_dataset.py``.
"""

from __future__ import annotations

import torch

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
            "edge_diff": diff.float(),
            "edge_dist": dist.float(),
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
