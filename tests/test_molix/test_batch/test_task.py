"""Unit tests for ``molix.batch.task.BatchTask`` and its DataModule dispatch.

Two concerns, one class each:

* ``TestBatchTask`` — the three members of the base itself: the default
  ``task_id``, ``execute`` raising on the base, ``__call__`` delegating to
  ``execute``. No DataLoader involved.
* ``TestDataModuleBatchTasks`` — ``DataModule(batch_tasks=[...])`` runs the
  tasks, in declaration order, on **both** collate paths (per-sample
  ``_CollateFn`` and block ``_BlockCollateFn``), and its ``__init__``
  exposes ``batch_tasks`` and no longer ``batch_nodes`` (the ETL ``Node``
  coupling this phase severs).

Order is proved by a decimal stamp: task with digit *d* rewrites the
per-graph ``stamp`` leaf as ``stamp * 10 + d``, so ``[1, 2]`` yields ``12``
and ``[2, 1]`` yields ``21`` — a wrong order cannot pass by accident.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from molix.batch import PadMolecularBatch
from molix.batch.collate import TargetSchema
from molix.batch.datamodule import DataModule, _BlockCollateFn, _CollateFn
from molix.batch.task import BatchTask
from molix.store import BlockDataset
from tests.test_molix.conftest import unit_a, unit_b, write_block

#: Graph-level ``U0`` only — the fixture carries no atom-level target.
SCHEMA = TargetSchema(graph_level=frozenset({"energy"}), atom_level=frozenset({"forces"}))


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _samples(n: int = 4) -> list[dict]:
    """Build *n* two-atom samples with one edge each, hand-built (no RNG).

    Args:
        n: Number of samples. ``targets.U0`` carries the identity
            ``float(i)``.

    Returns:
        Flat sample dicts with per-atom ``Z`` ``(2,)`` / ``pos`` ``(2, 3)``,
        per-edge ``edge_index`` ``(1, 2)`` / ``edge_diff`` ``(1, 3)`` /
        ``edge_dist`` ``(1,)`` and graph-level ``targets.U0`` ``(1,)``.
    """
    out: list[dict] = []
    for i in range(n):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0 + 0.1 * i, 0.0, 0.0]])
        diff = (pos[1] - pos[0]).unsqueeze(0)
        out.append(
            {
                "Z": torch.tensor([1, 6], dtype=torch.long),
                "pos": pos,
                "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
                "edge_diff": diff,
                "edge_dist": diff.norm(dim=-1),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return out


class _SampleDictDataset(Dataset):
    """Map-style dataset with no packed backing — forces the per-sample path."""

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Task stubs
# ---------------------------------------------------------------------------


class _NoopTask(BatchTask):
    """Subclass that overrides nothing but ``execute`` — pins the default id."""

    def execute(self, data: dict) -> dict:
        """Return *data* unchanged."""
        return data


class _RecordingTask(BatchTask):
    """Records every ``execute`` call so ``__call__`` delegation is checkable."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def execute(self, data: dict) -> dict:
        """Record *data* and return a marked copy."""
        self.seen.append(data)
        return {**data, "executed": True}


class _StampTask(BatchTask):
    """Append ``digit`` to a per-graph ``stamp`` leaf (base-10 call order)."""

    def __init__(self, digit: int) -> None:
        self.digit = int(digit)

    @property
    def task_id(self) -> str:
        return f"stamp:{self.digit}"

    def execute(self, data: TensorDict) -> TensorDict:
        """Rewrite ``("graphs", "stamp")`` as ``stamp * 10 + digit``."""
        graphs = data["graphs"]
        n_graphs = int(graphs.batch_size[0])
        if "stamp" in graphs.keys():
            prev = graphs["stamp"] * 10
        else:
            prev = torch.zeros(n_graphs, dtype=torch.long)
        data["graphs", "stamp"] = prev + self.digit
        return data


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------


class TestBatchTask:
    """The three members of the ``BatchTask`` base."""

    def test_task_id_defaults_to_class_name(self):
        """The default ``task_id`` is the concrete class name."""
        assert _NoopTask().task_id == "_NoopTask"

    def test_task_id_on_base_is_base_class_name(self):
        """The property is defined on the base, not injected by subclasses."""
        assert BatchTask().task_id == "BatchTask"

    def test_execute_raises_on_base(self):
        """The base has no transform — it must refuse, not silently pass data through."""
        with pytest.raises(NotImplementedError):
            BatchTask().execute({})

    def test_call_delegates_to_execute(self):
        """``task(batch)`` returns exactly what ``execute(batch)`` returns."""
        task = _RecordingTask()
        data = {"atoms": 1}
        out = task(data)
        assert out == {"atoms": 1, "executed": True}
        assert task.seen == [data]

    def test_call_on_base_raises(self):
        """Delegation is real: the base's ``__call__`` surfaces ``execute``'s error."""
        with pytest.raises(NotImplementedError):
            BatchTask()({})


# ---------------------------------------------------------------------------
# DataModule dispatch
# ---------------------------------------------------------------------------


def _packed_dm(tmp_path: Path, tag: str, **kwargs) -> DataModule:
    """DataModule over sealed blocks — routes through ``_BlockCollateFn``."""
    train = BlockDataset.load_from_disk(
        write_block(tmp_path / f"{tag}_train" / "b", [unit_a()]).parent
    )
    val = BlockDataset.load_from_disk(
        write_block(tmp_path / f"{tag}_val" / "b", [unit_b()], split="val").parent
    )
    kwargs.setdefault("target_schema", SCHEMA)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    return DataModule(train, val, **kwargs)


def _unpacked_dm(**kwargs) -> DataModule:
    """DataModule over a plain dict dataset — routes through ``_CollateFn``."""
    ds = _SampleDictDataset(_samples(4))
    kwargs.setdefault("target_schema", SCHEMA)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    return DataModule(ds, ds, **kwargs)


class TestDataModuleBatchTasks:
    """``batch_tasks`` replaces ``batch_nodes`` and dispatches via ``execute``."""

    def test_init_accepts_batch_tasks(self):
        """``batch_tasks`` is a keyword argument of ``DataModule.__init__``."""
        assert "batch_tasks" in inspect.signature(DataModule.__init__).parameters

    def test_init_has_no_batch_nodes(self):
        """The ETL ``Node`` parameter is gone — no shim, no alias."""
        assert "batch_nodes" not in inspect.signature(DataModule.__init__).parameters

    def test_default_batch_tasks_is_empty(self):
        """Omitting the kwarg leaves an empty task sequence, not ``None``."""
        assert tuple(_unpacked_dm().batch_tasks) == ()

    def test_no_tasks_leaves_batch_untouched(self):
        """With no tasks the collated batch carries no task-written leaf."""
        batch = next(iter(_unpacked_dm().val_dataloader()))
        assert "stamp" not in batch["graphs"].keys()

    def test_tasks_run_on_per_sample_path(self):
        """The per-sample ``_CollateFn`` dispatches through ``BatchTask.execute``."""
        dm = _unpacked_dm(batch_tasks=[_StampTask(1)])
        loader = dm.val_dataloader()
        assert isinstance(loader.collate_fn, _CollateFn)
        batch = next(iter(loader))
        assert torch.equal(batch["graphs", "stamp"], torch.tensor([1, 1]))

    def test_tasks_run_in_order_on_per_sample_path(self):
        """Declaration order is execution order on the per-sample path."""
        dm = _unpacked_dm(batch_tasks=[_StampTask(1), _StampTask(2)])
        batch = next(iter(dm.val_dataloader()))
        assert torch.equal(batch["graphs", "stamp"], torch.tensor([12, 12]))

    def test_reversed_tasks_reverse_the_stamp(self):
        """Reversing the sequence reverses the stamp — order is not incidental."""
        dm = _unpacked_dm(batch_tasks=[_StampTask(2), _StampTask(1)])
        batch = next(iter(dm.val_dataloader()))
        assert torch.equal(batch["graphs", "stamp"], torch.tensor([21, 21]))

    def test_tasks_run_on_packed_path(self, tmp_path):
        """The block ``_BlockCollateFn`` applies the same tasks."""
        dm = _packed_dm(tmp_path, "stamp", batch_tasks=[_StampTask(1)])
        loader = dm.val_dataloader()
        assert isinstance(loader.collate_fn, _BlockCollateFn)
        batch = next(iter(loader))
        assert torch.equal(batch["graphs", "stamp"], torch.tensor([1, 1]))

    def test_tasks_run_in_order_on_packed_path(self, tmp_path):
        """Declaration order is execution order on the packed fast path too."""
        dm = _packed_dm(tmp_path, "order", batch_tasks=[_StampTask(1), _StampTask(2)])
        batch = next(iter(dm.val_dataloader()))
        assert torch.equal(batch["graphs", "stamp"], torch.tensor([12, 12]))

    def test_pad_task_applies_on_per_sample_path(self):
        """The shipped ``PadMolecularBatch`` reaches the batch as a ``batch_tasks`` entry."""
        dm = _unpacked_dm(batch_tasks=[PadMolecularBatch(max_atoms=8, max_edges=8)])
        batch = next(iter(dm.val_dataloader()))
        assert batch["atoms"].batch_size[0] == 8
        assert batch["edges"].batch_size[0] == 8
        assert int(batch["atoms", "mask"].sum()) == 4  # 2 mols x 2 atoms

    def test_pad_task_applies_on_packed_path(self, tmp_path):
        """Same task, block fast path — identical fixed shapes."""
        dm = _packed_dm(tmp_path, "pad", batch_tasks=[PadMolecularBatch(max_atoms=16, max_edges=8)])
        batch = next(iter(dm.val_dataloader()))
        assert batch["atoms"].batch_size[0] == 16
        assert batch["edges"].batch_size[0] == 8
        assert int(batch["atoms", "mask"].sum()) == 8  # 2 frames x 4 atoms
