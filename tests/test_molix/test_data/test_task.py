"""Unit tests for ``molix.data.task.BatchTask`` and its DataModule dispatch.

Two concerns, one class each:

* ``TestBatchTask`` — the three members of the base itself: the default
  ``task_id``, ``execute`` raising on the base, ``__call__`` delegating to
  ``execute``. No DataLoader involved.
* ``TestDataModuleBatchTasks`` — ``DataModule(batch_tasks=[...])`` runs the
  tasks, in declaration order, on **both** collate paths (per-sample
  ``_CollateFn``), and its ``__init__``
  exposes ``batch_tasks`` and no longer ``batch_nodes`` (the ETL ``Node``
  coupling this phase severs).

Order is proved by a decimal stamp: task with digit *d* rewrites the
per-graph ``stamp`` leaf as ``stamp * 10 + d``, so ``[1, 2]`` yields ``12``
and ``[2, 1]`` yields ``21`` — a wrong order cannot pass by accident.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from molix.data import FrameCollator, MoleculeCollator, MoleculePadder
from molix.data.datamodule import DataModule, _CollateFn
from molix.data.task import BatchTask
from tests.conftest import frame

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _samples(n: int = 4) -> list:
    """Build *n* two-atom frames, hand-built (no RNG).

    Args:
        n: Number of frames; the ``energy`` meta scalar carries ``float(i)``.

    Returns:
        molpy ``Frame`` objects.
    """
    return [
        frame(Z=[1, 6], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], energy=float(i)) for i in range(n)
    ]


class _SampleDictDataset(Dataset):
    """Map-style dataset of frames — forces the per-sample collate path."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
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


def _unpacked_dm(**kwargs) -> DataModule:
    """DataModule over a plain dict dataset — routes through ``_CollateFn``."""
    ds = _SampleDictDataset(_samples(4))
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

    def test_padder_applies_on_per_sample_path(self):
        """``MoleculePadder`` is the collate_fn padder, not a batch_task."""
        dm = _unpacked_dm(
            collate_fn=FrameCollator(
                collator=MoleculeCollator(),
                padder=MoleculePadder(max_atoms=8, max_pairs=8, cutoff=1.0),
            )
        )
        batch = next(iter(dm.val_dataloader()))
        assert batch["atoms"].batch_size[0] == 8
        assert batch["pairs"].batch_size[0] == 8
        assert int(batch["atoms", "mask"].sum()) == 4  # 2 mols x 2 atoms
