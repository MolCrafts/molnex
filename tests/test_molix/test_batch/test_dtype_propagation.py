"""Precision contract — ``config.set_precision`` must reach the dataloader.

Regression net for the 2026-05-18 fp64 production-failure incident.
``config.set_precision("fp64")`` previously only updated model construction;
:class:`molix.batch.datamodule._CollateFn` ignored ``config["ftype"]`` and
emitted fp32 batches, which crashed the first training step under fp64.

These tests pin the contract that:

- ``_CollateFn`` captures ``config["ftype"]`` at construction time;
- every floating-point tensor in the emitted batch is cast to that dtype
  via :func:`molix.core.steps.batch_to`;
- long index tensors (``edge_index``, ``batch``, ``Z``, ``num_atoms``) are
  left untouched so they survive the cast.

If any of these break again, this file goes red before any model training
runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molix.batch.datamodule import DataModule
from molix.config import config
from molix.core.steps import batch_to
from molix.store import FrameStoreDataset
from tests.conftest import frame
from tests.test_molix.conftest import unit_a, unit_b, write_block

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_precision():
    config.set_precision("fp32")
    yield
    config.set_precision("fp32")


def _make_samples(n: int = 6) -> list:
    """fp32 source frames — what a dataset hands the collate."""
    return [
        frame(Z=[1, 6], pos=torch.randn(2, 3, dtype=torch.float32), energy=float(i))
        for i in range(n)
    ]


def _build_dm(tmp_path: Path, *, batch_size: int = 2) -> DataModule:
    return DataModule(
        FrameStoreDataset.load_from_disk(write_block(tmp_path / "train" / "b", [unit_a()]).parent),
        FrameStoreDataset.load_from_disk(
            write_block(tmp_path / "val" / "b", [unit_b()], split="val").parent
        ),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# batch_to helper — leaves long tensors alone on the dtype axis,
# moves everything (incl. long) on the device axis
# ---------------------------------------------------------------------------


class TestBatchTo:
    def test_noop_when_neither_set(self):
        x = torch.randn(4, dtype=torch.float32)
        assert batch_to(x) is x

    def test_casts_floating_tensor(self):
        x = torch.randn(4, dtype=torch.float32)
        out = batch_to(x, dtype=torch.float64)
        assert out.dtype is torch.float64

    def test_preserves_long_tensor_under_dtype(self):
        idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = batch_to(idx, dtype=torch.float64)
        assert out.dtype is torch.long, "long index tensors must survive dtype cast"

    def test_dict_recurses_selectively(self):
        batch = {
            "pos": torch.randn(3, dtype=torch.float32),
            "edge_index": torch.zeros(3, dtype=torch.long),
        }
        out = batch_to(batch, dtype=torch.float64)
        assert out["pos"].dtype is torch.float64
        assert out["edge_index"].dtype is torch.long

    def test_tensordict_apply_path(self, tmp_path):
        dm = _build_dm(tmp_path)
        batch = next(iter(dm.train_dataloader()))
        out = batch_to(batch, dtype=torch.float64)
        assert out["atoms", "pos"].dtype is torch.float64
        assert out["edges", "edge_diff"].dtype is torch.float64
        assert out["edges", "edge_index"].dtype is torch.long
        assert out["atoms", "Z"].dtype is torch.long

    def test_device_and_dtype_in_one_pass(self, tmp_path):
        dm = _build_dm(tmp_path)
        batch = next(iter(dm.train_dataloader()))
        out = batch_to(batch, device=torch.device("cpu"), dtype=torch.float64)
        assert out["atoms", "pos"].device.type == "cpu"
        assert out["atoms", "pos"].dtype is torch.float64
        assert out["edges", "edge_index"].device.type == "cpu"
        assert out["edges", "edge_index"].dtype is torch.long

    def test_same_device_move_returns_batch_unchanged(self, tmp_path):
        """Device-only move to where the batch already lives is a no-op fast path.

        The batch's leaves are all on CPU; moving to CPU must skip the
        ``TensorDict.apply`` rebuild and return the same object — this is the
        per-step Trainer-loop optimization, harmless because the values are
        already correct.
        """
        dm = _build_dm(tmp_path)
        batch = next(iter(dm.train_dataloader()))
        out = batch_to(batch, device=torch.device("cpu"))
        assert out is batch  # no rebuild
        assert out["atoms", "pos"].device.type == "cpu"

    def test_same_device_move_still_casts_with_dtype(self, tmp_path):
        """The fast path only triggers for device-only moves, never when dtype is set."""
        dm = _build_dm(tmp_path)
        batch = next(iter(dm.train_dataloader()))
        out = batch_to(batch, device=torch.device("cpu"), dtype=torch.float64)
        assert out is not batch
        assert out["atoms", "pos"].dtype is torch.float64


# ---------------------------------------------------------------------------
# End-to-end: set_precision flows through to the emitted batch
# ---------------------------------------------------------------------------


_FTYPE_PER_MODE = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16-mixed": torch.float32,  # AMP keeps batches in fp32, autocast inside fwd
    "bf16-mixed": torch.float32,
}


class TestPrecisionFlowsToBatch:
    @pytest.mark.parametrize("mode,expected", list(_FTYPE_PER_MODE.items()))
    def test_collate_emits_correct_dtype(self, tmp_path: Path, mode: str, expected: torch.dtype):
        config.set_precision(mode)
        dm = _build_dm(tmp_path)  # _CollateFn captures ftype here
        batch = next(iter(dm.train_dataloader()))
        assert batch["atoms", "pos"].dtype is expected
        assert batch["edges", "edge_diff"].dtype is expected
        assert batch["edges", "edge_dist"].dtype is expected
        assert batch["graphs", "energy"].dtype is expected
        # Integers untouched
        assert batch["atoms", "Z"].dtype is torch.long
        assert batch["edges", "edge_index"].dtype is torch.long
        assert batch["atoms", "batch"].dtype is torch.long
        assert batch["graphs", "num_atoms"].dtype is torch.long

    def test_collate_captures_ftype_at_construction(self, tmp_path: Path):
        """Spawned workers re-import molix.config and reset ftype to fp32;
        the captured value must survive that re-import. We can't directly
        test multi-process spawn here without overhead, but we can verify
        the captured attribute is the source of truth (not a live read)."""
        config.set_precision("fp64")
        dm = _build_dm(tmp_path)
        collate = dm._make_collate_fn()
        assert collate.ftype is torch.float64
        # Flip global; collate's captured value should NOT follow.
        config.set_precision("fp32")
        assert collate.ftype is torch.float64
        # And the captured value is what gets applied:
        samples = _make_samples(2)
        out = collate(samples)
        assert out["atoms", "pos"].dtype is torch.float64

    def test_dataset_loader_canonical_fp32_preserved(self, tmp_path: Path):
        """A block stays fp32 regardless of set_precision — the up-cast
        happens at collate, not at load. This keeps one block reusable
        across precisions."""
        config.set_precision("fp64")
        ds = FrameStoreDataset.load_from_disk(write_block(tmp_path / "b" / "x", [unit_a()]).parent)
        raw = ds[0]
        assert raw["atoms", "pos"].dtype is torch.float32
        assert config["ftype"] is torch.float64
