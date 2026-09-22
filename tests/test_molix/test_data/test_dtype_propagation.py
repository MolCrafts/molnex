"""Precision contract — ``config.set_precision`` must reach the dataloader.

Regression net for the 2026-05-18 fp64 production-failure incident.
``config.set_precision("fp64")`` previously only updated model construction;
:class:`molix.data.datamodule._CollateFn` ignored ``config["ftype"]`` and
emitted fp32 batches, which crashed the first training step under fp64.

These tests pin the contract that:

- ``_CollateFn`` captures ``config["ftype"]`` at
  construction time;
- compute floating leaves (``atoms.pos``, ``pair_diff``, …) follow that
  ``ftype`` via :func:`molix.core.steps.cast_compute_floats`;
- graph totals at ``ENERGY_KEY`` / ``E0_KEY`` skip the dtype axis on a
  TensorDict (full NestedKey match, never the leaf name ``"energy"``);
- Frame-route collate *emits* those totals as float64; store-route collate
  does not narrow them below the mmap column and does not upcast a float32
  store leaf;
- long index tensors (``pair_index``, ``batch``, ``Z``, ``num_atoms``) are
  left untouched so they survive the cast.

If any of these break again, this file goes red before any model training
runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix.config import config
from molix.core.steps import cast_compute_floats
from molix.data import FrameCollator
from molix.data.collate import MoleculeCollator
from molix.data.datamodule import DataModule, _CollateFn
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY
from tests.conftest import frame
from tests.test_molix.conftest import frames_dataset, unit_a, unit_b

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
        frames_dataset([unit_a()]),
        frames_dataset([unit_b()]),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )


def _label_batch(
    *,
    pos_dtype: torch.dtype = torch.float64,
    energy_dtype: torch.dtype = torch.float64,
    e0_dtype: torch.dtype = torch.float64,
    atomic_energy: bool = False,
) -> TensorDict:
    """Nested post-collate batch with graph totals at rest precision."""
    atoms: dict[str, torch.Tensor] = {
        "pos": torch.zeros(2, 3, dtype=pos_dtype),
        "Z": torch.tensor([1, 8], dtype=torch.long),
    }
    if atomic_energy:
        atoms["energy"] = torch.zeros(2, dtype=torch.float64)
    graphs = TensorDict(
        {
            "energy": torch.tensor([-1.5], dtype=energy_dtype),
            "e0": torch.tensor([-1.0], dtype=e0_dtype),
            "U0": torch.tensor([0.25], dtype=torch.float64),
            "num_atoms": torch.tensor([2], dtype=torch.long),
        },
        batch_size=[1],
    )
    return TensorDict(
        {
            "atoms": TensorDict(atoms, batch_size=[2]),
            "graphs": graphs,
        },
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# cast_compute_floats — the dtype policy the library has no API for: every float
# leaf takes the compute dtype except the graph totals, matched by NestedKey.
# Device moves are `TensorDict.to` and are not wrapped.
# ---------------------------------------------------------------------------


class TestCastComputeFloats:
    def test_casts_compute_floats_and_leaves_indices_alone(self, tmp_path):
        dm = _build_dm(tmp_path)
        batch = next(iter(dm.train_dataloader()))
        out = cast_compute_floats(batch, torch.float64)
        assert out["atoms", "pos"].dtype is torch.float64
        assert out["pairs", "pair_diff"].dtype is torch.float64
        assert out["pairs", "pair_index"].dtype is torch.long
        assert out["atoms", "Z"].dtype is torch.long

    def test_skips_graph_totals_by_nested_key(self):
        """``ENERGY_KEY`` and ``E0_KEY`` keep float64; compute leaves take dtype."""
        out = cast_compute_floats(_label_batch(), torch.float32)
        assert out["atoms", "pos"].dtype is torch.float32
        assert out[ENERGY_KEY].dtype is torch.float64
        assert out["graphs", "e0"].dtype is torch.float64
        assert out["graphs", "U0"].dtype is torch.float32
        assert out["atoms", "Z"].dtype is torch.long

    def test_atomic_energy_follows_dtype_not_the_leaf_name_energy(self):
        """A leaf-name skip of ``"energy"`` would wrongly protect per-atom energy."""
        out = cast_compute_floats(_label_batch(atomic_energy=True), torch.float32)
        assert out[ATOMIC_ENERGY_KEY].dtype is torch.float32
        assert out[ENERGY_KEY].dtype is torch.float64
        assert out["graphs", "e0"].dtype is torch.float64

    def test_float32_energy_key_is_not_upcast(self):
        """A late ``.double()`` is not a repair; store float32 stays float32."""
        out = cast_compute_floats(
            _label_batch(pos_dtype=torch.float32, energy_dtype=torch.float32),
            torch.float64,
        )
        assert out["atoms", "pos"].dtype is torch.float64
        assert out[ENERGY_KEY].dtype is torch.float32

    def test_a_protected_total_still_moves_with_the_batch(self, tmp_path):
        """The skip is the dtype axis only — the device is `TensorDict.to`'s job."""
        moved = _label_batch().to(torch.device("cpu"))
        out = cast_compute_floats(moved, torch.float32)
        assert out[ENERGY_KEY].device.type == "cpu"
        assert out[ENERGY_KEY].dtype is torch.float64
        assert out["atoms", "pos"].dtype is torch.float32

    def test_a_non_tensordict_batch_is_refused(self):
        """A NestedKey skip cannot be expressed on something with no nested keys.

        The old helper recursed into plain dicts and bare tensors, where the skip
        silently did not apply — so a protected total was cast without a word. No
        caller in the tree used those branches.
        """
        with pytest.raises(TypeError, match="needs a TensorDict batch"):
            cast_compute_floats({"graphs": {"energy": torch.zeros(1)}}, torch.float32)
        with pytest.raises(TypeError, match="needs a TensorDict batch"):
            cast_compute_floats(torch.zeros(3), torch.float32)


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
        assert batch["pairs", "pair_diff"].dtype is expected
        assert batch["pairs", "pair_dist"].dtype is expected
        # A graph total is a label, not a compute leaf: the Frame route emits it
        # at binary64 and cast_compute_floats must leave it there whatever ``ftype`` is.
        assert batch["graphs", "energy"].dtype is torch.float64
        # Integers untouched
        assert batch["atoms", "Z"].dtype is torch.long
        assert batch["pairs", "pair_index"].dtype is torch.long
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

    @pytest.mark.parametrize("mode,ftype", list(_FTYPE_PER_MODE.items()))
    def test_frame_route_collate_keeps_graph_energy_float64(self, mode: str, ftype: torch.dtype):
        config.set_precision(mode)
        e_raw_ev = -74800.123456789
        out = _CollateFn(FrameCollator(collator=MoleculeCollator()), ())(
            [frame(Z=[36], pos=[[0.0, 0.0, 0.0]], energy=e_raw_ev)]
        )
        assert out["atoms", "pos"].dtype is ftype
        assert out["graphs", "energy"].dtype is torch.float64
        assert out["graphs", "energy"].item() == e_raw_ev

    def test_autocast_does_not_change_frame_route_energy_dtype(self):
        """Protected Frame-route leaves never enter collate/move as float32."""
        config.set_precision("fp32")
        batch = _CollateFn(FrameCollator(collator=MoleculeCollator()), ())(
            [frame(Z=[36], pos=[[0.0, 0.0, 0.0]], energy=-74800.123456789)]
        )
        assert batch["graphs", "energy"].dtype is torch.float64
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            scaled = batch["graphs", "energy"] * 1.0
            assert scaled.dtype is torch.float64
