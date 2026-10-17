"""Precision contract — the ``DataModule``'s ``dtype`` must reach every emitted batch.

Regression net for the 2026-05-18 fp64 production-failure incident: the
model was built at fp64 while the dataloader kept emitting fp32 batches, and
the first training step died on a dtype-mismatched matmul. There is no global
precision any more; the contract pinned here is that

- :class:`molix.data.datamodule._CollateFn` carries the ``dtype`` it was
  built with (``DataModule(dtype=…)``), pickled into every worker with it;
- compute floating leaves (``atoms.pos``, ``pair_diff``, …) follow that
  ``dtype`` via :func:`molix.core.steps.cast_compute_floats`;
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


def _make_samples(n: int = 6) -> list:
    """fp32 source frames — what a dataset hands the collate."""
    return [
        frame(Z=[1, 6], pos=torch.randn(2, 3, dtype=torch.float32), energy=float(i))
        for i in range(n)
    ]


def _build_dm(
    tmp_path: Path, *, batch_size: int = 2, dtype: torch.dtype = torch.float32
) -> DataModule:
    return DataModule(
        frames_dataset([unit_a()]),
        frames_dataset([unit_b()]),
        dtype=dtype,
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
# End-to-end: DataModule(dtype=…) flows through to the emitted batch
# ---------------------------------------------------------------------------


class TestDtypeFlowsToBatch:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_collate_emits_the_datamodule_dtype(self, tmp_path: Path, dtype: torch.dtype):
        dm = _build_dm(tmp_path, dtype=dtype)
        batch = next(iter(dm.train_dataloader()))
        assert batch["atoms", "pos"].dtype is dtype
        assert batch["pairs", "pair_diff"].dtype is dtype
        assert batch["pairs", "pair_dist"].dtype is dtype
        # A graph total is a label, not a compute leaf: the Frame route emits it
        # at binary64 and cast_compute_floats must leave it there whatever ``dtype`` is.
        assert batch["graphs", "energy"].dtype is torch.float64
        # Integers untouched
        assert batch["atoms", "Z"].dtype is torch.long
        assert batch["pairs", "pair_index"].dtype is torch.long
        assert batch["atoms", "batch"].dtype is torch.long
        assert batch["graphs", "num_atoms"].dtype is torch.long

    def test_the_collate_callable_carries_the_dtype(self, tmp_path: Path):
        """``dtype`` is an attribute of the pickled callable — what a spawned worker applies."""
        collate = _build_dm(tmp_path, dtype=torch.float64)._make_collate_fn()
        assert collate.dtype is torch.float64
        out = collate(_make_samples(2))
        assert out["atoms", "pos"].dtype is torch.float64

    def test_a_collator_at_another_dtype_is_refused(self, tmp_path: Path):
        """One dtype, passed once: a collate_fn naming a different one is a contradiction."""
        with pytest.raises(ValueError, match="one dtype"):
            DataModule(
                frames_dataset([unit_a()]),
                frames_dataset([unit_b()]),
                collate_fn=FrameCollator(collator=MoleculeCollator(dtype=torch.float64)),
                dtype=torch.float32,
                num_workers=0,
                pin_memory=False,
            )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_frame_route_collate_keeps_graph_energy_float64(self, dtype: torch.dtype):
        e_raw_ev = -74800.123456789
        out = _CollateFn(FrameCollator(collator=MoleculeCollator()), (), dtype=dtype)(
            [frame(Z=[36], pos=[[0.0, 0.0, 0.0]], energy=e_raw_ev)]
        )
        assert out["atoms", "pos"].dtype is dtype
        assert out["graphs", "energy"].dtype is torch.float64
        assert out["graphs", "energy"].item() == e_raw_ev

    def test_autocast_does_not_change_frame_route_energy_dtype(self):
        """Protected Frame-route leaves never enter collate/move as float32."""
        batch = _CollateFn(FrameCollator(collator=MoleculeCollator()), (), dtype=torch.float32)(
            [frame(Z=[36], pos=[[0.0, 0.0, 0.0]], energy=-74800.123456789)]
        )
        assert batch["graphs", "energy"].dtype is torch.float64
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            scaled = batch["graphs", "energy"] * 1.0
            assert scaled.dtype is torch.float64
