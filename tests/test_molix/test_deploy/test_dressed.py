"""Tests for :class:`molix.deploy.DressedPotential` — the dress added back for export."""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.compile import Compiler
from molix.data import AtomicDress
from molix.deploy import AtomEnergyAdapter, DressedPotential, example_inputs
from molix.deploy.contract import DYNAMIC_SHAPES
from molix.export import aoti
from molix.F import batch_add
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY

FORCES = ("atoms.pos",)
# eV; a multi-thousand-eV baseline, so a float32 add would swamp the residual.
_C1 = -13.6
_C6 = -1030.08
_C8 = -2043.8


class _PairResidual(Potential):
    """The residual: ``E_e = |d_e|²`` split evenly over both endpoints."""

    def __init__(self, *, wrt: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._derivative = Derivative(self.energy, wrt=wrt)

    def energy(self, frame: TensorDict) -> TensorDict:
        pair_index, diff = frame["pairs", "pair_index"], frame["pairs", "pair_diff"]
        half = 0.5 * (diff * diff).sum(-1)
        n = frame["atoms"].batch_size[0]
        atomic = batch_add(half, pair_index[:, 0], n) + batch_add(half, pair_index[:, 1], n)
        frame[ATOMIC_ENERGY_KEY] = atomic
        frame[ENERGY_KEY] = batch_add(
            atomic, frame["atoms", "batch"], dim_size=frame["graphs"].batch_size[0]
        )
        return frame


def _dress() -> AtomicDress:
    e0 = torch.zeros(9, dtype=torch.float64)
    e0[1], e0[6], e0[8] = _C1, _C6, _C8
    support = e0 != 0
    return AtomicDress(e0=e0, support=support)


def _frame(
    *, dtype: torch.dtype = torch.float64, n_graphs: int = 2, mask: tuple[bool, ...] = (True,) * 4
) -> TensorDict:
    """Two graphs (H-C, O-H); *n_graphs* > 2 leaves trailing slots no atom indexes."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.97, 0.0]], dtype=dtype
    )
    pair_index = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])
    atoms = {
        "Z": torch.tensor([1, 6, 8, 1]),
        "pos": pos,
        "batch": torch.tensor([0, 0, 1, 1]),
        "mask": torch.tensor(mask),
    }
    return TensorDict(
        atoms=TensorDict(atoms, batch_size=[4]),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=pos[pair_index[:, 1]] - pos[pair_index[:, 0]],
            batch_size=[4],
        ),
        graphs=TensorDict(batch_size=[n_graphs]),
        batch_size=[],
    )


class TestDressedPotential:
    def test_energy_is_the_residual_plus_the_dress(self) -> None:
        residual = _PairResidual().energy(_frame())
        out = DressedPotential(_PairResidual(), _dress()).energy(_frame())
        dress = torch.tensor([_C1, _C6, _C8, _C1], dtype=torch.float64)
        torch.testing.assert_close(out[ATOMIC_ENERGY_KEY], residual[ATOMIC_ENERGY_KEY] + dress)
        torch.testing.assert_close(
            out[ENERGY_KEY], residual[ENERGY_KEY] + torch.tensor([_C1 + _C6, _C8 + _C1])
        )

    def test_forces_are_the_residual_forces(self) -> None:
        """``dc/dr = 0``."""
        want = _PairResidual(wrt=FORCES)(_frame())[FORCES_KEY]
        got = DressedPotential(_PairResidual(), _dress(), wrt=FORCES)(_frame())[FORCES_KEY]
        torch.testing.assert_close(got, want)

    def test_a_ghost_row_gets_no_dress(self) -> None:
        mask = (True, True, True, False)
        residual = _PairResidual().energy(_frame(mask=mask))[ATOMIC_ENERGY_KEY]
        out = DressedPotential(_PairResidual(), _dress()).energy(_frame(mask=mask))
        assert out[ATOMIC_ENERGY_KEY][3] == residual[3]
        torch.testing.assert_close(out[ENERGY_KEY][1], residual[2:].sum() + _C8)

    def test_a_frame_without_the_mask_column_is_refused(self) -> None:
        frame = _frame()
        del frame["atoms", "mask"]
        with pytest.raises(KeyError, match="mask"):
            DressedPotential(_PairResidual(), _dress()).energy(frame)

    def test_a_graph_slot_no_atom_indexes_stays_zero(self) -> None:
        out = DressedPotential(_PairResidual(), _dress()).energy(_frame(n_graphs=3))
        assert out[ENERGY_KEY].shape == (3,)
        assert out[ENERGY_KEY][2] == 0.0

    def test_a_float32_residual_gets_the_dress_added_in_float64(self) -> None:
        residual = _PairResidual().energy(_frame(dtype=torch.float32))[ATOMIC_ENERGY_KEY]
        out = DressedPotential(_PairResidual(), _dress()).energy(_frame(dtype=torch.float32))
        dress = torch.tensor([_C1, _C6, _C8, _C1], dtype=torch.float64)
        assert out[ATOMIC_ENERGY_KEY].dtype == torch.float32
        assert torch.equal(out[ATOMIC_ENERGY_KEY], (residual.double() + dress).float())

    def test_a_dress_with_an_intercept_is_refused(self) -> None:
        dress = _dress()
        with_intercept = AtomicDress(e0=dress.e0, support=dress.support, intercept=1.0)
        with pytest.raises(ValueError, match="intercept"):
            DressedPotential(_PairResidual(), with_intercept)

    def test_state_dict_carries_the_dress_buffer(self) -> None:
        assert "dress.e0" in DressedPotential(_PairResidual(), _dress()).state_dict()

    @pytest.mark.slow
    def test_energy_traces_to_one_graph(self) -> None:
        dressed = DressedPotential(_PairResidual(), _dress())
        explanation = Compiler.explain(dressed.energy, _frame())
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    @pytest.mark.slow
    def test_the_package_reports_absolute_energy(self, tmp_path) -> None:
        residual = AtomEnergyAdapter(_PairResidual())
        dressed = AtomEnergyAdapter(DressedPotential(_PairResidual(), copy.deepcopy(_dress())))
        written = aoti.package(
            dressed,
            example_inputs([1, 6, 8], 4.0, torch.float64, "cpu"),
            tmp_path / "dressed.pt2",
            dynamic_shapes=DYNAMIC_SHAPES,
            inductor_configs={"aot_inductor.use_runtime_constant_folding": True},
        )
        runner = aoti.load(written.path)
        Z, pos, pair_index, shifts, local_mask = example_inputs(
            [1, 6, 8], 5.0, torch.float64, "cpu"
        )
        local_mask[::3] = False
        energy, forces, atom_energy, _ = runner(Z, pos, pair_index, shifts, local_mask)
        want_energy, want_forces, want_atoms, _ = residual(Z, pos, pair_index, shifts, local_mask)
        owned = _dress().per_atom(Z) * local_mask.double()
        torch.testing.assert_close(energy, want_energy + owned.sum())
        torch.testing.assert_close(atom_energy, want_atoms + _dress().per_atom(Z))
        torch.testing.assert_close(forces, want_forces)
