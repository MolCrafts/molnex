"""Tests for the per-atom energy heads in :mod:`molpot.heads.energy`.

``TestEnergyHead.test_forward_energy`` was moved from the stale
``tests/test_molpot/test_readout/`` mirror — the production module is
``src/molpot/heads/energy.py``, so the mirror is
``tests/test_molpot/test_heads/test_energy.py``.
"""

from __future__ import annotations

import pytest
import torch

from molpot.heads.energy import AtomicEnergyMLP, AtomicReferenceEnergy, EnergyHead

#: Floating dtype every module under test is built at.
DTYPE = torch.float32

# omol_elem_refs[Z] / eV; index in the scatter table is atomic number.
# Offline extract of key omol_elem_refs from
# configs/allscaip/element_refs/uma_v1_hof_lin_refs.yaml
# (fairchem@41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9, 2026-09-11).
# SHA-256 e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
_OMOL_ATOMIC_NUMBERS = [1, 6, 8, 26, 36]
_OMOL_ELEM_REFS_EV = [
    -16.001316695716987,  # H
    -1037.0951083942327,  # C
    -2047.6980007878883,  # O
    -34381.61625929281,  # Fe
    -74932.45491967947,  # Kr
]
_C_H_EV = _OMOL_ELEM_REFS_EV[0]
_WATER_E0_EV = -2079.7006341793223  # 2*c[H] + c[O]


def _omol_hof_table() -> AtomicReferenceEnergy:
    return AtomicReferenceEnergy(
        atomic_energies=_OMOL_ELEM_REFS_EV,
        atomic_numbers=_OMOL_ATOMIC_NUMBERS,
    )


class TestAtomicEnergyMLP:
    def test_all_parameters_are_built_at_the_given_dtype(self):
        """Every parameter is fp64 when the head is built at fp64.

        A layer that ignored the constructor dtype would leave the head
        mixed-precision, and its first forward would die on a dtype-mismatched
        matmul.
        """
        head = AtomicEnergyMLP(hidden_dim=4, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


class TestEnergyHead:
    def test_forward_energy(self):
        head = EnergyHead(hidden_dim=4, dtype=DTYPE)
        h = torch.ones(3, 4)
        batch = torch.tensor([0, 0, 1])
        out = head(h, batch)
        assert out.shape == torch.Size([2])

    def test_all_parameters_are_built_at_the_given_dtype(self):
        """Every parameter is fp64 when the head is built at fp64."""
        head = EnergyHead(hidden_dim=4, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_at_the_given_dtype(self):
        """A head built at fp64 pools fp64 atomic energies into fp64 totals."""
        head = EnergyHead(hidden_dim=4, dtype=torch.float64)

        out = head(torch.ones(3, 4, dtype=torch.float64), torch.tensor([0, 0, 1]))

        assert out.shape == torch.Size([2])
        assert out.dtype == torch.float64


class TestAtomicReferenceEnergy:
    def test_scatter_forward_returns_omol_elem_refs_as_float64(self):
        """Scatter constructor returns the published H/C/O/Fe/Kr literals as float64.

        SHA-256 e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
        of ``uma_v1_hof_lin_refs.yaml``; fairchem commit
        ``41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9``, 2026-09-11.
        """
        head = AtomicReferenceEnergy(
            atomic_energies=_OMOL_ELEM_REFS_EV,
            atomic_numbers=_OMOL_ATOMIC_NUMBERS,
            scheme="combined_hof_linear",
        )
        out = head.forward(torch.tensor(_OMOL_ATOMIC_NUMBERS, dtype=torch.long))

        assert out.dtype == torch.float64
        torch.testing.assert_close(
            out,
            torch.tensor(_OMOL_ELEM_REFS_EV, dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )

    def test_pooled_water_equals_two_h_plus_o(self):
        head = _omol_hof_table()
        out = head.pooled(
            torch.tensor([8, 1, 1], dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            num_graphs=1,
        )

        assert out.dtype == torch.float64
        assert out.shape == (1,)
        assert out.item() == _WATER_E0_EV

    def test_state_dict_round_trips_buffers_and_extra_state(self):
        head = AtomicReferenceEnergy(
            atomic_energies=_OMOL_ELEM_REFS_EV,
            atomic_numbers=_OMOL_ATOMIC_NUMBERS,
            scheme="combined_hof_linear",
            units="eV",
        )

        assert list(head.parameters()) == []
        state = head.state_dict()
        assert state["atomic_energies"].dtype == torch.float64
        assert state["supported"].dtype == torch.bool
        extra = state["_extra_state"]
        assert extra["scheme"] == "combined_hof_linear"
        assert extra["units"] == "eV"

        clone = AtomicReferenceEnergy(
            atomic_energies=[0.0] * len(_OMOL_ELEM_REFS_EV),
            atomic_numbers=_OMOL_ATOMIC_NUMBERS,
            scheme="isolated_atom",
            units="eV",
        )
        clone.load_state_dict(state)
        assert clone.scheme == "combined_hof_linear"
        assert clone.units == "eV"
        torch.testing.assert_close(
            clone.atomic_energies,
            head.atomic_energies,
            atol=0.0,
            rtol=0.0,
        )
        assert torch.equal(clone.supported, head.supported)

    def test_load_state_dict_fills_supported_and_extra_state_when_absent(self):
        """Day-1 Class-I checkpoints predate these keys."""
        head = AtomicReferenceEnergy(
            atomic_energies=_OMOL_ELEM_REFS_EV,
            atomic_numbers=_OMOL_ATOMIC_NUMBERS,
            scheme="combined_hof_linear",
            units="eV",
        )
        state = {k: v for k, v in head.state_dict().items() if k == "atomic_energies"}
        clone = AtomicReferenceEnergy(
            atomic_energies=_OMOL_ELEM_REFS_EV,
            atomic_numbers=_OMOL_ATOMIC_NUMBERS,
            scheme="isolated_atom",
            units="eV",
        )
        clone.load_state_dict(state)
        assert torch.equal(clone.supported, head.supported)
        assert clone.scheme == "isolated_atom"

    def test_default_scheme_is_isolated_atom(self):
        head = AtomicReferenceEnergy(
            atomic_energies=[-13.6, -1000.0],
            atomic_numbers=[1, 8],
        )

        assert head.scheme == "isolated_atom"

    def test_dense_constructor_without_supported_raises(self):
        with pytest.raises(ValueError):
            AtomicReferenceEnergy(
                atomic_energies=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            )

    def test_supported_zero_coefficient_does_not_raise(self):
        head = AtomicReferenceEnergy(
            atomic_energies=[0.0],
            atomic_numbers=[2],
        )
        z = torch.tensor([2], dtype=torch.long)

        head.validate(z)
        out = head.pooled(z, torch.zeros(1, dtype=torch.long), num_graphs=1)
        assert out.item() == 0.0

    def test_unsupported_nitrogen_raises_from_validate_and_pooled(self):
        head = _omol_hof_table()
        z = torch.tensor([1, 7, 8], dtype=torch.long)
        batch = torch.zeros(3, dtype=torch.long)

        with pytest.raises(ValueError, match=r"\[7\]"):
            head.pooled(z, batch, num_graphs=1)
        with pytest.raises(ValueError, match=r"\[7\]"):
            head.validate(z)

    def test_out_of_range_z_is_not_clipped(self):
        head = _omol_hof_table()
        z = torch.tensor([1, 999], dtype=torch.long)

        with pytest.raises(ValueError, match=r"\[999\]"):
            head.validate(z)

    def test_pooled_without_mask_counts_ghost_hydrogen(self):
        head = _omol_hof_table()
        out = head.pooled(
            torch.tensor([8, 1, 1, 1], dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            num_graphs=1,
        )

        assert out.item() == _WATER_E0_EV + _C_H_EV

    def test_pooled_with_mask_excludes_ghost_hydrogen(self):
        head = _omol_hof_table()
        mask = torch.tensor([True, True, True, False])
        out = head.pooled(
            torch.tensor([8, 1, 1, 1], dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            num_graphs=1,
            mask=mask,
        )

        assert out.item() == _WATER_E0_EV

    def test_masked_unsupported_z_does_not_raise(self):
        head = _omol_hof_table()
        z = torch.tensor([8, 1, 1, 7], dtype=torch.long)
        mask = torch.tensor([True, True, True, False])

        head.validate(z, mask=mask)
        head.pooled(z, torch.zeros(4, dtype=torch.long), num_graphs=1, mask=mask)

    def test_lookup_buffer_is_float64(self):
        """``c[Z]`` is float64 whatever dtype the model is built at."""
        head = AtomicReferenceEnergy(
            atomic_energies=[-13.6, -1000.0],
            atomic_numbers=[1, 8],
        )

        assert head.atomic_energies.dtype == torch.float64

    def test_float_cast_leaves_atomic_energies_float64(self):
        head = AtomicReferenceEnergy(
            atomic_energies=[-13.6, -1000.0],
            atomic_numbers=[1, 8],
        )
        head.float()

        assert head.atomic_energies.dtype == torch.float64

    def test_float64_cancels_mev_residual_on_350_kr(self):
        """350 Kr, ``E_res = 0.35`` eV: float64 recovers the residual; float32 does not.

        Tool=fairchem YAML key ``omol_elem_refs``, commit
        ``41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9``, 2026-09-11.
        SHA-256 e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640.
        """
        head = _omol_hof_table()
        n_kr = 350
        z = torch.full((n_kr,), 36, dtype=torch.long)
        e0 = head.pooled(z, torch.zeros(n_kr, dtype=torch.long), num_graphs=1)
        residual = 0.35
        e_dft = e0 + residual

        assert e0.dtype == torch.float64
        # ulp(|e0| ≈ 2.62e7 eV) is ~3.7e-9 eV; 1e-12 is tighter than IEEE-754.
        assert abs((e_dft - e0).item() - residual) < 1e-8

        e0_float32_roundtrip = e0.to(torch.float32).to(torch.float64)
        assert abs((e_dft - e0_float32_roundtrip).item() - residual) > 0.1

    def test_forward_does_not_read_positions(self):
        head = _omol_hof_table()
        pos = torch.zeros(3, 3, requires_grad=True)
        head.atomic_energies.requires_grad_(True)
        e0 = head.forward(torch.tensor([8, 1, 1], dtype=torch.long))
        e0.sum().backward()

        assert pos.grad is None
