"""Unit tests for ``molix.data.residual.EnergyResidual``.

In-flight label residualizer: frozen Z-indexed ``c[Z]`` summed over real atoms
only, subtracted from ``graphs.energy`` in float64, written to
``graphs.energy_residual``. Raw totals stay put. Ghost rows are killed by
``atoms.mask`` (``torch.where``, not multiply — a NaN ghost gather would
poison ``e0``).

Goldens are hard-coded numpy float64 identities (2026-09-11, spec
``cache-layers-05-residual`` Domain basis). No ``energy_loss`` / ``MSELoss``,
no molpot, no DataLoader.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data import MoleculePadder, collate_molecules
from molix.data.residual import EnergyResidual
from molix.schema import E0_KEY, ENERGY_KEY, ENERGY_RESIDUAL_KEY, FORCES_KEY
from tests.conftest import frame

# numpy float64, 2026-09-11, Domain-basis H2O numbers (eV).
_H2O_Z = (8, 1, 1)
_H2O_POS = (
    (0.0, 0.0, 0.0),
    (0.9572, 0.0, 0.0),
    (-0.23999, 0.9272, 0.0),
)
_H2O_C1 = -13.6
_H2O_C8 = -2043.8
_H2O_E_TOTAL = -2071.1234567890123
_H2O_E0 = -2071.0
_H2O_E_RES = -0.1234567890123

# numpy float64, 2026-09-11, CH4 large-total case (eV).
_CH4_Z = (6, 1, 1, 1, 1)
_CH4_POS = (
    (0.0, 0.0, 0.0),
    (0.629, 0.629, 0.629),
    (0.629, -0.629, -0.629),
    (-0.629, 0.629, -0.629),
    (-0.629, -0.629, 0.629),
)
_CH4_C1 = -13.60569312299
_CH4_C6 = -1030.08
_CH4_E_TOTAL = -1084.412345678901
_CH4_E0 = -1084.50277249196
_CH4_E_RES = 0.090426813059

_TOL = 1e-12
_CUTOFF = 4.5


def _table(values: dict[int, float], size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense Z-indexed ``(coefficients, support)`` of length *size*."""
    coefficients = torch.zeros(size, dtype=torch.float64)
    support = torch.zeros(size, dtype=torch.bool)
    for atomic_number, coeff in values.items():
        coefficients[atomic_number] = coeff
        support[atomic_number] = True
    return coefficients, support


def _h2o_table(*, size: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    return _table({1: _H2O_C1, 8: _H2O_C8}, size)


def _ch4_table() -> tuple[torch.Tensor, torch.Tensor]:
    return _table({1: _CH4_C1, 6: _CH4_C6}, 7)


def _molecule_batch(
    z: tuple[int, ...],
    pos: tuple[tuple[float, float, float], ...],
    energy: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> TensorDict:
    return collate_molecules([frame(Z=z, pos=pos, energy=energy)], dtype=dtype)


def _h2o_batch(*, dtype: torch.dtype = torch.float64) -> TensorDict:
    return _molecule_batch(_H2O_Z, _H2O_POS, _H2O_E_TOTAL, dtype=dtype)


def _ch4_batch(*, dtype: torch.dtype = torch.float64) -> TensorDict:
    return _molecule_batch(_CH4_Z, _CH4_POS, _CH4_E_TOTAL, dtype=dtype)


def _per_atom_reference(
    z: torch.Tensor,
    mask: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Float64 ``torch.where(mask, c[Z], 0)`` — ``c[Z] * mask`` poisons on NaN ghosts."""
    table = coefficients.to(dtype=torch.float64)
    gathered = table[z.long().clamp(0, table.shape[0] - 1)]
    return torch.where(mask.to(dtype=torch.bool), gathered, torch.zeros_like(gathered))


def _scalar(value: torch.Tensor) -> float:
    return float(value.reshape(()).to(dtype=torch.float64))


class TestEnergyResidual:
    """``EnergyResidual`` — ``BatchTask`` that owns ``graphs.energy_residual``."""

    def test_positional_arguments_are_type_error(self) -> None:
        coefficients, support = _h2o_table()
        with pytest.raises(TypeError):
            EnergyResidual(coefficients, support)  # type: ignore[misc]

    def test_keyword_construction_runs(self) -> None:
        coefficients, support = _h2o_table()
        EnergyResidual(coefficients=coefficients, support=support)

    def test_coefficients_must_be_1d(self) -> None:
        support = torch.zeros(9, dtype=torch.bool)
        with pytest.raises(ValueError):
            EnergyResidual(coefficients=torch.zeros(9, 1), support=support)
        with pytest.raises(ValueError):
            EnergyResidual(coefficients=torch.tensor(1.0), support=torch.zeros(1, dtype=torch.bool))

    def test_support_must_be_1d(self) -> None:
        coefficients = torch.zeros(9, dtype=torch.float64)
        with pytest.raises(ValueError):
            EnergyResidual(
                coefficients=coefficients,
                support=torch.zeros(9, 1, dtype=torch.bool),
            )

    def test_coefficients_and_support_must_be_same_length(self) -> None:
        with pytest.raises(ValueError):
            EnergyResidual(
                coefficients=torch.zeros(9, dtype=torch.float64),
                support=torch.zeros(8, dtype=torch.bool),
            )

    def test_support_uint8_is_not_bool(self) -> None:
        coefficients = torch.zeros(9, dtype=torch.float64)
        support = torch.zeros(9, dtype=torch.uint8)
        support[1] = 1
        support[8] = 1
        with pytest.raises(ValueError):
            EnergyResidual(coefficients=coefficients, support=support)

    def test_unpadded_h2o_float64_subtract_matches_oracle(self) -> None:
        """e0 = c[8] + 2*c[1] = -2071.0 eV; E_res = E_total - e0 within 1e-12 eV."""
        coefficients, support = _h2o_table()
        batch = _h2o_batch()
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)

        e0 = out[E0_KEY]
        assert e0.dtype == torch.float64
        assert _scalar(e0) == pytest.approx(_H2O_E0, abs=_TOL)

        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - e0
        assert _scalar(residual_f64) == pytest.approx(_H2O_E_RES, abs=_TOL)
        assert out[ENERGY_RESIDUAL_KEY].dtype == batch["atoms", "pos"].dtype
        assert _scalar(out[ENERGY_RESIDUAL_KEY]) == pytest.approx(_H2O_E_RES, abs=_TOL)

    def test_graphs_energy_is_the_same_tensor_after_execute(self) -> None:
        coefficients, support = _h2o_table()
        batch = _h2o_batch()
        energy = batch[ENERGY_KEY]
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        assert out is batch
        assert out[ENERGY_KEY] is energy
        assert ("atoms", "energy") not in out.keys(include_nested=True)
        assert FORCES_KEY not in out.keys(include_nested=True)

    def test_graphs_e0_is_always_written_float64(self) -> None:
        coefficients, support = _h2o_table()
        out = EnergyResidual(coefficients=coefficients, support=support).execute(_h2o_batch())
        assert E0_KEY in out.keys(include_nested=True)
        assert out[E0_KEY].dtype == torch.float64
        assert out[E0_KEY].shape == (1,)

    def test_energy_residual_column_is_populated_at_the_schema_key(self) -> None:
        coefficients, support = _h2o_table()
        out = EnergyResidual(coefficients=coefficients, support=support).execute(_h2o_batch())
        residual = out[ENERGY_RESIDUAL_KEY]
        assert residual.shape == (1,)
        assert ENERGY_RESIDUAL_KEY == ("graphs", "energy_residual")

    def test_raw_energy_is_not_used_as_the_residual(self) -> None:
        coefficients, support = _h2o_table()
        batch = _h2o_batch()
        energy = batch[ENERGY_KEY]
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        residual = out[ENERGY_RESIDUAL_KEY]
        assert residual is not energy
        assert not torch.equal(residual.to(dtype=torch.float64), energy.to(dtype=torch.float64))

    def test_caller_coefficient_mutation_is_invisible(self) -> None:
        coefficients, support = _h2o_table()
        task = EnergyResidual(coefficients=coefficients, support=support)
        coefficients.zero_()
        out = task.execute(_h2o_batch())
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_fp32_pos_keeps_float64_e0_and_stores_residual_in_pos_dtype(self) -> None:
        """Subtract stays float64; stored residual follows ``atoms.pos``.

        ``float32(E_total) - float32(e0)`` does not recover the 1e-12 eV identity.
        """
        coefficients, support = _h2o_table()
        batch = _h2o_batch(dtype=torch.float32)
        assert batch["atoms", "pos"].dtype == torch.float32
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)

        e0 = out[E0_KEY]
        assert e0.dtype == torch.float64
        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - e0
        assert _scalar(e0) == pytest.approx(_H2O_E0, abs=_TOL)
        assert _scalar(residual_f64) == pytest.approx(_H2O_E_RES, abs=_TOL)

        stored = out[ENERGY_RESIDUAL_KEY]
        assert stored.dtype == torch.float32
        assert torch.equal(stored, residual_f64.to(dtype=torch.float32))

    def test_padded_ghost_hydrogen_contributes_zero(self) -> None:
        """Default pad Z=1 must not add ``c[1]`` per ghost; real H still counts."""
        coefficients, support = _h2o_table()
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=1)(
            _h2o_batch()
        )
        mask = batch["atoms", "mask"]
        assert bool((~mask).any())
        n_ghost = int((~mask).sum())
        assert torch.equal(batch["atoms", "Z"][~mask], torch.ones(n_ghost, dtype=torch.long))

        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        expected = _per_atom_reference(batch["atoms", "Z"], mask, coefficients)
        e0_oracle = torch.zeros(int(batch["graphs"].batch_size[0]), dtype=torch.float64)
        e0_oracle.scatter_add_(0, batch["atoms", "batch"], expected)

        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)
        assert torch.equal(out[E0_KEY].cpu(), e0_oracle)
        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - out[E0_KEY]
        assert _scalar(residual_f64) == pytest.approx(_H2O_E_RES, abs=_TOL)

    def test_ghost_nan_gather_does_not_poison_e0(self) -> None:
        """``c[Z]*mask`` is NaN when a ghost gathers NaN; ``torch.where`` is not."""
        coefficients, support = _h2o_table(size=10)
        coefficients = coefficients.clone()
        coefficients[3] = float("nan")
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=3)(
            _h2o_batch()
        )
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)

        expected = _per_atom_reference(batch["atoms", "Z"], batch["atoms", "mask"], coefficients)
        assert not torch.isnan(expected[batch["atoms", "mask"]]).any()
        assert torch.isnan(coefficients[3])
        assert not torch.isnan(out[E0_KEY]).any()
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_real_unsupported_z_raises(self) -> None:
        coefficients, support = _h2o_table()
        support = support.clone()
        support[8] = False
        with pytest.raises(ValueError, match="8"):
            EnergyResidual(coefficients=coefficients, support=support).execute(_h2o_batch())

    def test_ghost_unsupported_z_does_not_raise(self) -> None:
        coefficients, support = _h2o_table(size=10)
        coefficients = coefficients.clone()
        coefficients[9] = 100.0
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=9)(
            _h2o_batch()
        )
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_real_z_out_of_table_raises(self) -> None:
        coefficients, support = _h2o_table(size=9)
        batch = _molecule_batch((17,), ((0.0, 0.0, 0.0),), 0.0)
        with pytest.raises(ValueError, match="17"):
            EnergyResidual(coefficients=coefficients, support=support).execute(batch)

    def test_ghost_z_out_of_table_does_not_raise(self) -> None:
        coefficients, support = _h2o_table(size=9)
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=99)(
            _h2o_batch()
        )
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        ghost = ~batch["atoms", "mask"]
        expected = _per_atom_reference(batch["atoms", "Z"], batch["atoms", "mask"], coefficients)
        assert torch.equal(expected[ghost], torch.zeros(int(ghost.sum()), dtype=torch.float64))
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_zero_coefficient_without_support_still_raises_on_real_atom(self) -> None:
        coefficients, support = _h2o_table()
        coefficients = coefficients.clone()
        support = support.clone()
        coefficients[8] = 0.0
        support[8] = False
        with pytest.raises(ValueError, match="8"):
            EnergyResidual(coefficients=coefficients, support=support).execute(_h2o_batch())

    def test_plain_dict_is_refused(self) -> None:
        coefficients, support = _h2o_table()
        batch = _h2o_batch()
        mapping = {name: batch[name] for name in batch.keys()}
        with pytest.raises(TypeError):
            EnergyResidual(coefficients=coefficients, support=support).execute(mapping)

    def test_in_place_energy_write_leaves_residual_identity(self) -> None:
        coefficients, support = _h2o_table()
        out = EnergyResidual(coefficients=coefficients, support=support).execute(_h2o_batch())
        residual = out[ENERGY_RESIDUAL_KEY]
        residual_values = residual.clone()
        out[ENERGY_KEY] = torch.zeros_like(out[ENERGY_KEY])
        assert out[ENERGY_RESIDUAL_KEY] is residual
        assert torch.equal(out[ENERGY_RESIDUAL_KEY], residual_values)
        assert FORCES_KEY not in out.keys(include_nested=True)

    def test_e0_is_invariant_under_pos_translation(self) -> None:
        coefficients, support = _h2o_table()
        task = EnergyResidual(coefficients=coefficients, support=support)
        out = task.execute(_h2o_batch())
        e0 = out[E0_KEY].clone()
        shift = torch.tensor([1.0, -2.0, 3.0], dtype=out["atoms", "pos"].dtype)
        out["atoms", "pos"] = out["atoms", "pos"] + shift
        again = task.execute(out)
        assert torch.equal(again[E0_KEY], e0)

    def test_two_tables_produce_different_task_ids(self) -> None:
        h2o_c, h2o_s = _h2o_table()
        ch4_c, ch4_s = _ch4_table()
        h2o_id = EnergyResidual(coefficients=h2o_c, support=h2o_s).task_id
        ch4_id = EnergyResidual(coefficients=ch4_c, support=ch4_s).task_id
        assert h2o_id.startswith("energy_residual:")
        assert ch4_id.startswith("energy_residual:")
        assert h2o_id != ch4_id

    def test_ch4_float64_subtract_matches_oracle(self) -> None:
        """e0 = c[6] + 4*c[1]; E_res = E_total - e0 within 1e-12 eV."""
        coefficients, support = _ch4_table()
        batch = _ch4_batch()
        out = EnergyResidual(coefficients=coefficients, support=support).execute(batch)
        e0 = out[E0_KEY]
        assert e0.dtype == torch.float64
        assert _scalar(e0) == pytest.approx(_CH4_E0, abs=_TOL)
        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - e0
        assert _scalar(residual_f64) == pytest.approx(_CH4_E_RES, abs=_TOL)
        assert _scalar(out[ENERGY_RESIDUAL_KEY]) == pytest.approx(_CH4_E_RES, abs=_TOL)
