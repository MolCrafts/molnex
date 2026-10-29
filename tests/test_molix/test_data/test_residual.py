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
from molix.data.dress import AtomicDress
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


def _table(
    values: dict[int, float],
    size: int,
    *,
    extra: dict[int, float] | None = None,
    unsupported: tuple[int, ...] = (),
) -> AtomicDress:
    """Dense Z-indexed dress of length *size*.

    *extra* sets coefficients without support; *unsupported* clears support.
    """
    e0 = torch.zeros(size, dtype=torch.float64)
    support = torch.zeros(size, dtype=torch.bool)
    for atomic_number, coeff in values.items():
        e0[atomic_number] = coeff
        support[atomic_number] = True
    for atomic_number, coeff in (extra or {}).items():
        e0[atomic_number] = coeff
    for atomic_number in unsupported:
        support[atomic_number] = False
    return AtomicDress(e0=e0, support=support)


def _h2o_table(*, size: int = 9, **kwargs) -> AtomicDress:
    return _table({1: _H2O_C1, 8: _H2O_C8}, size, **kwargs)


def _ch4_table() -> AtomicDress:
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

    def test_takes_one_dress_positionally(self) -> None:
        dress = _h2o_table()
        assert EnergyResidual(dress).dress is dress
        with pytest.raises(TypeError):
            EnergyResidual(dress.e0, dress.support)  # type: ignore[call-arg]

    def test_a_bare_tensor_is_refused(self) -> None:
        with pytest.raises(TypeError, match="AtomicDress"):
            EnergyResidual(_h2o_table().e0)  # type: ignore[arg-type]

    def test_task_id_is_the_dress_digest(self) -> None:
        dress = _h2o_table()
        task = EnergyResidual(dress)
        assert task.task_id == "energy_residual:" + dress.digest
        assert EnergyResidual(_h2o_table()).task_id == task.task_id

    def test_unpadded_h2o_float64_subtract_matches_oracle(self) -> None:
        """e0 = c[8] + 2*c[1] = -2071.0 eV; E_res = E_total - e0 within 1e-12 eV."""
        dress = _h2o_table()
        batch = _h2o_batch()
        out = EnergyResidual(dress).execute(batch)

        e0 = out[E0_KEY]
        assert e0.dtype == torch.float64
        assert _scalar(e0) == pytest.approx(_H2O_E0, abs=_TOL)

        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - e0
        assert _scalar(residual_f64) == pytest.approx(_H2O_E_RES, abs=_TOL)
        assert out[ENERGY_RESIDUAL_KEY].dtype == batch["atoms", "pos"].dtype
        assert _scalar(out[ENERGY_RESIDUAL_KEY]) == pytest.approx(_H2O_E_RES, abs=_TOL)

    def test_graphs_energy_is_the_same_tensor_after_execute(self) -> None:
        dress = _h2o_table()
        batch = _h2o_batch()
        energy = batch[ENERGY_KEY]
        out = EnergyResidual(dress).execute(batch)
        assert out is batch
        assert out[ENERGY_KEY] is energy
        assert ("atoms", "energy") not in out.keys(include_nested=True)
        assert FORCES_KEY not in out.keys(include_nested=True)

    def test_graphs_e0_is_always_written_float64(self) -> None:
        dress = _h2o_table()
        out = EnergyResidual(dress).execute(_h2o_batch())
        assert E0_KEY in out.keys(include_nested=True)
        assert out[E0_KEY].dtype == torch.float64
        assert out[E0_KEY].shape == (1,)

    def test_energy_residual_column_is_populated_at_the_schema_key(self) -> None:
        dress = _h2o_table()
        out = EnergyResidual(dress).execute(_h2o_batch())
        residual = out[ENERGY_RESIDUAL_KEY]
        assert residual.shape == (1,)
        assert ENERGY_RESIDUAL_KEY == ("graphs", "energy_residual")

    def test_raw_energy_is_not_used_as_the_residual(self) -> None:
        dress = _h2o_table()
        batch = _h2o_batch()
        energy = batch[ENERGY_KEY]
        out = EnergyResidual(dress).execute(batch)
        residual = out[ENERGY_RESIDUAL_KEY]
        assert residual is not energy
        assert not torch.equal(residual.to(dtype=torch.float64), energy.to(dtype=torch.float64))

    def test_fp32_pos_keeps_float64_e0_and_stores_residual_in_pos_dtype(self) -> None:
        """Subtract stays float64; stored residual follows ``atoms.pos``.

        ``float32(E_total) - float32(e0)`` does not recover the 1e-12 eV identity.
        """
        dress = _h2o_table()
        batch = _h2o_batch(dtype=torch.float32)
        assert batch["atoms", "pos"].dtype == torch.float32
        out = EnergyResidual(dress).execute(batch)

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
        dress = _h2o_table()
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=1)(
            _h2o_batch()
        )
        mask = batch["atoms", "mask"]
        assert bool((~mask).any())
        n_ghost = int((~mask).sum())
        assert torch.equal(batch["atoms", "Z"][~mask], torch.ones(n_ghost, dtype=torch.long))

        out = EnergyResidual(dress).execute(batch)
        expected = _per_atom_reference(batch["atoms", "Z"], mask, dress.e0)
        e0_oracle = torch.zeros(int(batch["graphs"].batch_size[0]), dtype=torch.float64)
        e0_oracle.scatter_add_(0, batch["atoms", "batch"], expected)

        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)
        assert torch.equal(out[E0_KEY].cpu(), e0_oracle)
        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - out[E0_KEY]
        assert _scalar(residual_f64) == pytest.approx(_H2O_E_RES, abs=_TOL)

    def test_ghost_nan_gather_does_not_poison_e0(self) -> None:
        """``c[Z]*mask`` is NaN when a ghost gathers NaN; ``torch.where`` is not."""
        dress = _h2o_table(size=10, extra={3: float("nan")})
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=3)(
            _h2o_batch()
        )
        out = EnergyResidual(dress).execute(batch)

        expected = _per_atom_reference(batch["atoms", "Z"], batch["atoms", "mask"], dress.e0)
        assert not torch.isnan(expected[batch["atoms", "mask"]]).any()
        assert torch.isnan(dress.e0[3])
        assert not torch.isnan(out[E0_KEY]).any()
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_real_unsupported_z_raises(self) -> None:
        dress = _h2o_table(unsupported=(8,))
        with pytest.raises(ValueError, match="8"):
            EnergyResidual(dress).execute(_h2o_batch())

    def test_ghost_unsupported_z_does_not_raise(self) -> None:
        dress = _h2o_table(size=10, extra={9: 100.0})
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=9)(
            _h2o_batch()
        )
        out = EnergyResidual(dress).execute(batch)
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_real_z_out_of_table_raises(self) -> None:
        dress = _h2o_table(size=9)
        batch = _molecule_batch((17,), ((0.0, 0.0, 0.0),), 0.0)
        with pytest.raises(ValueError, match="17"):
            EnergyResidual(dress).execute(batch)

    def test_ghost_z_out_of_table_does_not_raise(self) -> None:
        dress = _h2o_table(size=9)
        batch = MoleculePadder(max_atoms=8, max_pairs=4, cutoff=_CUTOFF, pad_atom_type=99)(
            _h2o_batch()
        )
        out = EnergyResidual(dress).execute(batch)
        ghost = ~batch["atoms", "mask"]
        expected = _per_atom_reference(batch["atoms", "Z"], batch["atoms", "mask"], dress.e0)
        assert torch.equal(expected[ghost], torch.zeros(int(ghost.sum()), dtype=torch.float64))
        assert _scalar(out[E0_KEY]) == pytest.approx(_H2O_E0, abs=_TOL)

    def test_zero_coefficient_without_support_still_raises_on_real_atom(self) -> None:
        dress = _h2o_table(extra={8: 0.0}, unsupported=(8,))
        with pytest.raises(ValueError, match="8"):
            EnergyResidual(dress).execute(_h2o_batch())

    def test_plain_dict_is_refused(self) -> None:
        dress = _h2o_table()
        batch = _h2o_batch()
        mapping = {name: batch[name] for name in batch.keys()}
        with pytest.raises(TypeError):
            EnergyResidual(dress).execute(mapping)

    def test_in_place_energy_write_leaves_residual_identity(self) -> None:
        dress = _h2o_table()
        out = EnergyResidual(dress).execute(_h2o_batch())
        residual = out[ENERGY_RESIDUAL_KEY]
        residual_values = residual.clone()
        out[ENERGY_KEY] = torch.zeros_like(out[ENERGY_KEY])
        assert out[ENERGY_RESIDUAL_KEY] is residual
        assert torch.equal(out[ENERGY_RESIDUAL_KEY], residual_values)
        assert FORCES_KEY not in out.keys(include_nested=True)

    def test_e0_is_invariant_under_pos_translation(self) -> None:
        dress = _h2o_table()
        task = EnergyResidual(dress)
        out = task.execute(_h2o_batch())
        e0 = out[E0_KEY].clone()
        shift = torch.tensor([1.0, -2.0, 3.0], dtype=out["atoms", "pos"].dtype)
        out["atoms", "pos"] = out["atoms", "pos"] + shift
        again = task.execute(out)
        assert torch.equal(again[E0_KEY], e0)

    def test_two_tables_produce_different_task_ids(self) -> None:
        h2o_id = EnergyResidual(_h2o_table()).task_id
        ch4_id = EnergyResidual(_ch4_table()).task_id
        assert h2o_id.startswith("energy_residual:")
        assert ch4_id.startswith("energy_residual:")
        assert h2o_id != ch4_id

    def test_the_intercept_is_added_once_per_live_graph(self) -> None:
        """e0 = c[8] + 2*c[1] + b on the H2O graph; a dead padding slot gets nothing."""
        e0, support = _h2o_table().e0, _h2o_table().support
        dress = AtomicDress(e0=e0, support=support, intercept=310.25)
        batch = MoleculePadder(max_atoms=8, max_pairs=4, max_graphs=3, cutoff=_CUTOFF)(_h2o_batch())
        assert batch["graphs", "mask"].tolist() == [True, False, False]
        out = EnergyResidual(dress).execute(batch)
        assert out[E0_KEY].tolist() == [_H2O_E0 + 310.25, 0.0, 0.0]

    def test_ch4_float64_subtract_matches_oracle(self) -> None:
        """e0 = c[6] + 4*c[1]; E_res = E_total - e0 within 1e-12 eV."""
        dress = _ch4_table()
        batch = _ch4_batch()
        out = EnergyResidual(dress).execute(batch)
        e0 = out[E0_KEY]
        assert e0.dtype == torch.float64
        assert _scalar(e0) == pytest.approx(_CH4_E0, abs=_TOL)
        residual_f64 = out[ENERGY_KEY].to(dtype=torch.float64) - e0
        assert _scalar(residual_f64) == pytest.approx(_CH4_E_RES, abs=_TOL)
        assert _scalar(out[ENERGY_RESIDUAL_KEY]) == pytest.approx(_CH4_E_RES, abs=_TOL)
