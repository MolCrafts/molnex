"""Unit tests for ``molix.data.dress.AtomicDress`` — the frozen ``c[Z]`` table.

The subtract on the loader and its oracle values live in ``test_residual.py``;
this module asserts the table itself: construction rules, the digest, the
gather and the float64 inverse of the subtract.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from molix.data import collate_molecules
from molix.data.dress import AtomicDress
from molix.data.residual import EnergyResidual
from molix.schema import E0_KEY, ENERGY_KEY, ENERGY_RESIDUAL_KEY
from tests.conftest import frame

# eV; c[H], c[O] not round in binary.
_C1 = -13.6
_C8 = -2043.8
_H2O_Z = (8, 1, 1)
_H2O_POS = ((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.23999, 0.9272, 0.0))
_H2O_E_TOTAL = -2071.1234567890123


def _h2o_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    e0 = torch.zeros(9, dtype=torch.float64)
    support = torch.zeros(9, dtype=torch.bool)
    e0[1], e0[8] = _C1, _C8
    support[1] = support[8] = True
    return e0, support


def _h2o_dress() -> AtomicDress:
    e0, support = _h2o_tensors()
    return AtomicDress(e0=e0, support=support)


class TestAtomicDress:
    def test_e0_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            AtomicDress(e0=torch.zeros(9, 1), support=torch.zeros(9, dtype=torch.bool))
        with pytest.raises(ValueError, match="1-D"):
            AtomicDress(e0=torch.tensor(1.0), support=torch.zeros(1, dtype=torch.bool))

    def test_support_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            AtomicDress(e0=torch.zeros(9), support=torch.zeros(9, 1, dtype=torch.bool))

    def test_lengths_must_match(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            AtomicDress(e0=torch.zeros(9), support=torch.zeros(8, dtype=torch.bool))

    def test_an_empty_table_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AtomicDress(e0=torch.zeros(0), support=torch.zeros(0, dtype=torch.bool))

    def test_uint8_support_is_not_bool(self) -> None:
        with pytest.raises(ValueError, match="torch.bool"):
            AtomicDress(e0=torch.zeros(9), support=torch.zeros(9, dtype=torch.uint8))

    def test_buffers_are_persistent_float64_and_bool(self) -> None:
        dress = AtomicDress(e0=torch.zeros(3, dtype=torch.float32), support=torch.ones(3).bool())
        assert dress.e0.dtype == torch.float64
        assert dress.support.dtype == torch.bool
        assert list(dress.state_dict()) == ["e0", "support", "intercept"]

    def test_caller_mutation_is_invisible(self) -> None:
        e0, support = _h2o_tensors()
        dress = AtomicDress(e0=e0, support=support)
        e0.zero_()
        support.zero_()
        assert float(dress.e0[8]) == _C8
        assert bool(dress.support[8])

    def test_digest_is_sha256_of_table_support_then_intercept(self) -> None:
        e0, support = _h2o_tensors()
        payload = (
            e0.numpy().astype("<f8").tobytes()
            + support.to(torch.uint8).numpy().tobytes()
            + torch.tensor([2.5], dtype=torch.float64).numpy().astype("<f8").tobytes()
        )
        dress = AtomicDress(e0=e0, support=support, intercept=2.5)
        assert dress.digest == hashlib.sha256(payload).hexdigest()[:16]

    def test_equal_numbers_share_a_digest_and_one_bit_does_not(self) -> None:
        assert _h2o_dress().digest == _h2o_dress().digest
        e0, support = _h2o_tensors()
        e0[1] = torch.nextafter(e0[1], torch.tensor(0.0, dtype=torch.float64))
        assert AtomicDress(e0=e0, support=support).digest != _h2o_dress().digest
        e0, support = _h2o_tensors()
        support[2] = True
        assert AtomicDress(e0=e0, support=support).digest != _h2o_dress().digest
        e0, support = _h2o_tensors()
        assert AtomicDress(e0=e0, support=support, intercept=1e-9).digest != _h2o_dress().digest

    def test_per_atom_gathers_the_table_by_atomic_number(self) -> None:
        per_atom = _h2o_dress().per_atom(torch.tensor(_H2O_Z, dtype=torch.int32))
        assert per_atom.dtype == torch.float64
        assert per_atom.tolist() == [_C8, _C1, _C1]

    def test_restore_inverts_the_float64_subtract_exactly(self) -> None:
        dress = _h2o_dress()
        batch = collate_molecules(
            [frame(Z=_H2O_Z, pos=_H2O_POS, energy=_H2O_E_TOTAL)], dtype=torch.float64
        )
        label = batch[ENERGY_KEY].clone()
        out = EnergyResidual(dress).execute(batch)
        assert torch.equal(dress.restore(out[ENERGY_RESIDUAL_KEY], out[E0_KEY]), label)

    def test_restore_of_a_float32_residual_adds_in_float64(self) -> None:
        """A float32 residual is already rounded: restore is not bit-exact to the label."""
        dress = _h2o_dress()
        batch = collate_molecules(
            [frame(Z=_H2O_Z, pos=_H2O_POS, energy=_H2O_E_TOTAL)], dtype=torch.float32
        )
        out = EnergyResidual(dress).execute(batch)
        residual, graph_e0 = out[ENERGY_RESIDUAL_KEY], out[E0_KEY]
        restored = dress.restore(residual, graph_e0)
        assert restored.dtype == torch.float32
        assert torch.equal(restored, (residual.double() + graph_e0).float())
