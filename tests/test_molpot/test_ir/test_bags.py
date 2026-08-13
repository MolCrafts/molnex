"""IR parameter bags — field shapes and validation (learnable-classical-ff-01)."""

from dataclasses import fields

import pytest
import torch

from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    ProperTorsionBag,
)


class TestBondBag:
    def test_construct_with_equal_length_tensors(self):
        bag = BondBag(k=torch.tensor([100.0, 200.0]), r0=torch.tensor([1.0, 1.5]))
        assert bag.k.shape == bag.r0.shape
        assert bag.k.shape[0] == 2

    def test_mismatch_lengths_raise(self):
        with pytest.raises((ValueError, TypeError)):
            BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.0, 1.5]))


class TestAngleBag:
    def test_construct_with_equal_length_tensors(self):
        bag = AngleBag(
            k=torch.tensor([50.0, 60.0]),
            theta0=torch.tensor([1.9, 2.0]),
        )
        assert bag.k.shape == bag.theta0.shape

    def test_mismatch_lengths_raise(self):
        with pytest.raises((ValueError, TypeError)):
            AngleBag(k=torch.tensor([50.0, 60.0]), theta0=torch.tensor([1.9]))


class TestProperTorsionBag:
    def test_construct_with_equal_term_tables(self):
        # k / phase: [n_types, n_terms]; periodicity: [n_terms]; idivf/s: [n_types]
        bag = ProperTorsionBag(
            k=torch.tensor([[1.0, 0.5]]),
            periodicity=torch.tensor([1, 2], dtype=torch.long),
            phase=torch.tensor([[0.0, 0.0]]),
            idivf=torch.tensor([1.0]),
        )
        assert bag.k.shape[-1] == bag.periodicity.shape[0]
        assert bag.phase.shape == bag.k.shape

    def test_mismatch_term_counts_raise(self):
        with pytest.raises((ValueError, TypeError)):
            ProperTorsionBag(
                k=torch.tensor([[1.0, 0.5]]),
                periodicity=torch.tensor([1], dtype=torch.long),
                phase=torch.tensor([[0.0, 0.0]]),
                idivf=torch.tensor([1.0]),
            )

    def test_non_positive_periodicity_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ProperTorsionBag(
                k=torch.tensor([[1.0]]),
                periodicity=torch.tensor([0], dtype=torch.long),
                phase=torch.tensor([[0.0]]),
                idivf=torch.tensor([1.0]),
            )

    def test_negative_periodicity_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ProperTorsionBag(
                k=torch.tensor([[1.0]]),
                periodicity=torch.tensor([-1], dtype=torch.long),
                phase=torch.tensor([[0.0]]),
                idivf=torch.tensor([1.0]),
            )

    def test_fields_exclude_c0_and_cartesian_storage(self):
        names = {f.name for f in fields(ProperTorsionBag)}
        assert names == {"k", "phase", "periodicity", "idivf"}

    def test_from_cartesian_maps_hard_coded_goldens(self):
        dtype = torch.float64
        # (k, γ)=(2, 0) → (a, b)=(2.0, 0.0)
        bag = ProperTorsionBag.from_cartesian(
            a=torch.tensor([[2.0]], dtype=dtype),
            b=torch.tensor([[0.0]], dtype=dtype),
            periodicity=torch.tensor([1], dtype=torch.long),
        )
        assert torch.allclose(bag.k, torch.tensor([[2.0]], dtype=dtype), atol=1e-12)
        assert torch.allclose(bag.phase, torch.tensor([[0.0]], dtype=dtype), atol=1e-12)

        # (k, γ)=(1.5, π) → (a, b)=(-1.5, 0.0)
        bag = ProperTorsionBag.from_cartesian(
            a=torch.tensor([[-1.5]], dtype=dtype),
            b=torch.tensor([[0.0]], dtype=dtype),
            periodicity=torch.tensor([1], dtype=torch.long),
        )
        assert torch.allclose(bag.k, torch.tensor([[1.5]], dtype=dtype), atol=1e-12)
        assert torch.allclose(bag.phase, torch.tensor([[torch.pi]], dtype=dtype), atol=1e-12)

        # (a, b)=(0.6, 0.8) → (k, γ)=(1.0, atan2(0.8, 0.6))
        bag = ProperTorsionBag.from_cartesian(
            a=torch.tensor([[0.6]], dtype=dtype),
            b=torch.tensor([[0.8]], dtype=dtype),
            periodicity=torch.tensor([1], dtype=torch.long),
        )
        assert torch.allclose(bag.k, torch.tensor([[1.0]], dtype=dtype), atol=1e-12)
        assert torch.allclose(
            bag.phase,
            torch.tensor([[0.9272952180016123]], dtype=dtype),
            atol=1e-12,
        )

    def test_from_cartesian_round_trip_amplitudes(self):
        dtype = torch.float64
        a = torch.tensor([[0.6, -1.5]], dtype=dtype)
        b = torch.tensor([[0.8, 0.0]], dtype=dtype)
        bag = ProperTorsionBag.from_cartesian(
            a=a,
            b=b,
            periodicity=torch.tensor([1, 2], dtype=torch.long),
        )
        a_back = bag.k * torch.cos(bag.phase)
        b_back = bag.k * torch.sin(bag.phase)
        assert torch.allclose(a_back, a, atol=1e-12)
        assert torch.allclose(b_back, b, atol=1e-12)

    def test_from_cartesian_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            ProperTorsionBag.from_cartesian(
                a=torch.tensor([[1.0, 0.0]]),
                b=torch.tensor([[0.0]]),
                periodicity=torch.tensor([1, 2], dtype=torch.long),
            )

    def test_from_cartesian_rejects_non_2d(self):
        with pytest.raises(ValueError):
            ProperTorsionBag.from_cartesian(
                a=torch.tensor([0.6, 0.8]),
                b=torch.tensor([0.6, 0.8]),
                periodicity=torch.tensor([1, 2], dtype=torch.long),
            )

    def test_from_cartesian_zero_amplitude_has_zero_phase(self):
        bag = ProperTorsionBag.from_cartesian(
            a=torch.zeros(1, 1, dtype=torch.float64),
            b=torch.zeros(1, 1, dtype=torch.float64),
            periodicity=torch.tensor([1], dtype=torch.long),
        )
        assert torch.allclose(bag.k, torch.zeros(1, 1, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(bag.phase, torch.zeros(1, 1, dtype=torch.float64), atol=1e-12)

    def test_from_cartesian_default_idivf_is_ones(self):
        bag = ProperTorsionBag.from_cartesian(
            a=torch.tensor([[2.0], [0.0]], dtype=torch.float64),
            b=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            periodicity=torch.tensor([1], dtype=torch.long),
        )
        assert bag.idivf.shape == (2,)
        assert torch.equal(bag.idivf, torch.ones(2, dtype=bag.k.dtype))


class TestImproperPeriodicBag:
    def test_construct_with_equal_term_tables(self):
        bag = ImproperPeriodicBag(
            k=torch.tensor([[1.0]]),
            periodicity=torch.tensor([2], dtype=torch.long),
            phase=torch.tensor([[3.141592653589793]]),
            idivf=torch.tensor([1.0]),
        )
        assert bag.k.shape[-1] == 1


class TestImproperHarmonicBag:
    def test_construct_with_equal_length_tensors(self):
        bag = ImproperHarmonicBag(
            k=torch.tensor([10.0]),
            chi0=torch.tensor([0.0]),
        )
        assert bag.k.shape == bag.chi0.shape

    def test_mismatch_lengths_raise(self):
        with pytest.raises((ValueError, TypeError)):
            ImproperHarmonicBag(k=torch.tensor([10.0, 20.0]), chi0=torch.tensor([0.0]))


class TestLJBag:
    def test_construct_with_equal_length_tensors(self):
        bag = LJBag(
            epsilon=torch.tensor([0.1, 0.2]),
            sigma=torch.tensor([3.0, 3.5]),
        )
        assert bag.epsilon.shape == bag.sigma.shape


class TestChargeBag:
    def test_construct(self):
        bag = ChargeBag(q=torch.tensor([0.1, -0.1, 0.0]))
        assert bag.q.shape[0] == 3
