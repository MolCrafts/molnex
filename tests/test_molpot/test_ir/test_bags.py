"""IR parameter bags — field shapes and validation (learnable-classical-ff-01)."""

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
