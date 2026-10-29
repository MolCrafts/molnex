"""Unit tests for ``molix.data.dress_least_squares.LeastSquaresDress`` — PiNN's fit."""

from __future__ import annotations

import pytest
import torch

from molix.data import FrameDataset, SubsetDataset
from molix.data.dataset import BaseDataset
from molix.data.dress_least_squares import (
    PINV_RTOL,
    InterceptLeastSquaresDress,
    LeastSquaresDress,
)
from tests.conftest import frame

# eV; not round in binary.
_C1 = -13.60569312299
_C8 = -2043.80071263
_H2_POS = ((0.0, 0.0, 0.0), (0.74, 0.0, 0.0))
_H2O_POS = ((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.23999, 0.9272, 0.0))


def _dataset(*frames) -> FrameDataset:
    return FrameDataset.from_records([[f] for f in frames], task="energy")


def _h2_h2o(shift: float = 0.0) -> FrameDataset:
    h2 = [(x + shift, y, z) for x, y, z in _H2_POS]
    h2o = [(x + shift, y, z) for x, y, z in _H2O_POS]
    return _dataset(
        frame(Z=(1, 1), pos=h2, energy=2 * _C1),
        frame(Z=(8, 1, 1), pos=h2o, energy=2 * _C1 + _C8),
    )


class _Forbidden(BaseDataset):
    """Four H2 frames; reading index 2 or 3 fails the test."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        if idx >= 2:
            raise AssertionError(f"held-out index {idx} was read")
        return [frame(Z=(1, 1), pos=_H2_POS, energy=2 * _C1)]


class TestLeastSquaresDress:
    def test_elems_must_not_repeat(self) -> None:
        with pytest.raises(ValueError, match="repeat"):
            LeastSquaresDress([1, 8, 1])

    def test_elems_must_be_non_empty_and_non_negative(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LeastSquaresDress([])
        with pytest.raises(ValueError, match=">= 0"):
            LeastSquaresDress([-1, 8])

    def test_fit_recovers_the_coefficients_of_a_hand_matrix(self) -> None:
        counts = torch.tensor([[2, 0], [2, 1]])
        energy = torch.tensor([2 * _C1, 2 * _C1 + _C8], dtype=torch.float64)
        dress = LeastSquaresDress([1, 8]).fit(counts, energy)
        assert dress.e0.shape == (9,)
        assert dress.support.nonzero().flatten().tolist() == [1, 8]
        assert float(dress.e0[1]) == pytest.approx(_C1, abs=1e-12)
        assert float(dress.e0[8]) == pytest.approx(_C8, abs=1e-12)

    def test_an_element_with_no_atom_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"\[6\]"):
            LeastSquaresDress([1, 6, 8]).fit(
                torch.tensor([[2, 0, 1]]), torch.tensor([0.0], dtype=torch.float64)
            )

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LeastSquaresDress([1, 8]).fit(
                torch.tensor([[2, 1], [-1, 1]]), torch.zeros(2, dtype=torch.float64)
            )

    def test_float_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            LeastSquaresDress([1, 8]).fit(
                torch.tensor([[2.0, 1.0]]), torch.zeros(1, dtype=torch.float64)
            )

    def test_a_rank_deficient_fit_is_pinv_at_the_pinned_cutoff(self) -> None:
        """Two waters: every element present, Gram matrix of rank 1."""
        counts = torch.tensor([[2, 1], [2, 1]])
        energy = torch.tensor([-2071.1, -2071.3], dtype=torch.float64)
        dress = LeastSquaresDress([1, 8]).fit(counts, energy)
        x = counts.double()
        beta = torch.linalg.pinv(x.T @ x, rtol=PINV_RTOL) @ x.T @ energy
        assert torch.equal(dress.e0[[1, 8]], beta)

    def test_fit_dataset_reads_counts_and_labels_not_coordinates(self) -> None:
        dress = LeastSquaresDress([1, 8]).fit_dataset(_h2_h2o())
        assert float(dress.e0[1]) == pytest.approx(_C1, abs=1e-12)
        assert float(dress.e0[8]) == pytest.approx(_C8, abs=1e-12)
        assert LeastSquaresDress([1, 8]).fit_dataset(_h2_h2o(shift=7.5)).digest == dress.digest

    def test_fit_dataset_reads_only_the_split_it_is_given(self) -> None:
        dress = LeastSquaresDress([1]).fit_dataset(SubsetDataset(_Forbidden(), [0, 1]))
        assert float(dress.e0[1]) == pytest.approx(_C1, abs=1e-12)

    def test_a_frame_with_two_energy_labels_is_refused(self) -> None:
        both = frame(Z=(1, 1), pos=_H2_POS, energy=2 * _C1)
        both.meta["energy"] = 2 * _C1
        with pytest.raises(ValueError, match="two energy labels"):
            LeastSquaresDress([1]).fit_dataset(_dataset(both))

    def test_a_frame_without_an_energy_label_is_named(self) -> None:
        with pytest.raises(KeyError, match=r"dataset\[1\]"):
            LeastSquaresDress([1]).fit_dataset(
                _dataset(frame(Z=(1, 1), pos=_H2_POS, energy=2 * _C1), frame(Z=(1, 1), pos=_H2_POS))
            )

    def test_an_element_outside_elems_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[8\]"):
            LeastSquaresDress([1]).fit_dataset(_h2_h2o())


class TestInterceptLeastSquaresDress:
    def test_fit_recovers_the_coefficients_and_the_intercept(self) -> None:
        b = 312.4071
        counts = torch.tensor([[2, 0], [2, 1], [4, 1]])
        energy = (counts.double() @ torch.tensor([_C1, _C8], dtype=torch.float64)) + b
        dress = InterceptLeastSquaresDress([1, 8]).fit(counts, energy)
        assert float(dress.e0[1]) == pytest.approx(_C1, abs=1e-9)
        assert float(dress.e0[8]) == pytest.approx(_C8, abs=1e-9)
        assert float(dress.intercept) == pytest.approx(b, abs=1e-9)

    def test_the_intercept_is_a_ones_column_at_the_pinned_cutoff(self) -> None:
        counts = torch.tensor([[2, 0], [2, 1], [2, 1]])
        energy = torch.tensor([-27.2, -2071.1, -2071.3], dtype=torch.float64)
        dress = InterceptLeastSquaresDress([1, 8]).fit(counts, energy)
        x = torch.cat([counts.double(), torch.ones(3, 1, dtype=torch.float64)], dim=1)
        beta = torch.linalg.pinv(x.T @ x, rtol=PINV_RTOL) @ x.T @ energy
        assert torch.equal(dress.e0[[1, 8]], beta[:2])
        assert float(dress.intercept) == float(beta[2])
