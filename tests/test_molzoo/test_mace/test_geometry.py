"""Tests for molzoo.mace.geometry — MACE's edge displacement / length owner.

MACE's edge vector is ``r_ij = pos[target] - pos[source] + S_ij`` with an
*additive* PBC shift ``S_ij = n_ij · h``. This is a different mathematical
object from PiNet's minimum-image straight-through ``edge_bond_diff``
(``src/molzoo/pinet/geometry.py:28-49``), which is why MACE owns its own
geometry module — see that module's docstring for the rationale.

All expectations are hard-coded from three literal coordinates; nothing here
is derived by re-running the function under test.
"""

from __future__ import annotations

import pytest
import torch
from molzoo.mace.geometry import edge_lengths, edge_vectors

#: Three atoms at literal coordinates (Å).
POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

#: Four directed edges, ``[:, 0]`` = source, ``[:, 1]`` = target.
EDGE_INDEX = [[0, 1], [1, 0], [0, 2], [2, 1]]

#: ``pos[target] - pos[source]``, by hand.
EXPECTED_VECTORS = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, -2.0, 0.0]]

#: ``‖r_ij‖`` for the four edges above; ``sqrt(5) = 2.23606797749979``.
EXPECTED_LENGTHS = [1.0, 1.0, 2.0, 2.23606797749979]

#: Integer images ``n_ij`` for a 3.0 Å cubic box, i.e. ``S = n · 3.0``.
UNIT_SHIFTS = [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 1, 0]]
BOX_LENGTH = 3.0

#: ``pos[target] - pos[source] + S_ij``, by hand.
EXPECTED_SHIFTED_VECTORS = [
    [4.0, 0.0, 0.0],
    [-4.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0],
]

#: ``‖r_ij + S_ij‖``; ``sqrt(2) = 1.4142135623730951``.
EXPECTED_SHIFTED_LENGTHS = [4.0, 4.0, 2.0, 1.4142135623730951]

#: ``d(Σ r_ij)/d pos`` = (in-degree − out-degree) per atom, per component.
#: Atom 0 is a target once and a source twice, atom 1 twice/once, atom 2 once/once.
EXPECTED_VECTOR_SUM_GRAD = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]

#: ``d(Σ ‖r_ij‖)/d pos`` = Σ ±r̂; ``1/sqrt(5) = 0.4472135954999579``.
EXPECTED_LENGTH_SUM_GRAD = [
    [-2.0, -1.0, 0.0],
    [2.4472135954999579, -0.8944271909999159, 0.0],
    [-0.4472135954999579, 1.8944271909999159, 0.0],
]

#: Positions are exact to 1e-12; these are fp64 sums of a handful of terms.
POSITION_ATOL = 1e-12


@pytest.fixture
def pos() -> torch.Tensor:
    """``(3, 3)`` fp64 positions in Å."""
    return torch.tensor(POS, dtype=torch.float64)


@pytest.fixture
def edge_index() -> torch.Tensor:
    """``(4, 2)`` source/target index pairs."""
    return torch.tensor(EDGE_INDEX, dtype=torch.long)


@pytest.fixture
def shifts() -> torch.Tensor:
    """``(4, 3)`` fp64 PBC shift vectors for a 3.0 Å cubic box."""
    return torch.tensor(UNIT_SHIFTS, dtype=torch.float64) * BOX_LENGTH


class TestEdgeVectors:
    """Test ``edge_vectors(pos, edge_index, shifts=None)``."""

    def test_matches_hand_computed_displacements(self, pos, edge_index):
        """``r_ij = pos[target] - pos[source]`` — the repo-wide edge convention."""
        vectors = edge_vectors(pos, edge_index)
        expected = torch.tensor(EXPECTED_VECTORS, dtype=torch.float64)
        assert torch.allclose(vectors, expected, atol=POSITION_ATOL, rtol=0.0)

    def test_output_shape_is_one_vector_per_edge(self, pos, edge_index):
        """``(E, 3)`` regardless of the atom count."""
        assert edge_vectors(pos, edge_index).shape == (len(EDGE_INDEX), 3)

    def test_shifts_are_added_to_the_displacement(self, pos, edge_index, shifts):
        """``S_ij`` enters additively — not as a minimum-image wrap."""
        vectors = edge_vectors(pos, edge_index, shifts=shifts)
        expected = torch.tensor(EXPECTED_SHIFTED_VECTORS, dtype=torch.float64)
        assert torch.allclose(vectors, expected, atol=POSITION_ATOL, rtol=0.0)

    def test_shifted_result_equals_shift_free_plus_shifts(self, pos, edge_index, shifts):
        """The additive law, stated directly against the shift-free call."""
        shifted = edge_vectors(pos, edge_index, shifts=shifts)
        assert torch.allclose(
            shifted, edge_vectors(pos, edge_index) + shifts, atol=POSITION_ATOL, rtol=0.0
        )

    def test_position_gradient_matches_the_edge_incidence(self, pos, edge_index):
        """``∂r/∂pos`` is the signed incidence matrix of the directed graph."""
        leaf = pos.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(edge_vectors(leaf, edge_index).sum(), leaf)
        expected = torch.tensor(EXPECTED_VECTOR_SUM_GRAD, dtype=torch.float64)
        assert torch.allclose(grad, expected, atol=POSITION_ATOL, rtol=0.0)

    def test_position_gradient_is_unchanged_by_shifts(self, pos, edge_index, shifts):
        """``S_ij`` is constant w.r.t. ``pos``, so it drops out of the gradient."""
        leaf = pos.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(edge_vectors(leaf, edge_index, shifts=shifts).sum(), leaf)
        expected = torch.tensor(EXPECTED_VECTOR_SUM_GRAD, dtype=torch.float64)
        assert torch.allclose(grad, expected, atol=POSITION_ATOL, rtol=0.0)


class TestEdgeLengths:
    """Test ``edge_lengths(vectors, *, keepdim=False)``."""

    def test_matches_hand_computed_distances(self, pos, edge_index):
        """``d_ij = ‖r_ij‖`` for the four literal edges."""
        lengths = edge_lengths(edge_vectors(pos, edge_index))
        expected = torch.tensor(EXPECTED_LENGTHS, dtype=torch.float64)
        assert torch.allclose(lengths, expected, atol=POSITION_ATOL, rtol=0.0)

    def test_default_shape_is_one_scalar_per_edge(self, pos, edge_index):
        """``keepdim=False`` (the default) gives ``(E,)`` — MatPES's shape."""
        assert edge_lengths(edge_vectors(pos, edge_index)).shape == (len(EDGE_INDEX),)

    def test_keepdim_shape_is_a_trailing_singleton(self, pos, edge_index):
        """``keepdim=True`` gives ``(E, 1)`` — OMOL's shape."""
        lengths = edge_lengths(edge_vectors(pos, edge_index), keepdim=True)
        assert lengths.shape == (len(EDGE_INDEX), 1)

    def test_shifted_lengths_are_the_minimum_image_distances(self, pos, edge_index, shifts):
        """With ``S_ij`` folded in, the length is the imaged distance."""
        lengths = edge_lengths(edge_vectors(pos, edge_index, shifts=shifts))
        expected = torch.tensor(EXPECTED_SHIFTED_LENGTHS, dtype=torch.float64)
        assert torch.allclose(lengths, expected, atol=POSITION_ATOL, rtol=0.0)

    def test_position_gradient_is_the_analytic_unit_vectors(self, pos, edge_index):
        """``∂‖r‖/∂pos`` accumulates ``+r̂`` on the target and ``−r̂`` on the source."""
        leaf = pos.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(edge_lengths(edge_vectors(leaf, edge_index)).sum(), leaf)
        expected = torch.tensor(EXPECTED_LENGTH_SUM_GRAD, dtype=torch.float64)
        assert torch.allclose(grad, expected, atol=POSITION_ATOL, rtol=0.0)
