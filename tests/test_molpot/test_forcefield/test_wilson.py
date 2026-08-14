"""Unit tests for the Wilson B-matrix builder in molpot.forcefield.wilson.

Mirrors ``src/molpot/forcefield/wilson.py``. Each test targets exactly one
method of :class:`~molpot.forcefield.wilson.WilsonBMatrix`. Every expected
value is hard-coded from the closed-form s-vectors of Wilson/Decius/Cross —
no reference implementation and no third-party oracle is imported anywhere.
The finite-difference cross-checks below differentiate the coordinate
*definitions* (``r = ‖x_j − x_i‖``, ``θ = acos(û_ji · û_jk)``) written out in
this file; that is in-test math, not an external oracle.

Layout under test: ``B`` is ``(n_q, 3N)`` with flat Cartesian index
``i = 3a + c`` (``x0,y0,z0,x1,…``), rows ordered *all bonds, then all angles*.

Units: stretch rows are dimensionless (direction cosines, Å/Å); bend rows are
rad·Å⁻¹ ≡ Å⁻¹.

    ∂r/∂x_i = −û,   ∂r/∂x_j = +û
    ∂θ/∂x_i = (cos θ · u − v) / (r_ij sin θ),  u = (x_i − x_j)/r_ij
    ∂θ/∂x_k = (cos θ · v − u) / (r_kj sin θ),  v = (x_k − x_j)/r_kj
    ∂θ/∂x_j = −(∂θ/∂x_i + ∂θ/∂x_k)

    Wilson, E. B.; Decius, J. C.; Cross, P. C. *Molecular Vibrations*,
    McGraw-Hill, 1955, Ch. 4.
    Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941).
    https://doi.org/10.1063/1.1750829

All numeric tests run in float64 (except the explicit float32 dtype check),
always with ``rtol=0``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from molix.core.losses import ProjectedInternalHessianLoss
from molpot.forcefield.wilson import WilsonBMatrix

#: Exactly-representable goldens (integers / ±1 direction cosines).
EXACT_ATOL: float = 1e-12
#: One norm + one division away from exact (rotated / projected builds).
ROTATION_ATOL: float = 1e-10
#: Central-difference truncation + round-off budget at ``h = 1e-5`` Å.
FD_ATOL: float = 1e-7
#: ``torch.linalg.pinv`` round-off budget for the projected Hessian.
PINV_ATOL: float = 1e-6

#: Central-difference step (Å).
FD_STEP: float = 1e-5

#: Wilson stretch row for the unit-length x-aligned dimer: ``(−û, û)``.
DIATOMIC_STRETCH_ROW: list[list[float]] = [[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

#: Wilson bend row for the right-angle trimer i=(1,0,0), j=(0,0,0) (apex),
#: k=(0,1,0): cos θ = 0, sin θ = 1, r_ij = r_kj = 1 ⇒ s_i = −v = (0,−1,0),
#: s_k = −u = (−1,0,0), s_j = −(s_i + s_k) = (1,1,0).
RIGHT_ANGLE_BEND_ROW: list[list[float]] = [[0.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0]]

#: Water, O first then the two H (Å). ``r_OH1 = 0.9572``, ``r_OH2 = 0.9573180``,
#: ``θ_HOH = 104.4701°`` for these exact literals (the rounded coordinates of the
#: experimental 0.9572 Å / 104.52° geometry do not reproduce 104.52° exactly).
WATER_POS: list[list[float]] = [
    [0.0, 0.0, 0.0],
    [0.9572, 0.0, 0.0],
    [-0.23921, 0.92695, 0.0],
]
#: Water bonds O–H1, O–H2 as COO ``(2, n_bonds)``.
WATER_BONDS: list[list[int]] = [[0, 0], [1, 2]]
#: Water bend H1–O–H2 as COO ``(3, n_angles)``; row 1 (atom 0, O) is the apex.
WATER_ANGLE: list[list[int]] = [[1], [0], [2]]

#: Diatomic H₂ block used in the packed-batch tests (Å), separation 0.74.
H2_POS: list[list[float]] = [[3.0, 0.0, 0.0], [3.0, 0.0, 0.74]]

#: Force constant for the GF-consistency identity ``H_x = k BᵀB ⇒ diag(H_q) = k``.
GF_FORCE_CONSTANT: float = 2.0


def _pos(rows: list[list[float]], dtype: torch.dtype = torch.float64) -> Tensor:
    """Positions ``(N, 3)`` from hard-coded literals."""
    return torch.tensor(rows, dtype=dtype)


def _index(rows: list[list[int]]) -> Tensor:
    """COO index tensor ``(arity, n_terms)`` from hard-coded literals."""
    return torch.tensor(rows, dtype=torch.long)


def _bond_length(i: int, j: int) -> Callable[[Tensor], Tensor]:
    """Coordinate definition ``r = ‖x_j − x_i‖`` (Å) for finite differences."""

    def coordinate(pos: Tensor) -> Tensor:
        return torch.linalg.vector_norm(pos[j] - pos[i])

    return coordinate


def _bend_angle(i: int, j: int, k: int) -> Callable[[Tensor], Tensor]:
    """Coordinate definition ``θ = acos(û_ji · û_jk)`` (rad), apex ``j``."""

    def coordinate(pos: Tensor) -> Tensor:
        u = pos[i] - pos[j]
        v = pos[k] - pos[j]
        u = u / torch.linalg.vector_norm(u)
        v = v / torch.linalg.vector_norm(v)
        return torch.acos((u * v).sum())

    return coordinate


def _central_difference(
    coordinate: Callable[[Tensor], Tensor], pos: Tensor, step: float = FD_STEP
) -> Tensor:
    """Central difference ``∂q/∂x`` as a flat ``(3N,)`` row."""
    row = torch.zeros(3 * pos.shape[0], dtype=pos.dtype)
    for atom in range(pos.shape[0]):
        for axis in range(3):
            plus = pos.clone()
            plus[atom, axis] += step
            minus = pos.clone()
            minus[atom, axis] -= step
            row[3 * atom + axis] = (coordinate(plus) - coordinate(minus)) / (2.0 * step)
    return row


def _rotation_matrix(angle: float = 0.3) -> Tensor:
    """Rodrigues rotation about the fixed non-axis direction ``(1,2,3)/√14``."""
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    cross = torch.tensor(
        [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]],
        dtype=torch.float64,
    )
    eye = torch.eye(3, dtype=torch.float64)
    return eye + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _near_linear_pos(degrees: float) -> Tensor:
    """Trimer i=(1,0,0), apex j=(0,0,0), k on the unit circle at ``degrees``."""
    theta = math.radians(degrees)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [math.cos(theta), math.sin(theta), 0.0]],
        dtype=torch.float64,
    )


def _atom_block(rows: Tensor, row: int, atom: int) -> Tensor:
    """The ``(3,)`` s-vector of ``atom`` inside ``row`` of a ``(n_q, 3N)`` build."""
    return rows[row, 3 * atom : 3 * atom + 3]


class TestWilsonBMatrix:
    """``WilsonBMatrix`` row builders — ``stretch_rows`` and ``bend_rows``."""

    def test_stretch_rows_diatomic(self) -> None:
        """Unit x-aligned dimer ⇒ ``[[-1,0,0, 1,0,0]]``; per-atom blocks are unit."""
        pos = _pos([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        rows = WilsonBMatrix().stretch_rows(pos, _index([[0], [1]]))

        assert rows.shape == (1, 6)
        torch.testing.assert_close(
            rows,
            torch.tensor(DIATOMIC_STRETCH_ROW, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )
        for atom in range(2):
            norm = torch.linalg.vector_norm(_atom_block(rows, 0, atom))
            torch.testing.assert_close(
                norm, torch.tensor(1.0, dtype=torch.float64), atol=EXACT_ATOL, rtol=0.0
            )

    def test_bend_rows_right_angle(self) -> None:
        """θ = 90°, r = 1 Å, apex atom 1 ⇒ ``[0,-1,0, 1,1,0, -1,0,0]``."""
        pos = _pos([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        rows = WilsonBMatrix().bend_rows(pos, _index([[0], [1], [2]]))

        assert rows.shape == (1, 9)
        torch.testing.assert_close(
            rows,
            torch.tensor(RIGHT_ANGLE_BEND_ROW, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

    def test_multi_term_stacking(self) -> None:
        """Two bonds + one bend ⇒ 3 rows, bonds first, each row atom-local."""
        pos = _pos(WATER_POS)
        builder = WilsonBMatrix()

        bond_rows = builder.stretch_rows(pos, _index(WATER_BONDS))
        bend_rows = builder.bend_rows(pos, _index(WATER_ANGLE))
        stacked = torch.cat([bond_rows, bend_rows], dim=0)

        assert stacked.shape == (3, 9)
        # Row 0 = bond O–H1: zero on H2; row 1 = bond O–H2: zero on H1.
        assert torch.count_nonzero(_atom_block(stacked, 0, 2)).item() == 0
        assert torch.count_nonzero(_atom_block(stacked, 1, 1)).item() == 0
        # Both bond rows and the bend row are non-zero on their own atoms.
        for row, atoms in ((0, (0, 1)), (1, (0, 2)), (2, (0, 1, 2))):
            for atom in atoms:
                assert torch.linalg.vector_norm(_atom_block(stacked, row, atom)).item() > 0.0

    def test_water_finite_difference(self) -> None:
        """Water rows equal central differences of ``r`` and ``θ`` (h = 1e-5 Å)."""
        pos = _pos(WATER_POS)
        builder = WilsonBMatrix()

        bond_rows = builder.stretch_rows(pos, _index(WATER_BONDS))
        bend_rows = builder.bend_rows(pos, _index(WATER_ANGLE))

        expected_bonds = torch.stack(
            [
                _central_difference(_bond_length(0, 1), pos),
                _central_difference(_bond_length(0, 2), pos),
            ]
        )
        expected_bend = _central_difference(_bend_angle(1, 0, 2), pos).unsqueeze(0)
        torch.testing.assert_close(bond_rows, expected_bonds, atol=FD_ATOL, rtol=0.0)
        torch.testing.assert_close(bend_rows, expected_bend, atol=FD_ATOL, rtol=0.0)

    def test_translation_invariance(self) -> None:
        """Per-row s-vectors sum to zero and a rigid shift leaves ``B`` unchanged."""
        pos = _pos(WATER_POS)
        shift = torch.tensor([1.5, -2.25, 3.125], dtype=torch.float64)
        builder = WilsonBMatrix()
        bonds, angles = _index(WATER_BONDS), _index(WATER_ANGLE)

        rows = torch.cat([builder.stretch_rows(pos, bonds), builder.bend_rows(pos, angles)], dim=0)
        shifted = torch.cat(
            [
                builder.stretch_rows(pos + shift, bonds),
                builder.bend_rows(pos + shift, angles),
            ],
            dim=0,
        )

        per_atom_sum = rows.reshape(rows.shape[0], pos.shape[0], 3).sum(dim=1)
        torch.testing.assert_close(
            per_atom_sum, torch.zeros_like(per_atom_sum), atol=EXACT_ATOL, rtol=0.0
        )
        torch.testing.assert_close(shifted, rows, atol=EXACT_ATOL, rtol=0.0)

    def test_rotation_covariance(self) -> None:
        """Under ``x → Rx`` every per-atom s-vector maps ``s → R s`` (R ∈ SO(3))."""
        pos = _pos(WATER_POS)
        rotation = _rotation_matrix()
        builder = WilsonBMatrix()
        bonds, angles = _index(WATER_BONDS), _index(WATER_ANGLE)

        rows = torch.cat([builder.stretch_rows(pos, bonds), builder.bend_rows(pos, angles)], dim=0)
        rotated_pos = pos @ rotation.T
        rotated = torch.cat(
            [
                builder.stretch_rows(rotated_pos, bonds),
                builder.bend_rows(rotated_pos, angles),
            ],
            dim=0,
        )

        for row in range(rows.shape[0]):
            for atom in range(pos.shape[0]):
                torch.testing.assert_close(
                    _atom_block(rotated, row, atom),
                    rotation @ _atom_block(rows, row, atom),
                    atol=ROTATION_ATOL,
                    rtol=0.0,
                )

    def test_gf_consistency_diatomic(self) -> None:
        """``H_x = 2·BᵀB`` projects back to ``diag(H_q) = [2.0]`` for one coordinate."""
        pos = _pos([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        b_matrix = WilsonBMatrix().stretch_rows(pos, _index([[0], [1]]))

        cartesian_hessian = GF_FORCE_CONSTANT * b_matrix.T @ b_matrix
        internal_hessian = ProjectedInternalHessianLoss.project(cartesian_hessian, b_matrix)
        assert internal_hessian.shape == (1, 1)
        torch.testing.assert_close(
            torch.diag(internal_hessian),
            torch.tensor([GF_FORCE_CONSTANT], dtype=torch.float64),
            atol=PINV_ATOL,
            rtol=0.0,
        )

    def test_empty_terms_and_dtype(self) -> None:
        """Empty COO inputs give ``(0, 3N)``; dtype and device follow ``pos``."""
        pos = _pos(WATER_POS)
        builder = WilsonBMatrix()

        empty_bonds = builder.stretch_rows(pos, torch.zeros(2, 0, dtype=torch.long))
        empty_angles = builder.bend_rows(pos, torch.zeros(3, 0, dtype=torch.long))

        assert empty_bonds.shape == (0, 9)
        assert empty_angles.shape == (0, 9)
        for rows in (empty_bonds, empty_angles):
            assert rows.dtype == pos.dtype
            assert rows.device == pos.device

        pos32 = _pos(WATER_POS, dtype=torch.float32)
        rows32 = builder.stretch_rows(pos32, _index(WATER_BONDS))
        bend32 = builder.bend_rows(pos32, _index(WATER_ANGLE))
        assert rows32.dtype == torch.float32
        assert bend32.dtype == torch.float32
        assert rows32.device == pos32.device
        torch.testing.assert_close(
            rows32.double(),
            builder.stretch_rows(pos, _index(WATER_BONDS)),
            atol=1e-6,
            rtol=0.0,
        )

    def test_bad_shapes_raise(self) -> None:
        """``(E,2)`` bond_index and a non-3-row angle_index fail loud."""
        pos = _pos(WATER_POS)
        builder = WilsonBMatrix()
        edge_index = _index([[0, 1], [1, 2], [2, 0]])  # (E, 2) neighbour graph

        with pytest.raises(ValueError, match="edge_index"):
            builder.stretch_rows(pos, edge_index)

        with pytest.raises(ValueError):
            builder.bend_rows(pos, _index([[0, 1], [1, 2]]))  # (2, n) — apex row missing

    def test_degenerate_geometry_raises(self) -> None:
        """Zero-length bond and near-linear bend raise, naming the term/row index."""
        # Zero-length bond at term index 2 (atoms 2 and 3 coincide).
        coincident = _pos([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        builder = WilsonBMatrix(sin_tol=1e-6, length_tol=1e-9)
        with pytest.raises(ValueError, match=r"\b2\b") as bond_error:
            builder.stretch_rows(coincident, _index([[0, 0, 2], [1, 2, 3]]))
        assert "bond" in str(bond_error.value).lower()

        # Near-linear bend at row index 2. sin(179.9999°) = 1.745e-6, so the
        # default sin_tol = 1e-6 would NOT reject it — the guard is exercised
        # with sin_tol = 1e-5 (spec example corrected).
        theta = math.radians(179.9999)
        near_linear = _pos(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [math.cos(theta), math.sin(theta), 0.0],
            ]
        )
        strict = WilsonBMatrix(sin_tol=1e-5)
        angle_index = _index([[0, 2, 0], [1, 1, 1], [2, 3, 3]])
        with pytest.raises(ValueError, match=r"\b2\b") as angle_error:
            strict.bend_rows(near_linear, angle_index)
        message = str(angle_error.value).lower()
        assert "angle" in message or "bend" in message or "linear" in message

        # θ = 179° is well-defined: build succeeds and still matches finite differences.
        pos179 = _near_linear_pos(179.0)
        rows179 = WilsonBMatrix().bend_rows(pos179, _index([[0], [1], [2]]))
        expected = _central_difference(_bend_angle(0, 1, 2), pos179).unsqueeze(0)
        torch.testing.assert_close(rows179, expected, atol=FD_ATOL, rtol=0.0)


class TestWilsonBMatrixGraphBlocks:
    """``WilsonBMatrix.graph_blocks`` — one independent block per packed graph."""

    def test_two_molecule_blocks(self) -> None:
        """Water + H₂ ⇒ ``(3,9)`` and ``(1,6)`` blocks equal to standalone builds."""
        water = _pos(WATER_POS)
        h2 = _pos(H2_POS)
        pos = torch.cat([water, h2], dim=0)
        atom_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        builder = WilsonBMatrix()

        blocks = builder.graph_blocks(
            pos,
            atom_batch,
            bond_index=_index([[0, 0, 3], [1, 2, 4]]),
            angle_index=_index(WATER_ANGLE),
        )

        assert len(blocks) == 2
        assert blocks[0].shape == (3, 9)
        assert blocks[1].shape == (1, 6)
        water_reference = torch.cat(
            [
                builder.stretch_rows(water, _index(WATER_BONDS)),
                builder.bend_rows(water, _index(WATER_ANGLE)),
            ],
            dim=0,
        )
        h2_reference = builder.stretch_rows(h2, _index([[0], [1]]))
        torch.testing.assert_close(blocks[0], water_reference, atol=EXACT_ATOL, rtol=0.0)
        torch.testing.assert_close(blocks[1], h2_reference, atol=EXACT_ATOL, rtol=0.0)

    def test_cross_graph_and_nonmonotonic_raise(self) -> None:
        """A term spanning two graphs and a non-monotonic ``atom_batch`` both raise."""
        pos = torch.cat([_pos(WATER_POS), _pos(H2_POS)], dim=0)
        builder = WilsonBMatrix()

        with pytest.raises(ValueError):
            builder.graph_blocks(
                pos,
                torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
                bond_index=_index([[0, 2], [1, 3]]),  # bond 1 spans graph 0 → graph 1
            )

        with pytest.raises(ValueError):
            builder.graph_blocks(
                pos,
                torch.tensor([0, 0, 1, 1, 0], dtype=torch.long),  # non-decreasing violated
                bond_index=_index([[0, 2], [1, 3]]),
            )

    def test_empty_graph_block(self) -> None:
        """A graph carrying no terms still yields a ``(0, 3N_g)`` block."""
        pos = torch.cat([_pos(WATER_POS), _pos(H2_POS)], dim=0)
        atom_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

        blocks = WilsonBMatrix().graph_blocks(
            pos,
            atom_batch,
            bond_index=_index(WATER_BONDS),
            angle_index=_index(WATER_ANGLE),
        )

        assert len(blocks) == 2
        assert blocks[0].shape == (3, 9)
        assert blocks[1].shape == (0, 6)
        assert blocks[1].dtype == pos.dtype
