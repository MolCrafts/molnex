"""Public-API regression for the Wilson B-matrix builder and its consumer.

Spec: ``universal-class-i-07-wilson``.

``B = ∂q/∂x`` is ``(n_q, 3N)`` with flat Cartesian index ``i = 3a + c``
(``x0,y0,z0,x1,…``); rows are ordered *all bonds, then all angles*. Stretch
rows are dimensionless (direction cosines, Å/Å); bend rows carry rad·Å⁻¹ ≡ Å⁻¹.

All goldens below are **analytic** Wilson/Decius/Cross s-vectors, hand-derived
on 2026-08-14 with no tool and no third-party oracle imported, subprocessed or
consulted at runtime::

    ∂r/∂x_i = −û,   ∂r/∂x_j = +û,            û = (x_j − x_i)/r_ij
    ∂θ/∂x_i = (cos θ · u − v)/(r_ij sin θ),  u = (x_i − x_j)/r_ij
    ∂θ/∂x_k = (cos θ · v − u)/(r_kj sin θ),  v = (x_k − x_j)/r_kj
    ∂θ/∂x_j = −(∂θ/∂x_i + ∂θ/∂x_k)

    dimer  x = (0,0,0), (1,0,0) Å; bond (0,1)
        û = (1,0,0)                    ⇒ B = [[−1,0,0, 1,0,0]]
    trimer x = (1,0,0), (0,0,0), (0,1,0) Å; angle (0,1,2), apex atom 1
        u = (1,0,0), v = (0,1,0), cos θ = 0, sin θ = 1, r_ij = r_kj = 1
        s_i = −v = (0,−1,0), s_k = −u = (−1,0,0), s_j = (1,1,0)
                                       ⇒ B = [[0,−1,0, 1,1,0, −1,0,0]]
    GF identity  H_x = k BᵀB with k = 2.0 ⇒ H_q = project(H_x, B) = [[2.0]]

Packed batch (water then H₂, ``atom_batch = [0,0,0,1,1]``): blocks are
``(3, 9)`` and ``(1, 6)`` and equal the standalone single-molecule builds —
each block is only ``3 N_g`` wide, so cross-graph coupling is structurally
impossible.

    capture: PYTHONPATH=src python regressions/universal-class-i-07-wilson.py
    date: 2026-08-14
"""

from __future__ import annotations

import sys

import torch

from molix.core.losses import ProjectedInternalHessianLoss
from molpot import WilsonBMatrix

_DTYPE = torch.float64

#: Exactly-representable goldens (integers / ±1 direction cosines).
_EXACT_ATOL = 1e-12
#: ``torch.linalg.pinv`` round-off budget for the projected Hessian.
_PINV_ATOL = 1e-6

_DIMER_POS = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=_DTYPE)
_DIMER_BOND = torch.tensor([[0], [1]], dtype=torch.long)  # COO [2, n_bonds]
_DIMER_ROW_GOLDEN = torch.tensor([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=_DTYPE)

_TRIMER_POS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=_DTYPE)
_TRIMER_ANGLE = torch.tensor([[0], [1], [2]], dtype=torch.long)  # COO [3, n]; row 1 = apex
_TRIMER_ROW_GOLDEN = torch.tensor([[0.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0]], dtype=_DTYPE)

_GF_FORCE_CONSTANT = 2.0
_HQ_GOLDEN = torch.tensor([[2.0]], dtype=_DTYPE)

#: Water, O first then the two H (Å); H₂ at 0.74 Å, translated clear of water.
_WATER_POS = torch.tensor(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.23921, 0.92695, 0.0]], dtype=_DTYPE
)
_WATER_BONDS = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
_WATER_ANGLE = torch.tensor([[1], [0], [2]], dtype=torch.long)  # apex = O (atom 0)
_H2_POS = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.74]], dtype=_DTYPE)
_H2_BOND = torch.tensor([[0], [1]], dtype=torch.long)

#: Packed batch: water atoms 0–2 (graph 0), H₂ atoms 3–4 (graph 1).
_BATCH_ATOM_BATCH = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
_BATCH_BONDS = torch.tensor([[0, 0, 3], [1, 2, 4]], dtype=torch.long)
_BATCH_BLOCK_SHAPES = ((3, 9), (1, 6))


def main() -> int:
    builder = WilsonBMatrix()

    stretch = builder.stretch_rows(_DIMER_POS, _DIMER_BOND)
    if tuple(stretch.shape) != (1, 6):
        raise AssertionError(f"stretch rows must be (1, 6); got {tuple(stretch.shape)}")
    if not torch.allclose(stretch, _DIMER_ROW_GOLDEN, atol=_EXACT_ATOL, rtol=0.0):
        raise AssertionError(f"diatomic stretch row drift:\n{stretch}")

    bend = builder.bend_rows(_TRIMER_POS, _TRIMER_ANGLE)
    if tuple(bend.shape) != (1, 9):
        raise AssertionError(f"bend rows must be (1, 9); got {tuple(bend.shape)}")
    if not torch.allclose(bend, _TRIMER_ROW_GOLDEN, atol=_EXACT_ATOL, rtol=0.0):
        raise AssertionError(f"right-angle bend row drift:\n{bend}")

    cartesian_hessian = _GF_FORCE_CONSTANT * (stretch.T @ stretch)
    h_q = ProjectedInternalHessianLoss.project(cartesian_hessian, stretch)
    if not torch.allclose(h_q, _HQ_GOLDEN, atol=_PINV_ATOL, rtol=0.0):
        raise AssertionError(f"H_q={h_q}")

    blocks = builder.graph_blocks(
        torch.cat([_WATER_POS, _H2_POS], dim=0),
        _BATCH_ATOM_BATCH,
        bond_index=_BATCH_BONDS,
        angle_index=_WATER_ANGLE,
    )
    if len(blocks) != len(_BATCH_BLOCK_SHAPES):
        raise AssertionError(f"expected {len(_BATCH_BLOCK_SHAPES)} blocks; got {len(blocks)}")
    for graph, (block, shape) in enumerate(zip(blocks, _BATCH_BLOCK_SHAPES, strict=True)):
        if tuple(block.shape) != shape:
            raise AssertionError(f"block {graph} must be {shape}; got {tuple(block.shape)}")

    water_reference = torch.cat(
        [
            builder.stretch_rows(_WATER_POS, _WATER_BONDS),
            builder.bend_rows(_WATER_POS, _WATER_ANGLE),
        ],
        dim=0,
    )
    h2_reference = builder.stretch_rows(_H2_POS, _H2_BOND)
    for graph, (block, reference) in enumerate(
        zip(blocks, (water_reference, h2_reference), strict=True)
    ):
        if not torch.allclose(block, reference, atol=_EXACT_ATOL, rtol=0.0):
            raise AssertionError(
                f"block {graph} differs from the standalone build:\n{block}\nvs\n{reference}"
            )

    print(
        "universal-class-i-07-wilson: stretch/bend s-vector goldens, H_q = 2.0, "
        "packed blocks (3,9)/(1,6) == standalone OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
