"""Closed-form gradients of bonded potential energies.

The bonded kernels (``BondHarmonic`` / ``AngleHarmonic`` /
``ProperTorsionPeriodic`` / ``ImproperPeriodic``) have closed-form energies,
so their atomic forces ``F = -∂E/∂pos`` are closed-form too. Evaluating them
directly removes the ``pos[index]`` gather-and-scatter autograd graph and the
``create_graph`` double backward that the generic
:func:`~molpot.derivation.force.autograd_forces_from_energy` pays for those
terms.

Each force below is still an ordinary autograd function of the parameters
(``k``, ``r0``, ``theta0``, ``phase``, …), so a force loss reaches the
parameter heads with a single backward — no second derivative of the energy is
ever needed.
"""

from __future__ import annotations

import torch

from molpot.potentials.geom import GEOM_EPS

__all__ = ["angle_theta_grad", "dihedral_phi_grad"]


def angle_theta_grad(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(∂θ/∂u, ∂θ/∂v)`` of the C^∞ angle ``θ = atan2(bond_cos_sin(u, v))``.

    With ``A = u·u``, ``B = v·v``, ``C = u·v`` the regularized angle is
    ``θ = atan2(sqrt(AB − C² + ε), C)``; differentiating gives

        ∂θ/∂u = (C B u − v (AB + ε)) / (sqrt(AB − C² + ε) · (AB + ε))
        ∂θ/∂v = (C A v − u (AB + ε)) / (sqrt(AB − C² + ε) · (AB + ε))

    exactly the gradient of the angle in
    :func:`~molpot.potentials.geom.bond_cos_sin` (no separate acos branch).

    Args:
        u: ``(M, 3)`` Å, first bond vector from the central atom.
        v: ``(M, 3)`` Å, second bond vector from the central atom.

    Returns:
        ``(∂θ/∂u, ∂θ/∂v)``, each ``(M, 3)``.
    """
    uu = (u * u).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    denom2 = uu * vv + GEOM_EPS
    # Squared sine via the cross-product norm ``|u × v|²`` (Lagrange identity),
    # matching :func:`~molpot.potentials.geom.bond_cos_sin`. The algebraic form
    # ``uu*vv - uv*uv`` cancels catastrophically for near-linear angles in
    # float32 and can go negative; the cross form is a sum of squares.
    cr = torch.linalg.cross(u, v)
    cc = (cr * cr).sum(dim=-1)
    sqrt_cc = (cc + GEOM_EPS).sqrt()
    inv = (sqrt_cc * denom2).reciprocal()
    du = (uv.unsqueeze(-1) * vv.unsqueeze(-1) * u - v * denom2.unsqueeze(-1)) * inv.unsqueeze(-1)
    dv = (uv.unsqueeze(-1) * uu.unsqueeze(-1) * v - u * denom2.unsqueeze(-1)) * inv.unsqueeze(-1)
    return du, dv


def dihedral_phi_grad(
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(∂φ/∂r_i, ∂φ/∂r_j, ∂φ/∂r_k, ∂φ/∂r_l)`` — Blondel–Karplus gradient.

    ``b1 = r_j − r_i``, ``b2 = r_k − r_j``, ``b3 = r_l − r_k`` with the same
    sign convention as :func:`~molpot.potentials.geom.dihedral_cos_sin`
    (``sin φ = (n1 × n2) · b2``). The four partials are

        ∂φ/∂r_i = −(|b2| / |n1|²) n1
        ∂φ/∂r_l = +(|b2| / |n2|²) n2
        ∂φ/∂r_j = −(1 + b1·b2/|b2|²) ∂φ/∂r_i + (b2·b3/|b2|²) ∂φ/∂r_l
        ∂φ/∂r_k = (b1·b2/|b2|²) ∂φ/∂r_i − (1 + b2·b3/|b2|²) ∂φ/∂r_l

    where ``n1 = b1 × b2``, ``n2 = b2 × b3``. Squared norms carry the same
    :data:`GEOM_EPS` softening as :func:`~molpot.potentials.geom.dihedral_cos_sin`,
    so a collinear torsion (``n1`` or ``n2`` → 0) yields a finite gradient.

    Args:
        b1: ``(M, 3)`` Å.
        b2: ``(M, 3)`` Å.
        b3: ``(M, 3)`` Å.

    Returns:
        ``(∂φ/∂r_i, ∂φ/∂r_j, ∂φ/∂r_k, ∂φ/∂r_l)``, each ``(M, 3)``.
    """
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1sq = (n1 * n1).sum(dim=-1) + GEOM_EPS
    n2sq = (n2 * n2).sum(dim=-1) + GEOM_EPS
    b2sq = (b2 * b2).sum(dim=-1) + GEOM_EPS
    b1b2 = (b1 * b2).sum(dim=-1)
    b2b3 = (b2 * b3).sum(dim=-1)
    b2len = b2sq.sqrt()

    fi = -b2len.unsqueeze(-1) / n1sq.unsqueeze(-1) * n1
    fl = b2len.unsqueeze(-1) / n2sq.unsqueeze(-1) * n2
    c1 = b1b2 / b2sq
    c2 = b2b3 / b2sq
    fj = -(1.0 + c1).unsqueeze(-1) * fi + c2.unsqueeze(-1) * fl
    fk = c1.unsqueeze(-1) * fi - (1.0 + c2).unsqueeze(-1) * fl
    return fi, fj, fk, fl
