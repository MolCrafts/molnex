"""Class-I force constants from a QM Hessian by Seminario's method.

The job: given a molecule's Cartesian Hessian ``H`` at a stationary geometry,
recover the harmonic force constants a Class-I force field uses — one ``k`` per
bond and one per angle, under the convention ``E = sum 1/2 k (q - q0)^2``.

**Why not the Wilson projection.** ``H_q = (B^+)^T H B^+``
(:meth:`molix.core.losses.ProjectedInternalHessianLoss.project`) is exact for a
*non-redundant* internal-coordinate set, and its tests hold on exact two- and
three-atom goldens. Real molecules are redundant: 38 bonds and 59 angles on a
30-atom THEMol record is 97 coordinates for ``3N - 6 = 84`` degrees of freedom,
``cond(B)`` is then about ``2e16`` — singular to machine precision — and the
pseudo-inverse amplifies accordingly. Measured on 40 records of
``themol/hessian@1`` (2026-09-20): angle constants with a median of 1789 and a
maximum of ``2e12`` kcal mol⁻¹ rad⁻², against a physical range of roughly
35 (H–C–H) to 160 (C–C–C). Restricting the projection to bends alone does not
help, and the smallest molecule in the shard's first 400 records is still
redundant by one coordinate. So the Wilson route is kept for what it documents
and this module is what stage-1 fitting uses.

**Seminario's method is local.** Each constant comes from one 3x3 *interaction
block* of the Hessian — the mixed second derivatives of the two atoms involved —
and its eigen-decomposition. No matrix inverse, nothing global, so redundancy
cannot degrade it.

For a bond ``A–B`` with unit vector ``u_AB``::

    k_AB = sum_a  lambda_a  |u_AB . v_a|

over the three eigenpairs ``(lambda_a, v_a)`` of ``-H[A, B]``. For an angle
``A–B–C`` about apex ``B``, the two arms act as springs in series about the
normal ``n = u_CB x u_AB``::

    1/k_theta = 1 / (R_AB^2 sum_a lambda_a^{AB} |u_PA . v_a^{AB}|)
              + 1 / (R_CB^2 sum_a lambda_a^{CB} |u_PC . v_a^{CB}|)

with ``u_PA = n x u_AB`` and ``u_PC = u_CB x n`` the in-plane perpendiculars.

Units follow the Hessian: kcal mol⁻¹ Å⁻² in gives kcal mol⁻¹ Å⁻² for bonds and,
through the ``R^2`` factors, kcal mol⁻¹ rad⁻² for angles.

An interaction block is **not symmetric** in general — the full Hessian is, so
``H[A, B] = H[B, A]^T``, but a single off-diagonal block need not be — hence
:func:`torch.linalg.eig` and not ``eigh``, with the real part taken. Eigenvalue
magnitudes are used: the block of a bonded pair is negative-definite up to sign
conventions, and the method is written for restoring curvature.

Reference:
    Seminario, J. M. "Calculation of intramolecular force fields from
    second-derivative tensors." *Int. J. Quantum Chem.* **60**, 1271-1277 (1996).
    https://doi.org/10.1002/(SICI)1097-461X(1996)60:7<1271::AID-QUA8>3.0.CO;2-W

    The symmetrised variant (Allen, Payne & Cole, *J. Chem. Theory Comput.* 14,
    274, 2018) averages a term over every angle its atoms share; this module
    implements the original per-term form, which is what a per-record label
    needs.
"""

from __future__ import annotations

import torch

__all__ = ["SeminarioForceConstants"]


class SeminarioForceConstants:
    """Bond and angle force constants from a Cartesian Hessian.

    Stateless and constructed with no arguments; one instance is reusable.

    Example::

        seminario = SeminarioForceConstants()
        k_bond = seminario.bond_constants(pos, hessian, bond_index)
        k_angle = seminario.angle_constants(pos, hessian, angle_index)
    """

    @staticmethod
    def _blocks(hessian: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """The ``(n, 3, 3)`` interaction blocks ``-H[first, second]``.

        Args:
            hessian: ``(3N, 3N)`` Cartesian Hessian, flat index ``3a + c``.
            first: ``(n,)`` atom indices.
            second: ``(n,)`` atom indices.

        Returns:
            One negated 3x3 mixed-derivative block per pair.
        """
        rows = (3 * first).unsqueeze(1) + torch.arange(3, device=hessian.device)
        cols = (3 * second).unsqueeze(1) + torch.arange(3, device=hessian.device)
        return -hessian[rows.unsqueeze(2), cols.unsqueeze(1)]

    @staticmethod
    def _projected(blocks: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """``sum_a lambda_a |direction . v_a|`` for each block.

        Args:
            blocks: ``(n, 3, 3)`` interaction blocks.
            direction: ``(n, 3)`` unit vectors to project onto.

        Returns:
            ``(n,)`` projected curvature, the scalar Seminario sum.
        """
        values, vectors = torch.linalg.eig(blocks)
        # A single off-diagonal block need not be symmetric, so the spectrum can
        # be complex; the physical curvature is its real magnitude.
        lam = values.real.abs()
        # vectors[..., :, a] is eigenvector a; project the direction onto each.
        overlap = torch.einsum("ni,nia->na", direction, vectors.real).abs()
        return (lam * overlap).sum(dim=-1)

    def bond_constants(
        self, pos: torch.Tensor, hessian: torch.Tensor, bond_index: torch.Tensor
    ) -> torch.Tensor:
        """``(n_bonds,)`` stretch force constants, kcal mol⁻¹ Å⁻².

        Args:
            pos: ``(N, 3)`` positions, Å.
            hessian: ``(3N, 3N)`` Cartesian Hessian, kcal mol⁻¹ Å⁻².
            bond_index: ``(2, n_bonds)`` COO atom indices.

        Returns:
            One force constant per bond, under ``E = 1/2 k (r - r0)^2``.

        Raises:
            ValueError: *bond_index* is not ``(2, n_bonds)``.
        """
        if bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise ValueError(f"bond_index must be (2, n_bonds); got {tuple(bond_index.shape)}")
        if bond_index.shape[1] == 0:
            return pos.new_zeros(0)
        a, b = bond_index[0], bond_index[1]
        delta = pos[b] - pos[a]
        unit = delta / delta.norm(dim=-1, keepdim=True)
        return self._projected(self._blocks(hessian, a, b), unit)

    def angle_constants(
        self, pos: torch.Tensor, hessian: torch.Tensor, angle_index: torch.Tensor
    ) -> torch.Tensor:
        """``(n_angles,)`` bend force constants, kcal mol⁻¹ rad⁻².

        The two arms are springs in series about the plane normal, so their
        reciprocal contributions add — an angle is only as stiff as the softer
        arm allows.

        Args:
            pos: ``(N, 3)`` positions, Å.
            hessian: ``(3N, 3N)`` Cartesian Hessian, kcal mol⁻¹ Å⁻².
            angle_index: ``(3, n_angles)`` COO atom indices; row 1 is the apex,
                matching the ``angles`` batch namespace (``atomi``, ``atomj``,
                ``atomk`` with ``atomj`` the centre).

        Returns:
            One force constant per angle, under ``E = 1/2 k (theta - theta0)^2``.

        Raises:
            ValueError: *angle_index* is not ``(3, n_angles)``.
        """
        if angle_index.ndim != 2 or angle_index.shape[0] != 3:
            raise ValueError(f"angle_index must be (3, n_angles); got {tuple(angle_index.shape)}")
        if angle_index.shape[1] == 0:
            return pos.new_zeros(0)
        i, j, k = angle_index[0], angle_index[1], angle_index[2]
        v_ab, v_cb = pos[i] - pos[j], pos[k] - pos[j]
        r_ab = v_ab.norm(dim=-1, keepdim=True)
        r_cb = v_cb.norm(dim=-1, keepdim=True)
        u_ab, u_cb = v_ab / r_ab, v_cb / r_cb
        normal = torch.cross(u_cb, u_ab, dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(pos.dtype).eps)
        u_pa = torch.cross(normal, u_ab, dim=-1)
        u_pc = torch.cross(u_cb, normal, dim=-1)
        term_a = r_ab.squeeze(-1) ** 2 * self._projected(self._blocks(hessian, i, j), u_pa)
        term_c = r_cb.squeeze(-1) ** 2 * self._projected(self._blocks(hessian, k, j), u_pc)
        eps = torch.finfo(pos.dtype).eps
        return 1.0 / (1.0 / term_a.clamp_min(eps) + 1.0 / term_c.clamp_min(eps))
