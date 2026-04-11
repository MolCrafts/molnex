"""Induced-dipole polarization potential."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class Polarization(nn.Module):
    """Self-consistent induced-dipole polarization energy.

    Computes the polarization energy by:
      1. Computing the permanent electric field from partial charges.
      2. Building a sparse dipole-dipole interaction tensor T from edge_index.
      3. Solving ``(alpha_inv - T) @ mu = E_perm`` per molecule.
      4. ``U_pol = -0.5 * sum(mu . E_perm)`` per molecule.

    Args:
        damping_factor: Thole-style damping factor for the dipole-dipole tensor.
    """

    def __init__(self, damping_factor: float = 0.39):
        super().__init__()
        self.damping_factor = damping_factor

    def forward(
        self,
        *,
        pos: torch.Tensor,
        charge: torch.Tensor,
        alpha: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute induced-dipole polarization energy.

        Args:
            pos: Atom positions ``(N, 3)``.
            charge: Partial charges ``(N,)``.
            alpha: Isotropic polarizabilities ``(N,)``.
            batch: Graph index per atom ``(N,)``.
            edge_index: Neighbor pairs ``(E, 2)``.
            num_graphs: Number of graphs.

        Returns:
            Per-graph polarization energy ``(B,)`` or scalar.
        """
        N = pos.shape[0]
        device = pos.device
        dtype = pos.dtype

        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        # 1. Permanent electric field at each atom from charges
        src, dst = edge_index[:, 0], edge_index[:, 1]
        r_vec = pos[dst] - pos[src]  # (E, 3)
        r_norm = r_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (E, 1)
        r_hat = r_vec / r_norm  # (E, 3)

        # E_field from Coulomb: E_i = sum_j q_j * r_hat_ij / r_ij^2
        field_contrib = charge[dst].unsqueeze(-1) * r_hat / r_norm  # (E, 3)
        E_perm = torch.zeros(N, 3, dtype=dtype, device=device)
        E_perm.index_add_(0, src, field_contrib)

        # 2. Build dipole-dipole interaction tensor T (sparse, 3x3 blocks)
        # T_ij = (3 * r_hat r_hat^T - I) / r^3, with Thole damping
        r3 = r_norm.squeeze(-1).pow(3)  # (E,)

        # Thole damping: lambda(u) = 1 - (1 + u + 0.5*u^2) * exp(-u)
        # where u = damping_factor * r / (alpha_i * alpha_j)^(1/6)
        alpha_src = alpha[src].clamp(min=1e-12)
        alpha_dst = alpha[dst].clamp(min=1e-12)
        alpha_prod_sixth = (alpha_src * alpha_dst).pow(1.0 / 6.0)
        u = self.damping_factor * r_norm.squeeze(-1) / alpha_prod_sixth.clamp(min=1e-12)
        damping = 1.0 - (1.0 + u + 0.5 * u.pow(2)) * torch.exp(-u)

        # T contribution to the matrix-vector product:
        # We don't build the full NxN 3x3 matrix. Instead we'll build
        # the operator implicitly for the linear solve.
        # T_ij @ v = damping_ij / r_ij^3 * (3*(r_hat_ij . v)*r_hat_ij - v)
        damped_inv_r3 = damping / r3.clamp(min=1e-12)  # (E,)

        def apply_T(v: torch.Tensor) -> torch.Tensor:
            """Apply dipole-dipole operator T @ v, where v is (N, 3)."""
            v_dst = v[dst]  # (E, 3)
            dot = (r_hat * v_dst).sum(dim=-1, keepdim=True)  # (E, 1)
            t_contrib = damped_inv_r3.unsqueeze(-1) * (3.0 * dot * r_hat - v_dst)  # (E, 3)
            result = torch.zeros_like(v)
            result.index_add_(0, src, t_contrib)
            return result

        def apply_A(v: torch.Tensor) -> torch.Tensor:
            """Apply (alpha_inv - T) @ v."""
            alpha_inv = 1.0 / alpha.clamp(min=1e-12)
            return alpha_inv.unsqueeze(-1) * v - apply_T(v)

        # 3. Solve (alpha_inv - T) @ mu = E_perm using conjugate gradient
        mu = self._cg_solve(apply_A, E_perm, max_iter=50, tol=1e-6)

        # 4. U_pol = -0.5 * sum(mu . E_perm) per molecule
        per_atom_energy = -0.5 * (mu * E_perm).sum(dim=-1)  # (N,)

        energy = torch.zeros(num_graphs, dtype=dtype, device=device)
        energy.index_add_(0, batch, per_atom_energy)
        return energy

    @staticmethod
    def _cg_solve(
        matvec: Callable,
        b: torch.Tensor,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Conjugate gradient solver for A @ x = b.

        Args:
            matvec: Function computing A @ x.
            b: Right-hand side ``(N, 3)``.
            max_iter: Maximum iterations.
            tol: Convergence tolerance on relative residual norm.

        Returns:
            Solution x ``(N, 3)``.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = (r * r).sum()

        b_norm = b.norm()
        if b_norm < 1e-12:
            return x

        for _ in range(max_iter):
            Ap = matvec(p)
            pAp = (p * Ap).sum()
            if pAp.abs() < 1e-12:
                break
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() / b_norm < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x
