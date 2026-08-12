"""Ohno-style shielded Coulomb coupling for EEM/QEq hardness matrix."""

from __future__ import annotations

import torch

__all__ = ["ShieldedCoulombKernel"]


class ShieldedCoulombKernel:
    """Off-diagonal ``J_ij = 1 / sqrt(r_ij² + a_ij²)`` with ``a_ij`` from hardness.

    Uses the common Ohno form with ``a_ij = 2 / (η_i + η_j)`` (in the same
    charge units as η). Diagonal remains ``η_i``.

    Single responsibility: assemble the full A matrix for one padded batch.
    """

    def __init__(self, *, floor: float = 1e-6) -> None:
        self.floor = float(floor)

    def assemble(
        self,
        *,
        eta: torch.Tensor,
        dense_eta: torch.Tensor,
        dense_pos: torch.Tensor | None,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        if dense_pos is None:
            raise ValueError("ShieldedCoulombKernel requires positions")
        del eta
        b, n, _ = dense_pos.shape
        dtype = dense_eta.dtype
        # pairwise distances
        diff = dense_pos.unsqueeze(2) - dense_pos.unsqueeze(1)  # (B,n,n,3)
        r2 = (diff * diff).sum(dim=-1).clamp(min=0.0)
        # Ohno: a_ij = 2/(η_i+η_j)
        eta_i = dense_eta.unsqueeze(2)
        eta_j = dense_eta.unsqueeze(1)
        a_ij = 2.0 / (eta_i + eta_j).clamp(min=self.floor)
        j_mat = 1.0 / torch.sqrt(r2 + a_ij * a_ij)
        # diagonal ← η
        eye = torch.eye(n, device=dense_pos.device, dtype=dtype).expand(b, n, n)
        a = j_mat * (1.0 - eye) + torch.diag_embed(dense_eta)
        # zero padding rows/cols
        mask = atom_mask.to(dtype=dtype)
        a = a * mask.unsqueeze(1) * mask.unsqueeze(2)
        # keep pad diagonal invertible for the bordered system
        pad = (~atom_mask).to(dtype=dtype)
        a = a + torch.diag_embed(pad)
        return a
