"""Diagonal hardness-only coupling (no interatomic J)."""

from __future__ import annotations

import torch

__all__ = ["DiagonalHardnessKernel"]


class DiagonalHardnessKernel:
    """``A = diag(eta)`` — no geometry-dependent Coulomb coupling.

    Single responsibility: assemble the hardness block of the EEM matrix.
    """

    def assemble(
        self,
        *,
        eta: torch.Tensor,
        dense_eta: torch.Tensor,
        dense_pos: torch.Tensor | None,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return hardness matrix ``A`` of shape ``(B, n_max, n_max)``.

        Args:
            eta: Per-atom hardness ``(N,)`` (unused when ``dense_eta`` given).
            dense_eta: Padded hardness ``(B, n_max)``.
            dense_pos: Unused for diagonal kernel.
            atom_mask: Valid atom mask ``(B, n_max)``.
        """
        del eta, dense_pos
        # diag(eta) with zeros on padded atoms
        a = torch.diag_embed(dense_eta * atom_mask.to(dtype=dense_eta.dtype))
        return a
