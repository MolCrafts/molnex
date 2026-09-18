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

    @staticmethod
    def solve_bordered(
        *,
        dense_eta: torch.Tensor,
        dense_chi: torch.Tensor,
        total_charge: torch.Tensor,
        atom_mask: torch.Tensor,
        jitter: float,
    ) -> torch.Tensor:
        """Charges of the bordered EEM system, in closed form.

        With ``A = diag(η)`` the system

        ``[[diag(η), 1], [1ᵀ, 0]] [q, λ]ᵀ = [-χ, Q]ᵀ``

        needs no factorization: row *i* reads ``η_i q_i + λ = −χ_i``, so
        ``q_i = −(χ_i + λ) / η_i``, and the constraint ``Σ q = Q`` fixes

        ``λ = −(Q + Σ χ_i/η_i) / Σ 1/η_i``.

        The generic path factorizes a ``(n_max+1)²`` matrix per molecule
        instead. At the Class-I operating point that is a batched LU over
        ``513×513`` blocks for molecules averaging 55 atoms, and it was the
        single largest consumer of device time in the compiled train step —
        **22 %**, three times the next operator (profile 2026-09-16, large
        tier). This is the same solution, in O(n).

        Args:
            dense_eta: Padded hardness ``(B, n_max)``, positive on real atoms.
            dense_chi: Padded electronegativity ``(B, n_max)``.
            total_charge: Declared molecular charge ``(B,)``.
            atom_mask: True on real atoms ``(B, n_max)``.
            jitter: Diagonal jitter the solver adds to real atoms.

        Returns:
            Padded charges ``(B, n_max)``, zero on padding.
        """
        live = atom_mask.to(dtype=dense_eta.dtype)
        # Padding must not enter either sum; its reciprocal would be 1/jitter.
        eta = dense_eta + jitter
        safe_eta = torch.where(atom_mask, eta, torch.ones_like(eta))
        inv_eta = live / safe_eta
        chi = dense_chi * live

        sum_inv = inv_eta.sum(dim=-1)
        sum_chi_over_eta = (chi * inv_eta).sum(dim=-1)
        # A molecule with no real atoms has no equation to satisfy; keep the
        # denominator finite so the whole batch stays finite (mask idiom, not
        # a Python branch on a tensor value).
        empty = sum_inv == 0
        denom = torch.where(empty, torch.ones_like(sum_inv), sum_inv)
        lam = -(total_charge.to(dtype=dense_eta.dtype) + sum_chi_over_eta) / denom

        q = -(chi + lam.unsqueeze(-1)) * inv_eta
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(q), q) * live
