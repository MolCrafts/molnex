"""Electronegativity equalization (EEM / QEq-family) linear solve."""

from __future__ import annotations

from typing import Protocol

import torch

from molpot.charge.problem import ChargeEquilibrationProblem

__all__ = ["ElectronegativityEqualization", "HardnessKernel"]


class HardnessKernel(Protocol):
    def assemble(
        self,
        *,
        eta: torch.Tensor,
        dense_eta: torch.Tensor,
        dense_pos: torch.Tensor | None,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class ElectronegativityEqualization:
    """Solve the EEM/QEq KKT system for partial charges.

    For each molecule::

        [ A   1 ] [ q ]   [ -χ ]
        [ 1ᵀ  0 ] [ λ ] = [  Q ]

    with ``A`` from a :class:`HardnessKernel` (diagonal or shielded Coulomb).

    This is the algorithm class for the electronegativity-equalization /
    charge-equilibration family — not a product-branded wrapper.

    Differentiable via ``torch.linalg.solve``.
    """

    def __init__(self, kernel: HardnessKernel, *, jitter: float = 1e-8) -> None:
        self.kernel = kernel
        self.jitter = float(jitter)

    def solve(
        self,
        *,
        chi: torch.Tensor,
        eta: torch.Tensor,
        problem: ChargeEquilibrationProblem,
    ) -> torch.Tensor:
        """Return partial charges ``q`` of shape ``(N,)``.

        Args:
            chi: Electronegativity ``(N,)``.
            eta: Hardness ``(N,)``, must be positive.
            problem: Batch layout and total charges.
        """
        if chi.shape != eta.shape or chi.ndim != 1:
            raise ValueError("chi and eta must be 1-D of equal length")
        if chi.shape[0] != problem.atom_batch.shape[0]:
            raise ValueError("chi/eta length must match number of atoms")

        dense_chi, dense_eta, dense_pos, atom_mask, counts = self._pack(
            chi, eta, problem
        )
        a = self.kernel.assemble(
            eta=eta,
            dense_eta=dense_eta,
            dense_pos=dense_pos,
            atom_mask=atom_mask,
        )
        # diagonal jitter on real atoms
        bsz, n_max, _ = a.shape
        eye = torch.eye(n_max, device=a.device, dtype=a.dtype).expand(bsz, n_max, n_max)
        a = a + self.jitter * eye * atom_mask.unsqueeze(1).to(dtype=a.dtype)

        q_dense = self._solve_bordered(
            a,
            dense_chi,
            problem.total_charge,
            atom_mask,
            counts,
        )
        return self._unpack(q_dense, problem.atom_batch, counts)

    def _pack(
        self,
        chi: torch.Tensor,
        eta: torch.Tensor,
        problem: ChargeEquilibrationProblem,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        batch = problem.atom_batch
        b = problem.num_graphs
        counts = torch.zeros(b, dtype=torch.long, device=batch.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch))
        n_max = int(counts.max().item()) if b > 0 else 0
        offsets = torch.zeros(b, dtype=torch.long, device=batch.device)
        if b > 1:
            offsets[1:] = counts.cumsum(0)[:-1]
        # relative index within molecule
        # build by scanning — O(N)
        rel = torch.empty_like(batch)
        cursor = torch.zeros(b, dtype=torch.long, device=batch.device)
        for i in range(batch.shape[0]):
            g = int(batch[i].item())
            rel[i] = cursor[g]
            cursor[g] += 1

        dense_chi = torch.zeros(b, n_max, dtype=chi.dtype, device=chi.device)
        dense_eta = torch.zeros(b, n_max, dtype=eta.dtype, device=eta.device)
        dense_chi[batch, rel] = chi
        dense_eta[batch, rel] = eta
        atom_mask = torch.arange(n_max, device=batch.device).unsqueeze(0) < counts.unsqueeze(1)

        dense_pos = None
        if problem.pos is not None:
            dense_pos = torch.zeros(b, n_max, 3, dtype=problem.pos.dtype, device=problem.pos.device)
            dense_pos[batch, rel] = problem.pos
        return dense_chi, dense_eta, dense_pos, atom_mask, counts

    def _solve_bordered(
        self,
        a: torch.Tensor,
        dense_chi: torch.Tensor,
        total_charge: torch.Tensor,
        atom_mask: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Solve bordered system per graph; return dense q ``(B, n_max)``."""
        b, n, _ = a.shape
        device, dtype = a.device, a.dtype
        ones = atom_mask.to(dtype=dtype)
        # M = [[A, 1], [1^T, 0]]
        m = torch.zeros(b, n + 1, n + 1, device=device, dtype=dtype)
        m[:, :n, :n] = a
        m[:, :n, n] = ones
        m[:, n, :n] = ones
        # rhs = [-χ, Q]
        rhs = torch.zeros(b, n + 1, device=device, dtype=dtype)
        rhs[:, :n] = -dense_chi * ones
        rhs[:, n] = total_charge.to(dtype=dtype)

        # For empty molecules (count=0), set 1x1 identity
        empty = counts == 0
        if empty.any():
            m[empty] = torch.eye(n + 1, device=device, dtype=dtype)

        sol = torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)
        q = sol[:, :n] * ones
        return q

    def _unpack(
        self,
        q_dense: torch.Tensor,
        atom_batch: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        b, n_max = q_dense.shape
        rel = torch.empty_like(atom_batch)
        cursor = torch.zeros(b, dtype=torch.long, device=atom_batch.device)
        for i in range(atom_batch.shape[0]):
            g = int(atom_batch[i].item())
            rel[i] = cursor[g]
            cursor[g] += 1
        return q_dense[atom_batch, rel]
