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

    **Capacity, not data-dependent shape.** The batched solve pads every
    molecule to one dense ``(B, n_max)`` layout. With ``max_atoms=None`` that
    ``n_max`` is read off the batch (``counts.max().item()``), a host sync and
    a data-dependent shape that ``torch.compile`` can only graph-break on. A
    declared ``max_atoms`` fixes the layout instead — the fixed-capacity +
    mask idiom of ``molix.data.NeighborList`` — so the whole solve traces as
    one graph. A molecule over capacity fails closed twice over: eagerly it
    raises a :class:`ValueError` naming the capacity; inside a compiled region
    every charge of the batch becomes ``NaN`` — never a truncated, finite,
    wrong charge.

    Args:
        kernel: The hardness block assembler.
        jitter: Diagonal jitter added to real atoms for conditioning.
        max_atoms: Fixed per-molecule capacity of the dense solve, or ``None``
            to size it from the batch on every call (eager-only host sync).
    """

    def __init__(
        self, kernel: HardnessKernel, *, jitter: float = 1e-8, max_atoms: int | None = None
    ) -> None:
        self.kernel = kernel
        self.jitter = float(jitter)
        if max_atoms is not None and int(max_atoms) < 1:
            raise ValueError(f"max_atoms must be >= 1, got {max_atoms}")
        self.max_atoms = None if max_atoms is None else int(max_atoms)

    def solve(
        self,
        *,
        chi: torch.Tensor,
        eta: torch.Tensor,
        problem: ChargeEquilibrationProblem,
    ) -> torch.Tensor:
        """Return partial charges ``charge`` of shape ``(N,)``.

        Args:
            chi: Electronegativity ``(N,)``.
            eta: Hardness ``(N,)``, must be positive.
            problem: Batch layout and total charges.
        """
        if chi.shape != eta.shape or chi.ndim != 1:
            raise ValueError("chi and eta must be 1-D of equal length")
        if chi.shape[0] != problem.atom_batch.shape[0]:
            raise ValueError("chi/eta length must match number of atoms")

        chi = torch.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0)
        eta = torch.nan_to_num(eta, nan=1.0, posinf=20.0, neginf=1.0e-3)
        dense_chi, dense_eta, dense_pos, atom_mask, counts = self._pack(chi, eta, problem)
        # A kernel that knows its own structure solves its own system. The
        # diagonal kernel has a closed form; assembling ``A`` and factorizing
        # it would be a batched LU over ``(n_max+1)²`` blocks — 22 % of device
        # time in the compiled train step, the largest single consumer
        # (profile 2026-09-16). Kernels that couple atoms keep the dense path.
        closed_form = getattr(self.kernel, "solve_bordered", None)
        if closed_form is not None:
            q_dense = closed_form(
                dense_eta=dense_eta,
                dense_chi=dense_chi,
                total_charge=problem.total_charge,
                atom_mask=atom_mask,
                jitter=self.jitter,
            )
            q_dense = self._refuse_over_capacity(q_dense, counts)
        else:
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
        live = problem.atom_mask.to(dtype=chi.dtype) if problem.atom_mask is not None else None
        return self._unpack(q_dense, problem.atom_batch, counts, live)

    def _pack(
        self,
        chi: torch.Tensor,
        eta: torch.Tensor,
        problem: ChargeEquilibrationProblem,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        batch = problem.atom_batch
        b = problem.total_charge.shape[0]
        live = (
            problem.atom_mask.to(dtype=batch.dtype)
            if problem.atom_mask is not None
            else torch.ones_like(batch)
        )
        counts = batch.new_zeros(b)
        counts.scatter_add_(0, batch, live)
        if self.max_atoms is None:
            n_max = int(counts.max().item()) if int(b) > 0 else 0
        else:
            n_max = self.max_atoms
            if not torch.compiler.is_compiling() and bool((counts > n_max).any()):
                raise ValueError(
                    "ElectronegativityEqualization: a molecule has "
                    f"{int(counts.max())} atoms, over the declared max_atoms={n_max} "
                    "capacity of the dense solve; build the solver with a larger "
                    "max_atoms (or None to size it from each batch)"
                )
        offsets = counts.cumsum(0) - counts
        # Packed live atoms of each graph are contiguous, so relative indices
        # are one vectorized subtraction. MoleculePadder ghosts sit at the
        # end with batch=0; their global index is not a per-graph slot.
        rel = torch.arange(batch.shape[0], device=batch.device) - offsets[batch]
        live_bool = live.bool()
        # ``chi * 0`` is NaN when a ghost's chi is inf (dummy valence
        # message-passing onto four ghosts). ``where`` keeps a finite 0.
        chi_live = torch.where(live_bool, chi, chi.new_zeros(()))
        eta_live = torch.where(live_bool, eta, eta.new_zeros(()))
        if n_max > 0:
            # Clamp keeps the scatter in bounds; accumulate so a ghost's 0
            # does not wipe a live slot it collides with (graph-0 last atom).
            rel = rel.clamp(min=0, max=n_max - 1)
        dense_chi = torch.zeros(b, n_max, dtype=chi.dtype, device=chi.device)
        dense_eta = torch.zeros(b, n_max, dtype=eta.dtype, device=eta.device)
        if n_max > 0:
            dense_chi.index_put_((batch, rel), chi_live, accumulate=True)
            dense_eta.index_put_((batch, rel), eta_live, accumulate=True)
        atom_mask = torch.arange(n_max, device=batch.device).unsqueeze(0) < counts.unsqueeze(1)

        dense_pos = None
        if problem.pos is not None:
            dense_pos = torch.zeros(b, n_max, 3, dtype=problem.pos.dtype, device=problem.pos.device)
            if n_max > 0:
                pos = problem.pos
                pos_live = torch.where(live_bool.unsqueeze(-1), pos, pos.new_zeros(()))
                dense_pos.index_put_((batch, rel), pos_live, accumulate=True)
        return dense_chi, dense_eta, dense_pos, atom_mask, counts

    def _refuse_over_capacity(self, q: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Poison the batch when a molecule exceeds the declared capacity.

        The same in-graph fail-closed the dense path applies, so the closed
        form cannot be the quiet way past a capacity the operator declared.
        """
        if self.max_atoms is None:
            return q
        over = (counts > q.shape[-1]).any()
        return torch.where(over, torch.full_like(q, float("nan")), q)

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
        # Padding variables are disconnected from the physical KKT system.
        # Pin them with an identity diagonal so shorter graphs remain
        # nonsingular inside the shared dense solve.
        m[:, :n, :n] += torch.diag_embed((~atom_mask).to(dtype=dtype))
        m[:, :n, n] = ones
        m[:, n, :n] = ones
        # rhs = [-χ, Q]
        rhs = torch.zeros(b, n + 1, device=device, dtype=dtype)
        rhs[:, :n] = -dense_chi * ones
        rhs[:, n] = total_charge.to(dtype=dtype)

        # For empty molecules (count=0), set the identity — a mask select, not
        # a Python branch on a tensor value, so the solve traces as one graph.
        empty = counts == 0
        eye = torch.eye(n + 1, device=device, dtype=dtype).expand(b, n + 1, n + 1)
        m = torch.where(empty.view(b, 1, 1), eye, m)
        a_ok = torch.isfinite(m).all(dim=(-2, -1), keepdim=True)
        m = torch.where(a_ok, m, eye)
        rhs = torch.nan_to_num(rhs)

        sol = torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)
        q = sol[:, :n] * ones
        if self.max_atoms is not None:
            # In-graph fail-closed: any molecule over capacity poisons the batch.
            over = (counts > n).any()
            q = torch.where(over, torch.full_like(q, float("nan")), q)
        return q

    def _unpack(
        self,
        q_dense: torch.Tensor,
        atom_batch: torch.Tensor,
        counts: torch.Tensor,
        live: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n_max = q_dense.shape
        n = atom_batch.shape[0]
        if n_max < 1:
            return q_dense.new_zeros(n)
        offsets = counts.cumsum(0) - counts
        rel = (torch.arange(n, device=atom_batch.device) - offsets[atom_batch]).clamp(
            min=0, max=n_max - 1
        )
        q = q_dense[atom_batch, rel]
        if live is not None:
            q = q * live.to(dtype=q.dtype)
        return q
