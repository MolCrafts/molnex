"""Shared toy potential / batch fixtures for the derivation unit tests.

The toy is the analytic quadratic potential used throughout this directory::

    E_g = s · Σ_{i ∈ g} ‖r_i‖²        (eV, positions in Å)
    F_i = -∂E/∂r_i = -2 s · r_i       (eV/Å)

Both derivation backends must reproduce ``-2 s r`` exactly, so it pins the
kernels without any reference implementation. ``n_forward`` counts energy-core
invocations — that is how the "one forward per force pass" contract is checked.

``test_readouts.py`` carries its own float32 copy of the same toy (predating
this conftest); it is intentionally left untouched, so the names here are
distinct (``KernelToyPotential`` / ``kernel_toy_batch``) and default to float64
for the 1e-12 kernel assertions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation.protocol import write_energy


class KernelToyPotential(nn.Module):
    """``E = s · Σ‖pos‖²`` written onto the batch; counts energy-core calls.

    Args:
        scale: Initial value of the single parameter ``s`` (eV/Å²).
        write_atomic: Also write per-atom energies at ``("atoms", "energy")``.
            ``False`` exercises the "no atomic energy" branch of the kernels.
        write_energy_key: Write ``("graphs", "energy")`` at all. ``False``
            makes the core violate the contract, which the kernels must
            reject with a ``RuntimeError`` naming ``graphs.energy``.
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        write_atomic: bool = True,
        write_energy_key: bool = True,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        self.write_atomic = write_atomic
        self.write_energy_key = write_energy_key
        self.n_forward = 0

    def forward(self, batch: TensorDict) -> TensorDict:
        """Write ``graphs.energy`` (eV) for ``batch`` and return it in place."""
        self.n_forward += 1
        if not self.write_energy_key:
            return batch
        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        atom_energy = self.scale.to(pos.dtype) * (pos * pos).sum(dim=-1)
        n_graphs = int(batch["graphs"].batch_size[0])
        energy = torch.zeros(n_graphs, dtype=pos.dtype, device=pos.device)
        energy = energy.index_add(0, atom_batch, atom_energy)
        write_energy(
            batch,
            energy,
            atomic_energy=atom_energy if self.write_atomic else None,
        )
        return batch

    def total_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Scalar total energy (eV) for raw positions ``(N, 3)`` — finite-difference oracle."""
        return self.scale.to(pos.dtype) * (pos * pos).sum()


def kernel_toy_batch(
    n_atoms: int = 4,
    n_graphs: int = 2,
    *,
    dtype: torch.dtype = torch.float64,
) -> TensorDict:
    """Post-collate batch with ``pos = arange(3 N).reshape(N, 3) * 0.1`` Å."""
    per_graph = n_atoms // n_graphs
    pos = torch.arange(n_atoms * 3, dtype=dtype).reshape(n_atoms, 3) * 0.1
    return TensorDict(
        atoms=TensorDict(
            pos=pos,
            batch=torch.arange(n_graphs).repeat_interleave(per_graph),
            batch_size=[n_atoms],
        ),
        graphs=TensorDict(batch_size=[n_graphs]),
        batch_size=[],
    )


def central_difference_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
    *,
    h: float = 1e-5,
) -> torch.Tensor:
    """``F = -dE/dr`` by central differences — the domain oracle for the kernels.

    Args:
        energy_fn: Total energy (eV) of a position configuration ``(N, 3)`` Å.
            Must be free of autograd side effects; called ``6 N`` times.
        pos: Positions ``(N, 3)`` in Å (values only; gradients are not used).
        h: Displacement in Å. ``1e-5`` keeps float64 round-off near 1e-10 eV/Å
            while the ``O(h²)`` truncation term stays far below 1e-6 eV/Å.

    Returns:
        Numerical forces ``(N, 3)`` in eV/Å.
    """
    base = pos.detach().clone()
    forces = torch.zeros_like(base)
    with torch.no_grad():
        for atom in range(base.shape[0]):
            for axis in range(base.shape[1]):
                plus = base.clone()
                plus[atom, axis] += h
                minus = base.clone()
                minus[atom, axis] -= h
                forces[atom, axis] = -(energy_fn(plus) - energy_fn(minus)) / (2.0 * h)
    return forces
