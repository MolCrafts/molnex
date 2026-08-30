"""Fixed-shape forward for CUDA-graph-capturable AOTI inference.

A CUDA graph can only be captured over a region with **static shapes** and no
capture-illegal ops (no host sync / dynamic alloc). MD changes the edge count
every step, so :class:`StaticForward` fixes ``N`` atoms and pads the pair list to
``E_max`` edges, taking a per-edge ``edge_mask`` as a 4th input. Padded edges are
made **inert without changing weights or physics**: their ``edge_diff`` is
overwritten to a length past the cutoff, so the model's ``cutoff(edge_dist)``
zeros their energy and force contribution (PiNet consumes the provided
``edge_diff`` via ``edge_bond_diff`` and recomputes ``edge_dist`` from it).

Schema v3 adds a 5th input, ``local_mask (N,)``, and a 3rd output,
``atomic_energy (N,)``, matching :class:`molix.engine.adapter.EngineForward`.
The two masks are deliberately *not* both called ``mask``: ``edge_mask`` selects
live rows of a padded **edge** tensor, ``local_mask`` selects the **atoms** this
MPI rank owns, and they have different lengths, different meanings and adjacent
positions in the argument list — the ideal conditions for a silent mis-wiring.

Export this with ``dynamic_shapes=None`` and load the ``.pt2`` with
``run_single_threaded=True`` (PyTorch #158834, fixed in torch 2.8) — then the
deployer (C++ ``pair_style molnex``) can capture ``runner->run`` in an
``at::cuda::CUDAGraph`` and replay it each step. Measured ~2.9x on aspirin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.engine.adapter import EngineAdapter
from molix.schema import EDGE_DIST_CLAMP_MIN
from molix.units import DEAD_EDGE_CUTOFF_FACTOR


class StaticForward(nn.Module):
    """Fixed ``(N, E_max)`` flat forward, schema v3.

    ``(Z, pos, edge_index, edge_mask, local_mask) -> (energy, forces, atomic_energy)``

    The fixed-shape counterpart of :class:`molix.engine.adapter.EngineForward`
    and partitioned identically: ``energy = (atomic_energy * local_mask).sum()``
    and ``forces = -∂energy/∂pos`` over *all* sites, so masked-out ghosts keep
    their cross-term forces (spec ``lammps-iface-05-pbc`` D1/D2 — see
    ``EngineForward`` for the full argument). The model's own scalar energy and
    forces are read only through
    :meth:`~molix.engine.adapter.EngineAdapter.read_energy_triple`, never
    returned as the energy and forces, because they answer the unmasked
    question.

    Args:
        model: A molnex potential taking a nested ``TensorDict`` and returning
            ``{energy, atomic_energy}``, optionally with ``forces`` (e.g.
            ``PiNetPotential(compute_forces=False)`` — the preferred
            energy-only shape), in either of the two spellings
            :meth:`~molix.engine.adapter.EngineAdapter.read_energy_triple`
            accepts.
        n_atoms: Fixed atom count ``N`` baked into the graph.
        e_max: Fixed padded edge count. The deployer must pass exactly ``e_max``
            edges (real ones first, ``edge_mask=True``; the rest padded,
            ``edge_mask=False``) and fail loudly when the real edge count
            exceeds ``e_max``.
        cutoff: Neighbour cutoff in Å; padded edges get ``edge_diff`` of length
            ``DEAD_EDGE_CUTOFF_FACTOR * cutoff`` (see :mod:`molix.units`) so
            ``cutoff(edge_dist)`` zeros them.
    """

    def __init__(self, model: nn.Module, n_atoms: int, e_max: int, cutoff: float) -> None:
        super().__init__()
        self.model = model
        self.n_atoms = int(n_atoms)
        self.e_max = int(e_max)
        self.pad_len = float(cutoff) * DEAD_EDGE_CUTOFF_FACTOR

    def model_outputs(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Build the padded batch, run the model, read its unmasked v3 triple.

        Same seam as :meth:`molix.engine.adapter.EngineForward.model_outputs`,
        with the extra ``edge_mask`` this path needs to neutralise padding:
        :func:`molix.engine.export_for_lammps` runs its eager pre-trace checks
        through it on either path.
        """
        src = edge_index[:, 0]
        tgt = edge_index[:, 1]
        edge_diff = pos[tgt] - pos[src]
        # padded edges (edge_mask False) -> length past cutoff -> inert
        edge_diff = torch.where(
            edge_mask.unsqueeze(-1), edge_diff, torch.full_like(edge_diff, self.pad_len)
        )
        edge_dist = edge_diff.norm(dim=-1).clamp(min=EDGE_DIST_CLAMP_MIN)
        td = TensorDict(
            atoms=TensorDict(
                Z=Z,
                pos=pos,
                batch=torch.zeros(self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[self.n_atoms],
            ),
            edges=TensorDict(
                edge_index=edge_index,
                edge_diff=edge_diff,
                edge_dist=edge_dist,
                batch_size=[self.e_max],
            ),
            graphs=TensorDict(
                num_atoms=torch.full((1,), self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[1],
            ),
            batch_size=[],
        )
        # Monomorphic forward since b85d12f: a potential that derives forces was
        # constructed that way (``PiNetPotential(compute_forces=True)``).
        return EngineAdapter.read_energy_triple(self.model(td))

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``Z (N,)`` int64, ``pos (N,3)``, ``edge_index (E_max,2)`` int64,
        ``edge_mask (E_max,)`` bool, ``local_mask (N,)`` bool. Returns
        ``(energy (), forces (N,3), atomic_energy (N,))``."""
        pos = pos.detach().requires_grad_(True)
        with torch.enable_grad():
            _, _, atomic_energy = self.model_outputs(Z, pos, edge_index, edge_mask)
            # Mask-multiply, never a slice: `n_local` is a runtime scalar and
            # torch.export cannot trace a shape that depends on one.
            energy = (atomic_energy * local_mask.to(atomic_energy.dtype)).sum()
            (grad,) = torch.autograd.grad(energy, pos)
        return energy.detach().reshape(()), -grad, atomic_energy.detach()
