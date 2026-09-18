"""Fixed-shape forward for CUDA-graph-capturable AOTI inference.

A CUDA graph can only be captured over a region with **static shapes** and no
capture-illegal ops (no host sync / dynamic alloc). MD changes the edge count
every step, so :class:`StaticForward` fixes ``N`` atoms and pads the pair list to
``E_max`` edges, taking a per-edge ``pair_mask`` as a 4th input. Padded edges are
made **inert without changing weights or physics**: their ``pair_diff`` is
overwritten with the schema's dead-edge vector (:func:`molix.schema.dead_pair_shift`,
the same padding the MD neighbour list and ``MoleculePadder`` write), so the
model's ``cutoff(pair_dist)`` zeros their energy and force contribution (PiNet
consumes the provided ``pair_diff`` via ``edge_bond_diff`` and recomputes
``pair_dist`` from it).

Schema v3 adds a 5th input, ``local_mask (N,)``, and a 3rd output,
``atomic_energy (N,)``, matching :class:`molix.engine.adapter.EngineForward`.
The two masks are deliberately *not* both called ``mask``: ``pair_mask`` selects
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

from molix.autograd import leaf
from molix.engine.adapter import EngineAdapter
from molix.schema import PAIR_DIST_CLAMP_MIN, dead_pair_shift


class StaticForward(nn.Module):
    """Fixed ``(N, E_max)`` flat forward, schema v3.

    ``(Z, pos, pair_index, pair_mask, local_mask) -> (energy, forces, atomic_energy)``

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
            ``PiNetPotential()`` with the default ``wrt=()`` — the preferred
            energy-only shape), in either of the two spellings
            :meth:`~molix.engine.adapter.EngineAdapter.read_energy_triple`
            accepts.
        n_atoms: Fixed atom count ``N`` baked into the graph.
        e_max: Fixed padded edge count. The deployer must pass exactly ``e_max``
            edges (real ones first, ``pair_mask=True``; the rest padded,
            ``pair_mask=False``) and fail loudly when the real edge count
            exceeds ``e_max``.
        cutoff: Neighbour cutoff in Å; padded edges get the schema's dead-edge
            ``pair_diff`` (:func:`molix.schema.dead_pair_shift`, length
            ``dead_pair_length(cutoff)``) so ``cutoff(pair_dist)`` zeros them.
    """

    def __init__(self, model: nn.Module, n_atoms: int, e_max: int, cutoff: float) -> None:
        super().__init__()
        self.model = model
        self.n_atoms = int(n_atoms)
        self.e_max = int(e_max)
        self.cutoff = float(cutoff)
        # A buffer, not a per-call ``torch.tensor``: it travels with ``.to()``
        # and is a graph constant under export / compile, never a host copy.
        self.dead_shift: torch.Tensor
        self.register_buffer("dead_shift", dead_pair_shift(self.cutoff))

    def model_outputs(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Build the padded batch, run the model, read its unmasked v3 triple.

        Same seam as :meth:`molix.engine.adapter.EngineForward.model_outputs`,
        with the extra ``pair_mask`` this path needs to neutralise padding:
        :func:`molix.engine.export_for_lammps` runs its eager pre-trace checks
        through it on either path.
        """
        src = pair_index[:, 0]
        tgt = pair_index[:, 1]
        pair_diff = pos[tgt] - pos[src]
        # padded edges (pair_mask False) -> the dead-edge vector -> inert
        dead = self.dead_shift.to(dtype=pair_diff.dtype)
        pair_diff = torch.where(pair_mask.unsqueeze(-1), pair_diff, dead)
        pair_dist = pair_diff.norm(dim=-1).clamp(min=PAIR_DIST_CLAMP_MIN)
        td = TensorDict(
            atoms=TensorDict(
                Z=Z,
                pos=pos,
                batch=torch.zeros(self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[self.n_atoms],
            ),
            pairs=TensorDict(
                pair_index=pair_index,
                pair_diff=pair_diff,
                pair_dist=pair_dist,
                batch_size=[self.e_max],
            ),
            graphs=TensorDict(
                num_atoms=torch.full((1,), self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[1],
            ),
            batch_size=[],
        )
        # Monomorphic forward: a potential that derives forces was constructed
        # that way (``PiNetPotential(wrt=("atoms.pos",))``).
        return EngineAdapter.read_energy_triple(self.model(td))

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        pair_mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``Z (N,)`` int64, ``pos (N,3)``, ``pair_index (E_max,2)`` int64,
        ``pair_mask (E_max,)`` bool, ``local_mask (N,)`` bool. Returns
        ``(energy (), forces (N,3), atomic_energy (N,))``."""
        pos = leaf(pos)
        with torch.enable_grad():
            _, _, atomic_energy = self.model_outputs(Z, pos, pair_index, pair_mask)
            # Mask-multiply, never a slice: `n_local` is a runtime scalar and
            # torch.export cannot trace a shape that depends on one.
            energy = (atomic_energy * local_mask.to(atomic_energy.dtype)).sum()
            (grad,) = torch.autograd.grad(energy, pos)
        return energy.detach().reshape(()), -grad, atomic_energy.detach()
