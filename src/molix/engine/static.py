"""Fixed-shape forward for CUDA-graph-capturable AOTI inference.

A CUDA graph can only be captured over a region with **static shapes** and no
capture-illegal ops (no host sync / dynamic alloc). MD changes the edge count
every step, so :class:`StaticForward` fixes ``N`` atoms and pads the pair list to
``E_max`` edges, taking a per-edge ``mask`` as a 4th input. Padded edges are made
**inert without changing weights or physics**: their ``edge_diff`` is overwritten
to a length past the cutoff, so the model's ``cutoff(edge_dist)`` zeros their
energy and force contribution (PiNet consumes the provided ``edge_diff`` via
``_edge_bond_diff`` and recomputes ``edge_dist`` from it).

Export this with ``dynamic_shapes=None`` and load the ``.pt2`` with
``run_single_threaded=True`` (PyTorch #158834, fixed in torch 2.8) — then the
deployer (C++ ``pair_style molnex``) can capture ``runner->run`` in an
``at::cuda::CUDAGraph`` and replay it each step. Measured ~2.9x on aspirin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class StaticForward(nn.Module):
    """Fixed ``(N, E_max)`` flat forward: ``(Z, pos, edge_index, mask) -> (energy, forces)``.

    Args:
        model: A molnex potential taking a nested ``TensorDict`` and returning
            ``{"energy", "forces"}`` (e.g. ``PiNetPotential(compute_forces=True)``).
        n_atoms: Fixed atom count ``N`` baked into the graph.
        e_max: Fixed padded edge count. The deployer must pass exactly ``e_max``
            edges (real ones first, ``mask=True``; the rest padded, ``mask=False``)
            and fail loudly when the real edge count exceeds ``e_max``.
        cutoff: Neighbour cutoff in Å; padded edges get ``edge_diff`` of length
            ``10*cutoff`` so ``cutoff(edge_dist)`` zeros them.
    """

    def __init__(self, model: nn.Module, n_atoms: int, e_max: int, cutoff: float) -> None:
        super().__init__()
        self.model = model
        self.n_atoms = int(n_atoms)
        self.e_max = int(e_max)
        self.pad_len = float(cutoff) * 10.0

    def forward(
        self, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: ``Z (N,)`` int64, ``pos (N,3)``, ``edge_index (E_max,2)`` int64,
        ``mask (E_max,)`` bool. Returns ``(energy (), forces (N,3))``."""
        src = edge_index[:, 0]
        tgt = edge_index[:, 1]
        edge_diff = pos[tgt] - pos[src]
        # padded edges (mask False) -> length past cutoff -> inert
        edge_diff = torch.where(
            mask.unsqueeze(-1), edge_diff, torch.full_like(edge_diff, self.pad_len)
        )
        edge_dist = edge_diff.norm(dim=-1).clamp(min=1e-8)
        td = TensorDict(
            atoms=TensorDict(
                Z=Z, pos=pos,
                batch=torch.zeros(self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[self.n_atoms],
            ),
            edges=TensorDict(
                edge_index=edge_index, edge_diff=edge_diff, edge_dist=edge_dist,
                batch_size=[self.e_max],
            ),
            graphs=TensorDict(
                num_atoms=torch.full((1,), self.n_atoms, dtype=torch.long, device=pos.device),
                batch_size=[1],
            ),
            batch_size=[],
        )
        out = self.model(td, compute_forces=True)
        return out["energy"].reshape(()), out["forces"]
