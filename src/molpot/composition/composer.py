"""Potential composer: head -> potential(mixing -> energy) -> forces."""

from __future__ import annotations

import torch
import torch.nn as nn


class PotentialComposer(nn.Module):
    """Compose parameter head + potentials into a force field.

    Pipeline::

        node_features → head → atom_params (dict)
            for each potential:
                potential.mixing_fn(atom_params, edge_index) → pair_params
                potential(distance, **pair_params) → energy

    Each potential carries its own ``mixing_fn`` that knows how to
    convert per-atom parameters to per-pair parameters.

    Args:
        head: Maps node features to per-atom parameter dict.
        potentials: Named potentials.
    """

    def __init__(
        self,
        head: nn.Module,
        potentials: dict[str, nn.Module],
    ):
        super().__init__()
        if not potentials:
            raise ValueError("PotentialComposer requires at least one potential.")
        self.head = head
        self.potentials = nn.ModuleDict(potentials)

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        data: dict[str, torch.Tensor],
        compute_forces: bool = False,
        head_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute composed energy and optionally forces.

        Args:
            node_features: Pooled per-node features ``(N, D)``.
            data: Must contain ``"edge_index"`` ``(E, 2)``,
                ``"batch"`` ``(N,)``. Must contain ``"pos"`` ``(N, 3)``
                if ``compute_forces=True``.
            compute_forces: Derive forces as ``-dE/dpos``.
            head_kwargs: Extra keyword arguments forwarded to the head
                (e.g. ``batch``, ``Z`` for MultiHead).

        Returns:
            Dict with ``"energy"``, ``"term_energies"``,
            ``"parameters"``, and optionally ``"forces"``.
        """
        edge_index = data["edge_index"]
        batch = data["batch"]
        src, dst = edge_index[:, 0], edge_index[:, 1]

        # 1. Head: features → per-atom params
        atom_params = self.head(node_features, **(head_kwargs or {}))

        # 2. Distance (recompute from pos for autograd)
        pos = data.get("pos")
        if pos is not None:
            distance = (pos[dst] - pos[src]).norm(dim=-1)
        else:
            distance = data["bond_dist"]

        # 3. Evaluate each potential (each applies its own mixing)
        edge_batch = batch[src]
        num_graphs = data.get("num_graphs")

        term_energies: dict[str, torch.Tensor] = {}
        total_energy: torch.Tensor | None = None

        for name, potential in self.potentials.items():
            pair_params = potential.mixing_fn(atom_params, edge_index)
            energy = potential(
                distance=distance,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
                **pair_params,
            )
            term_energies[name] = energy
            total_energy = energy if total_energy is None else total_energy + energy

        assert total_energy is not None

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "energy": total_energy,
            "term_energies": term_energies,
            "parameters": atom_params,
        }

        # 4. Forces
        if compute_forces:
            pos = data["pos"]
            if not pos.requires_grad:
                raise RuntimeError("compute_forces=True requires data['pos'].requires_grad=True")
            forces = -torch.autograd.grad(
                total_energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            outputs["forces"] = forces

        return outputs
