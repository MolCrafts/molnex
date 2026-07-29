"""PiNet energy + force potential (encoder composition + derivation).

Physics (force derivation, energy aggregation) lives in ``molpot`` via
:class:`molpot.composition.energy_force.EnergyForceModel`. This module only
wires the PiNet encoder + per-block OutLayers — the long-term home for a fully
generic encoder→energy façade remains ``molpot``; public import stays
``from molzoo.pinet import PiNetPotential``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition.energy_force import EnergyForceModel
from molpot.derivation import EnergyAggregation
from molrep.interaction.pinet import OutLayer

from .encoder import PiNet


class PiNetPotential(EnergyForceModel):
    """PiNet energy + force prediction model — ready to use from hyperparameters.

    Pass PiNet hyperparameters directly; the encoder is built internally::

        model = PiNetPotential(atom_types=[1, 6, 7, 8], r_max=4.5, depth=5,
                               hidden_dim=64, compute_forces=True)

    Forces use ``ForceDerivation(method="functorch")`` (pure-PyTorch graph)
    through :class:`EnergyForceModel`. All linears are fully specified at
    construction — no lazy materialisation.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        compute_forces: bool = False,
        encoder: PiNet | None = None,
        **pinet_kwargs: object,
    ) -> None:
        super().__init__(force_method="functorch", compute_forces=compute_forces)
        if encoder is not None and pinet_kwargs:
            raise ValueError("Pass either encoder=... or PiNet kwargs, not both.")
        self.encoder = encoder if encoder is not None else PiNet(**pinet_kwargs)  # type: ignore[arg-type]
        # Accepted for API compatibility; PiNet2 accumulates per-block OutLayers
        # residually — there is no layer axis to reduce for energy.
        self.layer_reduction = layer_reduction

        depth: int = int(getattr(self.encoder, "depth", 1))
        feature_dim: int = int(getattr(self.encoder, "feature_dim", hidden_dim))
        self.out_layers = nn.ModuleList(
            [
                OutLayer(
                    [hidden_dim],
                    in_dim=feature_dim,
                    out_units=1,
                    activation="tanh",
                )
                for _ in range(depth)
            ]
        )
        self.energy_aggregation = EnergyAggregation(pooling="sum")

    def energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Compilable energy: encoder → per-block OutLayer sum → aggregate."""
        batch = self.encoder(batch)

        block_outputs = batch["atoms", "p1_block_outputs"]  # (N, depth, D)
        output = block_outputs.new_zeros(block_outputs.shape[0], 1)
        for i, out_layer in enumerate(self.out_layers):
            output = out_layer(block_outputs[:, i, :], output)
        atom_energy = output.squeeze(-1)

        atom_batch = batch["atoms", "batch"]
        if "mask" in batch["atoms"].keys():
            atom_energy = atom_energy * batch["atoms", "mask"].to(atom_energy.dtype)
        num_graphs = batch["graphs"].batch_size[0]
        energy = self.energy_aggregation(atom_energy, atom_batch, num_graphs=num_graphs)

        return {
            "atomic_energy": atom_energy,
            "energy": energy,
        }

    # Back-compat alias used by older call sites / docs.
    def _energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        return self.energy_forward(batch)
