"""PiNet energy + force potential (encoder composition + derivation).

Industrial note: this module is a *composition* of molrep encoder blocks and
molpot derivation/heads. Long-term home is ``molpot``; it remains under
``molzoo.pinet`` for import stability while the package boundary migration
completes. Public import path: ``from molzoo.pinet import PiNetPotential``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation import EnergyAggregation, ForceDerivation
from molrep.interaction.pinet import OutLayer

from .encoder import PiNet


class PiNetPotential(nn.Module):
    """PiNet energy + force prediction model — ready to use from hyperparameters.

    Pass PiNet hyperparameters directly; the encoder is built internally::

        model = PiNetPotential(atom_types=[1, 6, 7, 8], r_max=4.5, depth=5,
                               hidden_dim=64, compute_forces=True)

    Forces use ``ForceDerivation(method="functorch")`` (pure-PyTorch graph).
    All linears are fully specified at construction — no lazy materialisation.
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
        super().__init__()
        if encoder is not None and pinet_kwargs:
            raise ValueError("Pass either encoder=... or PiNet kwargs, not both.")
        self.encoder = encoder if encoder is not None else PiNet(**pinet_kwargs)  # type: ignore[arg-type]
        # Accepted for API compatibility; PiNet2 accumulates per-block OutLayers
        # residually — there is no layer axis to reduce for energy.
        self.layer_reduction = layer_reduction
        self.compute_forces_default = compute_forces
        self._compiled_energy_forward = None

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
        self.force_derivation = ForceDerivation(method="functorch")

    def forward(
        self, batch: TensorDict, *, compute_forces: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if compute_forces is None:
            compute_forces = self.compute_forces_default
        if compute_forces:
            return self._forward_functorch(batch)
        energy_forward = self._compiled_energy_forward or self._energy_forward
        return energy_forward(batch)

    def _forward_functorch(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Force forward via ``torch.func.grad`` (no double-backward barrier).

        **Training**: eager energy pass (params connected) + functorch force
        pass. **Eval**: single ``grad(..., has_aux=True)`` pass.
        """
        pos = batch["atoms", "pos"].detach()
        base = batch.clone()
        base["atoms", "pos"] = pos

        def energy_fn(p: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            b = base.clone()
            b["atoms", "pos"] = p
            out = self._energy_forward(b)
            return out["energy"].sum(), out

        if not self.training:
            grad, out = torch.func.grad(energy_fn, has_aux=True)(pos)
            out["forces"] = -grad
            return out

        out = self._energy_forward(base.clone())
        out["forces"] = self.force_derivation(lambda p: energy_fn(p)[0], pos)
        return out

    def _energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
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

    def compile_energy(self, *, backend: str = "inductor", **kwargs) -> None:
        """Compile the energy-only forward (not the force training path).

        See module docs / ``docs/molix/explanation/throughput-and-compilation.md``
        for when to use this vs whole-model CUDA graphs on padded batches.
        """
        self._compiled_energy_forward = torch.compile(
            self._energy_forward, backend=backend, **kwargs
        )
