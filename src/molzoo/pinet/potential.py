"""PiNet energy + force potential (encoder + monomorphic init-fixed pipeline).

All branching is in ``__init__`` (``compute_forces``, ``method``). ``forward``
always runs one static pipeline — energy-only has **no** Derivative session.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation import EnergyAggregation
from molpot.derivation.protocol import (
    ENERGY_KEY,
    POS_KEY,
    absorb_model_output,
    call_energy,
    has_energy,
    write_energy,
    write_forces,
)
from molrep.interaction.pinet import OutLayer

from .encoder import PiNet


class PiNetPotential(nn.Module):
    """PiNet energy (+ optional forces) with in-place batch writes.

    Configuration is fixed at construction; ``forward`` has no flags::

        # energy only
        model = PiNetPotential(..., compute_forces=False)
        batch = model(batch)  # writes graphs.energy

        # energy + forces (func = compile-friendly 1-pass)
        model = PiNetPotential(..., compute_forces=True, method="func")
        batch = model(batch)  # + atoms.forces

        # train force-matching often prefers method="grad"
        model = PiNetPotential(..., compute_forces=True, method="grad")
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        compute_forces: bool = False,
        method: Literal["func", "grad"] = "func",
        encoder: PiNet | None = None,
        **pinet_kwargs: object,
    ) -> None:
        super().__init__()
        if encoder is not None and pinet_kwargs:
            raise ValueError("Pass either encoder=... or PiNet kwargs, not both.")
        if method not in ("func", "grad"):
            raise ValueError(f"method must be 'func' or 'grad', got {method!r}")
        if encoder is not None:
            self.encoder = encoder
        else:
            pinet_kwargs.setdefault("emit_property_features", False)
            self.encoder = PiNet(**pinet_kwargs)  # type: ignore[arg-type]

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
        self.compute_forces = compute_forces
        self.method = method

        # Monomorphic pipeline — one bound call path, no runtime if/else.
        if not compute_forces:
            self._pipeline = self._write_energy
        elif method == "func":
            self._pipeline = self._pipeline_ef_func
        else:
            self._pipeline = self._pipeline_ef_grad

    def forward(self, batch: TensorDict) -> TensorDict:
        """Run the init-fixed pipeline; mutate ``batch`` in place."""
        return self._pipeline(batch)

    # ------------------------------------------------------------------ energy
    def _write_energy(self, batch: TensorDict) -> TensorDict:
        """Energy core only — also the energy-only public pipeline."""
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
        write_energy(batch, energy, atomic_energy=atom_energy)
        return batch

    # ----------------------------------------------------------- force kernels
    def _pipeline_ef_func(self, batch: TensorDict) -> TensorDict:
        """Init-fixed: single ``torch.func.grad(..., has_aux=True)`` pass.

        Compile-friendly monomorphic path (no session / no set_non_tensor).
        """
        pos = batch[POS_KEY].detach()
        base = batch.clone()
        base[POS_KEY] = pos

        def energy_fn_aux(p: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
            b = base.clone()
            b[POS_KEY] = p
            out = call_energy(self, b)
            b = absorb_model_output(b, out)
            if not has_energy(b):
                raise RuntimeError("energy core must write graphs.energy")
            return b[ENERGY_KEY].sum(), b

        grad, filled = torch.func.grad(energy_fn_aux, has_aux=True)(pos)
        batch[ENERGY_KEY] = filled[ENERGY_KEY]
        if "atoms" in filled.keys() and "energy" in filled["atoms"].keys():
            batch["atoms", "energy"] = filled["atoms", "energy"]
        write_forces(batch, -grad)
        return batch

    def _pipeline_ef_grad(self, batch: TensorDict) -> TensorDict:
        """Init-fixed: one energy pass + ``torch.autograd.grad`` on positions.

        Often faster for force-supervised training (``create_graph`` when
        ``self.training``).
        """
        pos = batch[POS_KEY].detach().requires_grad_(True)
        batch[POS_KEY] = pos
        batch = self._write_energy(batch)
        create_graph = bool(self.training)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(
                batch[ENERGY_KEY].sum(),
                pos,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        write_forces(batch, -g)
        return batch

    # ---------------------------------------------------------------- compile
    def compile(
        self,
        *,
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        mode: str | None = None,
    ) -> nn.Module:
        """Return ``torch.compile(self, ...)`` of this monomorphic module.

        Energy-only and ``method='func'`` force paths are the intended targets.
        Prefer fixed shapes (``PadMolecularBatch``) for ``fullgraph=True`` /
        ``mode='reduce-overhead'`` (CUDA graphs)::

            model = PiNetPotential(..., compute_forces=True, method="func")
            model = model.compile(fullgraph=True)  # OptimizedModule

        Also works via :class:`molix.compile.Compiler` /
        ``Trainer.compile(...)``.
        """
        return torch.compile(
            self,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
        )
