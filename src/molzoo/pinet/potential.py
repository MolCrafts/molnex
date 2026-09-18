"""PiNet energy + force potential (encoder + monomorphic init-fixed op).

All branching is in ``__init__``: what :meth:`PiNetPotential.forward` derives
beside the energy is the ``wrt`` given at construction, and ``forward`` is the
one :class:`molix.autograd.Derivative` built over :meth:`PiNetPotential.energy`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative
from molpot.derivation import EnergyAggregation
from molpot.derivation.protocol import write_energy
from molrep.interaction.pinet import OutLayer

from .encoder import PiNet


class PiNetPotential(nn.Module):
    """PiNet energy (+ optional derivatives) with in-place batch writes.

    Configuration is fixed at construction; ``forward`` has no flags::

        # energy only
        model = PiNetPotential(...)
        batch = model(batch)  # writes graphs.energy, atoms.energy

        # energy + forces through the derivative op
        model = PiNetPotential(..., wrt=("atoms.pos",))
        batch = model(batch)  # + atoms.forces

    ``create_graph`` follows the ambient grad mode, so a training step keeps
    energy and forces attached to the parameters and an evaluation under
    ``torch.no_grad()`` gets detached values. Compile the model as a whole with
    ``molix.compile.Compiler(...)(model)``.

    Args:
        hidden_dim: Width of the per-layer output MLPs.
        wrt: TensorFrame keys :meth:`forward` differentiates the energy
            against (:class:`molix.autograd.Derivative`); ``()`` is energy only.
        encoder: A built :class:`~molzoo.pinet.encoder.PiNet`; exclusive
            with ``pinet_kwargs``.
        **pinet_kwargs: Forwarded to :class:`~molzoo.pinet.encoder.PiNet`.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        wrt: Sequence[str | Sequence[str]] = (),
        encoder: PiNet | None = None,
        **pinet_kwargs: object,
    ) -> None:
        super().__init__()
        if encoder is not None and pinet_kwargs:
            raise ValueError("Pass either encoder=... or PiNet kwargs, not both.")
        if encoder is not None:
            self.encoder = encoder
        else:
            pinet_kwargs.setdefault("emit_property_features", False)
            self.encoder = PiNet(**pinet_kwargs)

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
        #: The one derivative route: the op over :meth:`energy`, fixed at
        #: construction (a bound method, so no submodule is registered).
        self._derivative = Derivative(self.energy, wrt=wrt)

    def forward(self, batch: TensorDict) -> TensorDict:
        """Run the init-fixed op; mutate ``batch`` in place."""
        return self._derivative(batch)

    @property
    def derivative(self) -> Derivative:
        """The op :meth:`forward` runs: ``Derivative(self.energy, wrt=...)``, fixed in ``__init__``.

        What a training step marks its leaves through
        (:meth:`molix.autograd.Derivative.mark_leaves`).
        """
        return self._derivative

    def energy(self, batch: TensorDict) -> TensorDict:
        """The energy alone — ``graphs.energy`` ``(B,)`` and ``atoms.energy`` ``(N,)``, in place."""
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
