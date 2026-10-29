"""Message-passing encoder: atom and bond streams updated over the covalent bonds.

Reads ``atoms.Z``, the declared integer channels and the bond list, never
``atoms.pos``. A seed bond embedding, then ``n_blocks`` pre-norm covalent hops
(:class:`~molrep.encoders.message_passing.blocks.MessagePassingBlock`), then a
LayerNorm on each stream; padded bond rows are zeroed after the norm. It
returns :class:`~molrep.encoders.features.EncoderFeatures` — one row per atom
and per bond — and nothing about angles or torsions: the force-field heads
assemble those from these rows.

Reference:
    Wang, Y. et al. "End-to-end differentiable construction of molecular
    mechanics force fields." Chem. Sci. 13, 12016 (2022).
    https://arxiv.org/abs/2010.01196
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molrep.encoders.features import EncoderFeatures
from molrep.encoders.message_passing.blocks import MessagePassingBlock
from molrep.encoders.message_passing.embed import (
    AtomChemEmbedding,
    BondChemEmbedding,
    DiscreteChannels,
)
from molrep.encoders.message_passing.spec import MessagePassingEncoderSpec

__all__ = ["MessagePassingEncoder"]


class MessagePassingEncoder(nn.Module):
    """Topology-only message-passing encoder.

    Every declared channel is a **required** batch column: a batch that lacks
    one raises ``KeyError`` naming it, never a silent zero.

    Args:
        spec: Encoder configuration.
        dtype: Floating dtype of every parameter.
    """

    def __init__(self, spec: MessagePassingEncoderSpec, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.spec = spec
        hidden = spec.hidden_dim
        self.atom_embed = AtomChemEmbedding(
            atom_dim=spec.atom_dim, num_elements=spec.num_elements, dtype=dtype
        )
        self.atom_channels = DiscreteChannels(spec.atom_channels, spec.atom_dim, dtype=dtype)
        self.bond_channels = DiscreteChannels(spec.bond_channels, spec.bond_dim, dtype=dtype)
        self.bond_embed = BondChemEmbedding(
            atom_dim=spec.atom_dim, bond_dim=spec.bond_dim, hidden_dim=hidden, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            MessagePassingBlock(
                atom_dim=spec.atom_dim,
                bond_dim=spec.bond_dim,
                hidden_dim=hidden,
                local_attention=spec.local_attention,
                dtype=dtype,
            )
            for _ in range(spec.n_blocks)
        )
        self.atom_norm = nn.LayerNorm(spec.atom_dim, dtype=dtype)
        self.bond_norm = nn.LayerNorm(spec.bond_dim, dtype=dtype)

    def forward(self, frame: TensorDict) -> EncoderFeatures:
        """Atom ``(N, atom_dim)`` and bond ``(N_bonds, bond_dim)`` features.

        Args:
            frame: Batch with ``atoms.Z``, the declared channels and
                ``bonds.bond_index`` ``(2, N_bonds)``.

        Returns:
            The features, in batch atom and bond order.
        """
        atoms, bonds = frame["atoms"], frame["bonds"]
        index = bonds["bond_index"].long()
        atomi, atomj = index[0], index[1]
        h_atom = self.atom_embed(atoms["Z"].long(), extra=self._atom_extra(frame))
        extra = self._bond_extra(bonds)
        live = bonds["mask"].to(h_atom.dtype).unsqueeze(-1) if "mask" in bonds.keys() else None
        mask = bonds["mask"] if "mask" in bonds.keys() else None
        h_bond = self.bond_embed(h_atom, atomi, atomj, extra=extra)
        if live is not None:
            h_bond = h_bond * live
        for block in self.blocks:
            h_atom, h_bond = block(h_atom, h_bond, atomi, atomj, extra=extra, mask=mask)
        h_bond = self.bond_norm(h_bond)
        if live is not None:
            h_bond = h_bond * live
        return EncoderFeatures(atom=self.atom_norm(h_atom), bond=h_bond)

    def _atom_extra(self, frame: TensorDict) -> torch.Tensor | None:
        """The summed atom channels, or ``None`` when none are declared."""
        if not len(self.atom_channels):
            return None
        per_graph = set(self.atom_channels.per_graph)
        columns: dict[str, torch.Tensor] = {}
        for name in self.atom_channels.names:
            namespace = "graphs" if name in per_graph else "atoms"
            if name not in frame[namespace].keys():
                raise KeyError(
                    f"the encoder declares the channel {namespace}.{name}; the batch has no "
                    "such column"
                )
            columns[name] = frame[namespace, name]
        batch = frame["atoms", "batch"] if per_graph else None
        return self.atom_channels(batch=batch, **columns)

    def _bond_extra(self, bonds: TensorDict) -> torch.Tensor | None:
        """The summed bond channels, or ``None`` when none are declared."""
        if not len(self.bond_channels):
            return None
        columns: dict[str, torch.Tensor] = {}
        for name in self.bond_channels.names:
            if name not in bonds.keys():
                raise KeyError(
                    f"the encoder declares the channel bonds.{name}; the batch has no such column"
                )
            columns[name] = bonds[name]
        return self.bond_channels(**columns)
