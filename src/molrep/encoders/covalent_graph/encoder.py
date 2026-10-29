"""The covalent-graph encoder: an edge-augmented graph transformer on the covalent graph.

:class:`CovalentGraphEncoder` reads the covalent graph and never ``atoms.pos``, and
returns :class:`~molrep.encoders.features.EncoderFeatures` — one row per atom and per
bond — which is everything a parameter head reads. Spec:
``.claude/specs/rondo-enc-01-graph-encoder.md``.

References:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molrep.encoders.covalent_graph.layers import ChemicalEmbedding, EdgeAttentionLayer
from molrep.encoders.covalent_graph.spec import CovalentGraphEncoderSpec
from molrep.encoders.features import EncoderFeatures

__all__ = ["CovalentGraphEncoder"]


class CovalentGraphEncoder(torch.nn.Module):
    """Covalent-graph encoder: input embeddings, edge-augmented attention, jumping knowledge.

    Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md`` §3 (``graph_block.py:Graph2DBlock``).
    Reads the covalent graph and never ``atoms.pos``. The adapter to the heads
    is the identity: the features are the published model's own, so its pretrained heads
    read them unchanged.

    Every bond is two directed edges, ``2b`` (``i → j``) and ``2b + 1``
    (``j → i``); after the last layer the two directions are averaged, so a
    bond has one feature row.

    Args:
        spec: Encoder hyperparameters.
        dtype: Floating dtype of every parameter.

    Raises:
        ValueError: The atom MLP's output is not the transformer width.
    """

    def __init__(self, spec: CovalentGraphEncoderSpec, *, dtype: torch.dtype) -> None:
        super().__init__()
        if spec.node_mlp[1] != spec.width:
            raise ValueError(
                f"the atom MLP ends at {spec.node_mlp[1]}, the transformer width is {spec.width}"
            )
        self.spec = spec
        self.feature = ChemicalEmbedding(
            atom_embedding_dim=spec.atom_embedding_dim,
            channel_dim=spec.channel_dim,
            node_mlp=spec.node_mlp,
            edge_mlp=spec.edge_mlp,
            dtype=dtype,
        )
        self.edge_in = torch.nn.Linear(self.feature.edge_dim, spec.width, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            EdgeAttentionLayer(
                spec.width, heads=spec.heads, head_dim=spec.head_dim, ffn=spec.ffn, dtype=dtype
            )
            for _ in range(spec.n_layers)
        )
        self.jk = torch.nn.Linear(spec.n_layers * spec.width, spec.width, dtype=dtype)
        self.edge_out = torch.nn.Linear(spec.width, self.feature.edge_dim, dtype=dtype)

    @property
    def width(self) -> int:
        """Width of the atom and bond features handed to the heads."""
        return self.spec.width

    def forward(self, frame: TensorDict) -> EncoderFeatures:
        """Atom ``(N, width)`` and bond ``(N_bonds, edge_dim)`` features.

        Args:
            frame: Batch with ``atoms.Z`` / ``formal_charge`` /
                ``ring_connectivity`` / ``min_ring_size`` / ``equi_index`` (the
                representative atom of each atom's equivalence class) and
                ``bonds.bond_index`` ``(2, N_bonds)`` / ``bond_type`` (1 single,
                2 double, 3 triple, 4 aromatic) / ``in_ring``. A bond's class is
                the pair of its endpoints' classes.

        Returns:
            The features, in batch atom and bond order.
        """
        atoms, bonds = frame["atoms"], frame["bonds"]
        bond_index = bonds["bond_index"].long()
        i, j = bond_index[0], bond_index[1]
        Z = atoms["Z"].long()
        # Live bonds only: a padded batch's dead bond rows all join the same two
        # ghost atoms, whose degree would otherwise run past the embedding. A dead
        # row's ``bond_type`` is 0, no order at all, so it reads order index 0;
        # it joins ghosts only, so what it reads reaches no real atom.
        live = bonds["mask"].long() if "mask" in bonds.keys() else torch.ones_like(i)
        degree = torch.zeros_like(Z).index_add(0, i, live).index_add(0, j, live)
        atom_class = atoms["equi_index"].long()
        x = self.feature.atoms(
            Z,
            degree,
            atoms["formal_charge"].long(),
            atoms["ring_connectivity"].long(),
            atoms["min_ring_size"].long(),
            atom_class,
        )
        bond = self.feature.bonds(
            bonds["in_ring"].long(),
            torch.where(live.bool(), bonds["bond_type"].long() - 1, 0),
            self.feature.equivalence.bonds(i, j, atom_class),
        )
        source = torch.stack((i, j), dim=1).reshape(-1)
        target = torch.stack((j, i), dim=1).reshape(-1)
        e = self.edge_in(bond.repeat_interleave(2, dim=0))
        layers = []
        for layer in self.layers:
            x, e = layer(x, e, source, target)
            x = torch.nn.functional.gelu(x)
            layers.append(x)
        x = self.jk(torch.cat(layers, dim=-1))
        e = self.edge_out(e).view(-1, 2, self.feature.edge_dim).mean(dim=1)
        return EncoderFeatures(atom=x, bond=e)
