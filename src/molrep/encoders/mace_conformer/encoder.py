"""The MACE conformer encoder: a MACE body on one reference conformer.

Read out onto the features the force-field heads take.

:class:`MaceConformerEncoder` returns :class:`~molrep.encoders.features.EncoderFeatures`
— one row per atom and per bond at the widths ByteFF-26's heads read — from the
geometry of one conformer per molecule, so a force field built on it has
**static** parameters: every conformer of a profile is priced with the
parameters of the profile's first conformer, and no parameter depends on the
positions being differentiated. The positions are read detached
(``pairs.pair_diff``), so no ∂parameter/∂R term enters any force. Spec:
``.claude/specs/rondo-enc-02-mace.md``.

Primitives, composed by :meth:`MaceConformerEncoder.forward`:

* :meth:`~MaceConformerEncoder.invariants` — every layer's ``0e`` block, each
  conformer on its own geometry;
* :meth:`~MaceConformerEncoder.reference_atoms` — each atom's twin in the first
  conformer of its profile;
* :meth:`~MaceConformerEncoder.readout` — invariants and reference positions →
  atom and bond features, averaged over topological equivalence classes.

Reading out without the gather (``readout(frame, invariants(frame), pos)``) gives
each conformer its own parameters: the cross-conformer spread of spec 05.

References:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817 (the equivalence averaging and the feature
    interface)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molrep.chem.equivalence import Equivalence
from molrep.embedding.geometry import edge_lengths
from molrep.encoders.features import EncoderFeatures
from molrep.encoders.mace.encoder import MACEEncoder
from molrep.encoders.mace_conformer.layers import AtomReadout, BondReadout
from molrep.encoders.mace_conformer.spec import MaceConformerEncoderSpec

__all__ = ["MaceConformerEncoder"]


class MaceConformerEncoder(torch.nn.Module):
    """MACE invariants of a reference conformer → atom and bond features for the heads.

    Args:
        spec: Body and readout configuration.
        dtype: Floating dtype of every parameter.
    """

    def __init__(self, spec: MaceConformerEncoderSpec, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.spec = spec
        self.dtype = dtype
        self.mace = MACEEncoder(spec, dtype=dtype)
        self.num_features = spec.num_features
        scalars = spec.num_interactions * spec.num_features
        self.atom_readout = AtomReadout(
            scalars, width=spec.width, hidden=spec.hidden, charge_dim=spec.charge_dim, dtype=dtype
        )
        self.bond_readout = BondReadout(
            scalars,
            width=spec.bond_width,
            hidden=spec.hidden,
            lengths=spec.bond_lengths,
            dtype=dtype,
        )
        self.equivalence = Equivalence()

    @property
    def width(self) -> int:
        """Width of the atom features handed to the heads."""
        return self.spec.width

    @property
    def bond_width(self) -> int:
        """Width of the bond features handed to the heads."""
        return self.spec.bond_width

    def invariants(self, frame: TensorDict) -> torch.Tensor:
        """Every layer's ``0e`` block, concatenated: ``(N, num_interactions · num_features)``.

        Each conformer on its own geometry, read detached from
        ``pairs.pair_diff`` (dead padding pairs sit beyond the cutoff there,
        which a self-loop rebuilt from ``atoms.pos`` would not).

        Args:
            frame: Batch with ``atoms.Z``, ``pairs.pair_index`` ``(E, 2)`` (both
                directions of every pair) and ``pairs.pair_diff`` ``(E, 3)``.
        """
        Z = frame["atoms", "Z"]
        self.mace._validate_once(Z)
        edge_index = frame["pairs", "pair_index"]
        vectors = frame["pairs", "pair_diff"].detach().to(self.dtype)
        elements = self.mace.element_index(Z)
        edge_feats, cutoff = self.mace.radial_features(edge_lengths(vectors), Z, edge_index)
        per_layer = self.mace.layer_features(
            node_feats=self.mace.initial_node_features(elements),
            elements=elements,
            edge_attrs=self.mace.angular_features(vectors),
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        return torch.cat([h[:, : self.num_features] for h in per_layer], dim=-1)

    @staticmethod
    def reference_atoms(frame: TensorDict) -> torch.Tensor:
        """Each atom's twin in the first conformer of its profile ``(N,)``.

        The conformers of one profile (``graphs.profile``, batch-local, in
        ``[0, B)``) are one molecule in one atom order. An atom whose
        reference graph has another atom count — a ghost in the padder's sink,
        whose profile is shared with empty dead slots — is its own reference.
        Static shapes, no host sync.

        Args:
            frame: Batch with ``atoms.batch`` and ``graphs.profile``.
        """
        batch = frame["atoms", "batch"].long()
        profile = frame["graphs", "profile"].long()
        n, graphs = batch.shape[0], profile.shape[0]
        atom = torch.arange(n, device=batch.device)
        start = torch.full((graphs,), n, dtype=torch.long, device=batch.device)
        start = start.scatter_reduce(0, batch, atom, reduce="amin")
        count = torch.zeros_like(start).index_add(0, batch, torch.ones_like(batch))
        graph = torch.arange(graphs, device=batch.device)
        first = torch.full_like(graph, graphs).scatter_reduce(0, profile, graph, reduce="amin")
        reference = first[profile][batch]
        twin = atom - start[batch] + start[reference].clamp(max=n - 1)
        same = count[reference] == count[batch]
        return torch.where(same, twin, atom)

    def readout(
        self, frame: TensorDict, invariants: torch.Tensor, pos: torch.Tensor
    ) -> EncoderFeatures:
        """Atom ``(N, width)`` and bond ``(N_bonds, bond_width)`` features.

        Args:
            frame: Batch with ``atoms.formal_charge`` / ``equi_index`` and
                ``bonds.bond_index`` ``(2, N_bonds)`` / ``bond_type`` (1 single,
                2 double, 3 triple, 4 aromatic; 0 on a dead row) / ``in_ring``.
            invariants: Per-atom invariants (:meth:`invariants`, gathered or not).
            pos: Positions the bond lengths are read from ``(N, 3)``; detached here.
        """
        atoms, bonds = frame["atoms"], frame["bonds"]
        atom_class = atoms["equi_index"].long()
        atom = self.equivalence(
            self.atom_readout(invariants, atoms["formal_charge"].long()), atom_class
        )
        bond_index = bonds["bond_index"].long()
        i, j = bond_index[0], bond_index[1]
        if "mask" in bonds.keys():
            live = bonds["mask"].bool()
        else:
            live = torch.ones_like(i, dtype=torch.bool)
        pos = pos.detach().to(self.dtype)
        bond = self.bond_readout(
            invariants[i],
            invariants[j],
            torch.linalg.vector_norm(pos[j] - pos[i], dim=-1),
            torch.where(live, bonds["bond_type"].long() - 1, 0),
            bonds["in_ring"].long(),
        )
        bond = self.equivalence(bond, self.equivalence.bonds(i, j, atom_class))
        return EncoderFeatures(atom=atom, bond=bond)

    def forward(self, frame: TensorDict) -> EncoderFeatures:
        """Features from each profile's first conformer, for every conformer of it.

        Args:
            frame: Batch with the columns :meth:`invariants`,
                :meth:`reference_atoms` and :meth:`readout` read.
        """
        twin = self.reference_atoms(frame)
        return self.readout(frame, self.invariants(frame)[twin], frame["atoms", "pos"][twin])
