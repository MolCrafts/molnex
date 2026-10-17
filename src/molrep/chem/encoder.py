"""ChemEncoder — topology-aware continuous chemical perception.

Reads valence namespaces from a nested :class:`~tensordict.TensorDict` batch
and writes continuous chem features under ``*.chem_features``. No energy,
no force, no molpot import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix import config
from molrep.chem.blocks import ChemGraphBlock
from molrep.chem.bond_table import BondTable
from molrep.chem.context import (
    AngleContext,
    BondContext,
    ImproperContext,
    ProperContext,
)
from molrep.chem.embed import AtomChemEmbedding, DiscreteChannel, DiscreteChannels
from molrep.chem.features import ChemEmbeddings

__all__ = ["ChemEncoder"]


def _ns(batch: Any, name: str) -> Any | None:
    """Fetch a top-level namespace from TensorDict / Mapping."""
    if batch is None:
        return None
    try:
        if name in batch:
            return batch[name]
    except Exception:  # noqa: BLE001 — TensorDict key miss variants
        return None
    return None


def _get_tensor(ns: Any, *keys: str) -> torch.Tensor | None:
    if ns is None:
        return None
    for key in keys:
        try:
            if isinstance(ns, (Mapping, TensorDict)) and key in ns:
                return ns[key]
        except Exception:  # noqa: BLE001
            continue
    return None


def _bond_endpoints(bonds: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(atomi, atomj)`` from column keys or COO ``bond_index``."""
    if bonds is None:
        return None
    atomi = _get_tensor(bonds, "atomi")
    atomj = _get_tensor(bonds, "atomj")
    if atomi is not None and atomj is not None:
        return atomi.long(), atomj.long()
    packed = _get_tensor(bonds, "bond_index")
    if packed is None:
        return None
    # Contract: bond_index is COO (2, N_bonds)
    if packed.dim() != 2:
        raise ValueError(f"bond_index must be 2-D, got shape {tuple(packed.shape)}")
    if packed.shape[0] == 2:
        return packed[0].long(), packed[1].long()
    if packed.shape[1] == 2:
        # Tolerate (N, 2) layout
        return packed[:, 0].long(), packed[:, 1].long()
    raise ValueError(f"bond_index must be (2, N) or (N, 2), got shape {tuple(packed.shape)}")


def _empty(dim: int, like: torch.Tensor) -> torch.Tensor:
    """A ``(0, dim)`` stand-in for a namespace this batch does not carry.

    dtype and device come from *like* — pass the atom embedding, whose dtype is
    by definition the one these features would have had. Reading a process-wide
    config here instead was a **graph break**: molcfg resolves every
    attribute through a custom ``__getattribute__``, which dynamo cannot trace
    ("expected 1 argument, got 2"), and this function sits on the traced half of
    a Class-I step (measured on a GH200).
    """
    return like.new_zeros((0, dim))


class ChemEncoder(nn.Module):
    """Topology-aware chemical perception encoder.

    Reads ``atoms.Z`` and valence namespaces (``bonds`` / ``angles`` /
    ``propers`` / ``impropers``) and writes continuous features under
    ``*.chem_features``.

    Primary path:
        ``forward(batch) -> batch`` (mutates / returns TensorDict with features).

    Secondary path:
        ``compose(batch) -> ChemEmbeddings`` then
        ``write_batch(batch, emb)``; ``embeddings(batch)`` views written fields.

    Coordinates (``atoms.pos`` / ``edge_diff`` / ``edge_dist``) are never
    read. After the seed bond context, ``n_blocks`` covalent hops update
    ``h_atom`` / ``h_bond``; angle / proper / improper contexts consume the
    post-message-passing atom features **and the features of the term's own
    bonds**, looked up by endpoint pair in a
    :class:`~molrep.chem.bond_table.BondTable` built once per batch (ByteFF
    ``OutputWithBond``). A batch that carries angles, propers or impropers
    without a ``bonds`` namespace is refused by name: a term without its bonds
    is not a term this encoder can describe.

    Every feature channel it returns is layer-normalized (atom, bond, angle,
    proper, improper), and the hops are pre-norm
    (:class:`~molrep.chem.blocks.ChemGraphBlock`). A parameter head therefore
    reads features of unit scale however the trunk's weights have grown, which
    is what keeps a head's pre-activation, and so a predicted force constant,
    from being moved by the trunk out of all proportion to one optimizer step.
    Padded rows are zeroed after the norm, as before.

    Extra integer columns enter as :class:`~molrep.chem.embed.DiscreteChannel`
    declarations — ByteFF's atom features are element, total charge, formal
    charge, ring connectivity and smallest ring; its bond features are bond
    order and ring membership (``mol_data.py:174-201``). Each declared column
    is **required**: a batch that lacks it raises ``KeyError`` naming the
    column, never a silent zero.

    Args:
        atom_dim: Atom feature dimension ``D_a``.
        bond_dim: Bond feature dimension ``D_b``.
        angle_dim: Angle feature dimension.
        proper_dim: Proper-torsion feature dimension.
        improper_dim: Improper feature dimension.
        num_elements: Atomic-number vocabulary size.
        hidden_dim: Shared hidden width for context MLPs and graph blocks
            (optional).
        atom_channels: Extra atom columns, read from ``atoms.<name>``
            (``per="node"``) or ``graphs.<name>`` (``per="graph"``, broadcast
            through ``atoms.batch``) and summed into the atom seed.
        bond_channels: Extra bond columns, read from ``bonds.<name>`` and
            summed into the bond seed and every hop.
        n_blocks: Number of :class:`~molrep.chem.blocks.ChemGraphBlock`
            hops. A non-negative count; tier policy lives on the experiment,
            not here. Default ``5``.
        local_attention: If ``True``, each block softmax-normalizes over an
            atom's covalent neighborhood only. Default ``False``.
    """

    out_keys = [
        ("atoms", "chem_features"),
        ("bonds", "chem_features"),
        ("angles", "chem_features"),
        ("propers", "chem_features"),
        ("impropers", "chem_features"),
    ]

    def __init__(
        self,
        *,
        atom_dim: int = 32,
        bond_dim: int = 32,
        angle_dim: int = 32,
        proper_dim: int = 32,
        improper_dim: int = 32,
        num_elements: int = 119,
        hidden_dim: int | None = None,
        atom_channels: Sequence[DiscreteChannel] = (),
        bond_channels: Sequence[DiscreteChannel] = (),
        n_blocks: int = 5,
        local_attention: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        n_blocks = int(n_blocks)
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.angle_dim = int(angle_dim)
        self.proper_dim = int(proper_dim)
        self.improper_dim = int(improper_dim)
        self.num_elements = int(num_elements)
        self.n_blocks = n_blocks
        self.local_attention = bool(local_attention)

        self.atom_embed = AtomChemEmbedding(
            atom_dim=self.atom_dim,
            num_elements=self.num_elements,
            dtype=dtype,
        )
        self.atom_channels = DiscreteChannels(atom_channels, self.atom_dim, dtype=dtype)
        self.bond_channels = DiscreteChannels(bond_channels, self.bond_dim, dtype=dtype)
        per_graph = set(self.atom_channels.per_graph)
        self.in_keys: list[tuple[str, str]] = [("atoms", "Z")]
        for name in self.atom_channels.names:
            self.in_keys.append(("graphs", name) if name in per_graph else ("atoms", name))
        if per_graph:
            self.in_keys.append(("atoms", "batch"))
        self.in_keys.extend(("bonds", name) for name in self.bond_channels.names)
        self.bond_context = BondContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            [
                ChemGraphBlock(
                    atom_dim=self.atom_dim,
                    bond_dim=self.bond_dim,
                    hidden_dim=hidden_dim,
                    local_attention=self.local_attention,
                    dtype=dtype,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.angle_context = AngleContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            angle_dim=self.angle_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        self.proper_context = ProperContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            proper_dim=self.proper_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        self.improper_context = ImproperContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            improper_dim=self.improper_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        self.atom_norm = nn.LayerNorm(self.atom_dim, dtype=dtype)
        self.bond_norm = nn.LayerNorm(self.bond_dim, dtype=dtype)
        self.angle_norm = nn.LayerNorm(self.angle_dim, dtype=dtype)
        self.proper_norm = nn.LayerNorm(self.proper_dim, dtype=dtype)
        self.improper_norm = nn.LayerNorm(self.improper_dim, dtype=dtype)

    # ------------------------------------------------------------------
    # compose / write / view
    # ------------------------------------------------------------------

    def compose(self, batch: TensorDict | Mapping[str, Any]) -> ChemEmbeddings:
        """Build :class:`ChemEmbeddings` from a valence TensorDict batch.

        Args:
            batch: Nested batch with ``atoms.Z`` and optional valence
                namespaces using column keys ``atomi``/``atomj``/… (or
                ``bonds.bond_index`` COO).

        Returns:
            :class:`ChemEmbeddings` with counts matching present topology.
            Missing optional namespaces yield empty ``(0, D)`` tensors.
        """
        atoms = _ns(batch, "atoms")
        if atoms is None:
            raise KeyError("ChemEncoder requires batch['atoms']")
        z = _get_tensor(atoms, "Z")
        if z is None:
            raise KeyError("ChemEncoder requires batch['atoms', 'Z']")

        h_atom = self.atom_embed(z.long(), extra=self._atom_extra(batch, atoms))

        # Bonds
        bonds = _ns(batch, "bonds")
        endpoints = _bond_endpoints(bonds)
        table: BondTable | None = None
        if endpoints is None:
            h_bond = _empty(self.bond_dim, h_atom)
        else:
            atomi, atomj = endpoints
            extra = self._bond_extra(bonds)
            h_bond = self.bond_context(h_atom, atomi, atomj, extra=extra)
            bmask = _get_tensor(bonds, "mask")
            if bmask is not None:
                h_bond = h_bond * bmask.to(dtype=h_bond.dtype).unsqueeze(-1)
            if int(atomi.shape[0]) > 0:
                for block in self.blocks:
                    h_atom, h_bond = block(h_atom, h_bond, atomi, atomj, extra=extra, mask=bmask)
            h_bond = self.bond_norm(h_bond)
            if bmask is not None:
                h_bond = h_bond * bmask.to(dtype=h_bond.dtype).unsqueeze(-1)
            table = BondTable(atomi, atomj, int(z.shape[0]))
        h_atom = self.atom_norm(h_atom)

        # Angles
        angles = _ns(batch, "angles")
        ai = _get_tensor(angles, "atomi")
        aj = _get_tensor(angles, "atomj")
        ak = _get_tensor(angles, "atomk")
        if ai is None or aj is None or ak is None:
            h_angle = _empty(self.angle_dim, h_atom)
        else:
            table = self._term_bonds(table, "angles")
            h_angle = self.angle_norm(
                self.angle_context(
                    h_atom,
                    ai,
                    aj,
                    ak,
                    table.gather(h_bond, ai, aj),
                    table.gather(h_bond, aj, ak),
                )
            )
            amask = _get_tensor(angles, "mask")
            if amask is not None:
                h_angle = h_angle * amask.to(dtype=h_angle.dtype).unsqueeze(-1)

        # Propers
        propers = _ns(batch, "propers")
        pi = _get_tensor(propers, "atomi")
        pj = _get_tensor(propers, "atomj")
        pk = _get_tensor(propers, "atomk")
        pl = _get_tensor(propers, "atoml")
        if pi is None or pj is None or pk is None or pl is None:
            h_proper = _empty(self.proper_dim, h_atom)
        else:
            table = self._term_bonds(table, "propers")
            h_proper = self.proper_norm(
                self.proper_context(
                    h_atom,
                    pi,
                    pj,
                    pk,
                    pl,
                    table.gather(h_bond, pi, pj),
                    table.gather(h_bond, pj, pk),
                    table.gather(h_bond, pk, pl),
                )
            )
            pmask = _get_tensor(propers, "mask")
            if pmask is not None:
                h_proper = h_proper * pmask.to(dtype=h_proper.dtype).unsqueeze(-1)

        # Impropers (atomi = center)
        impropers = _ns(batch, "impropers")
        ii = _get_tensor(impropers, "atomi")
        ij = _get_tensor(impropers, "atomj")
        ik = _get_tensor(impropers, "atomk")
        il = _get_tensor(impropers, "atoml")
        if ii is None or ij is None or ik is None or il is None:
            h_improper = _empty(self.improper_dim, h_atom)
        else:
            table = self._term_bonds(table, "impropers")
            h_improper = self.improper_norm(
                self.improper_context(
                    h_atom,
                    ii,
                    ij,
                    ik,
                    il,
                    table.gather(h_bond, ii, ij),
                    table.gather(h_bond, ii, ik),
                    table.gather(h_bond, ii, il),
                )
            )
            imask = _get_tensor(impropers, "mask")
            if imask is not None:
                h_improper = h_improper * imask.to(dtype=h_improper.dtype).unsqueeze(-1)

        return ChemEmbeddings(
            atom=h_atom,
            bond=h_bond,
            angle=h_angle,
            proper=h_proper,
            improper=h_improper,
        )

    def _atom_extra(self, batch: Any, atoms: Any) -> torch.Tensor | None:
        """The summed atom channels of *batch*, or ``None`` when none are declared."""
        if not len(self.atom_channels):
            return None
        per_graph = set(self.atom_channels.per_graph)
        graphs = _ns(batch, "graphs") if per_graph else None
        columns: dict[str, torch.Tensor] = {}
        for name in self.atom_channels.names:
            source, namespace = (graphs, "graphs") if name in per_graph else (atoms, "atoms")
            column = _get_tensor(source, name)
            if column is None:
                raise KeyError(
                    f"ChemEncoder declares the channel {namespace}.{name}; the batch has no "
                    f"such column"
                )
            columns[name] = column
        index = _get_tensor(atoms, "batch") if per_graph else None
        if per_graph and index is None:
            raise KeyError("ChemEncoder needs atoms.batch to broadcast a per-graph channel")
        return self.atom_channels(batch=index, **columns)

    @staticmethod
    def _term_bonds(table: BondTable | None, namespace: str) -> BondTable:
        """The batch's bond table, which a valence namespace cannot do without."""
        if table is None:
            raise KeyError(
                f"the batch carries {namespace} but no bonds; ChemEncoder reads the features "
                f"of every term's own bonds, so a batch with {namespace} needs the bonds "
                "namespace (atomi/atomj or bond_index)"
            )
        return table

    def _bond_extra(self, bonds: Any) -> torch.Tensor | None:
        """The summed bond channels of *bonds*, or ``None`` when none are declared."""
        if not len(self.bond_channels):
            return None
        columns: dict[str, torch.Tensor] = {}
        for name in self.bond_channels.names:
            column = _get_tensor(bonds, name)
            if column is None:
                raise KeyError(
                    f"ChemEncoder declares the channel bonds.{name}; the batch has no such column"
                )
            columns[name] = column
        return self.bond_channels(**columns)

    def write_batch(
        self,
        batch: TensorDict,
        emb: ChemEmbeddings,
    ) -> TensorDict:
        """Write chem features onto the batch under ``*.chem_features``.

        Creates missing valence namespaces when the corresponding feature
        tensor is empty so all five out-keys are always present after a
        write.

        Args:
            batch: Nested TensorDict batch (mutated in place).
            emb: Feature payload from :meth:`compose`.

        Returns:
            The same ``batch`` with features written.
        """
        if "atoms" not in batch:
            raise KeyError("write_batch requires batch['atoms']")
        batch["atoms", "chem_features"] = emb.atom

        self._ensure_ns(batch, "bonds", emb.bond.shape[0])
        batch["bonds", "chem_features"] = emb.bond

        self._ensure_ns(batch, "angles", emb.angle.shape[0])
        batch["angles", "chem_features"] = emb.angle

        self._ensure_ns(batch, "propers", emb.proper.shape[0])
        batch["propers", "chem_features"] = emb.proper

        self._ensure_ns(batch, "impropers", emb.improper.shape[0])
        batch["impropers", "chem_features"] = emb.improper
        return batch

    @staticmethod
    def _ensure_ns(batch: TensorDict, name: str, n: int) -> None:
        if name not in batch:
            batch[name] = TensorDict({}, batch_size=[n] if n > 0 else [])

    def embeddings(self, batch: TensorDict | Mapping[str, Any]) -> ChemEmbeddings:
        """View written ``*.chem_features`` as :class:`ChemEmbeddings`.

        Args:
            batch: Batch previously passed through :meth:`forward` /
                :meth:`write_batch`.

        Returns:
            :class:`ChemEmbeddings` viewing the feature fields.

        Raises:
            KeyError: If required chem feature keys are missing.
        """
        if isinstance(batch, TensorDict):
            return ChemEmbeddings(
                atom=batch["atoms", "chem_features"],
                bond=batch["bonds", "chem_features"],
                angle=batch["angles", "chem_features"],
                proper=batch["propers", "chem_features"],
                improper=batch["impropers", "chem_features"],
            )
        return ChemEmbeddings(
            atom=batch["atoms"]["chem_features"],
            bond=batch["bonds"]["chem_features"],
            angle=batch["angles"]["chem_features"],
            proper=batch["propers"]["chem_features"],
            improper=batch["impropers"]["chem_features"],
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """Compose features and write them back onto ``batch``.

        Args:
            batch: Nested TensorDict with ``atoms.Z`` and valence topology.

        Returns:
            The same batch with ``*.chem_features`` populated.
        """
        emb = self.compose(batch)
        return self.write_batch(batch, emb)
