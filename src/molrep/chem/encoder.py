"""ChemEncoder — topology-aware continuous chemical perception.

Reads valence namespaces from a nested :class:`~tensordict.TensorDict` batch
and writes continuous chem features under ``*.chem_features``. No energy,
no force, no molpot import.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix import config
from molrep.chem.context import (
    AngleContext,
    BondContext,
    ImproperContext,
    ProperContext,
)
from molrep.chem.embed import AtomChemEmbedding
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
    return like.new_zeros((0, dim), dtype=config.ftype)


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

    Args:
        atom_dim: Atom feature dimension ``D_a``.
        bond_dim: Bond feature dimension ``D_b``.
        angle_dim: Angle feature dimension.
        proper_dim: Proper-torsion feature dimension.
        improper_dim: Improper feature dimension.
        num_elements: Atomic-number vocabulary size.
        hidden_dim: Shared hidden width for context MLPs (optional).
        num_bond_types: Bond-type table size; ``0`` disables type conditioning.
    """

    in_keys = [
        ("atoms", "Z"),
    ]
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
        num_bond_types: int = 0,
    ) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.angle_dim = int(angle_dim)
        self.proper_dim = int(proper_dim)
        self.improper_dim = int(improper_dim)
        self.num_elements = int(num_elements)

        self.atom_embed = AtomChemEmbedding(
            atom_dim=self.atom_dim,
            num_elements=self.num_elements,
        )
        self.bond_context = BondContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=hidden_dim,
            num_bond_types=num_bond_types,
        )
        self.angle_context = AngleContext(
            atom_dim=self.atom_dim,
            angle_dim=self.angle_dim,
            hidden_dim=hidden_dim,
        )
        self.proper_context = ProperContext(
            atom_dim=self.atom_dim,
            proper_dim=self.proper_dim,
            hidden_dim=hidden_dim,
        )
        self.improper_context = ImproperContext(
            atom_dim=self.atom_dim,
            improper_dim=self.improper_dim,
            hidden_dim=hidden_dim,
        )

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

        h_atom = self.atom_embed(z.long())
        ref = h_atom if h_atom.numel() else z

        # Bonds
        bonds = _ns(batch, "bonds")
        endpoints = _bond_endpoints(bonds)
        if endpoints is None:
            h_bond = _empty(self.bond_dim, ref)
        else:
            atomi, atomj = endpoints
            btype = _get_tensor(bonds, "bond_types", "type", "bond_type")
            h_bond = self.bond_context(h_atom, atomi, atomj, bond_type=btype)

        # Angles
        angles = _ns(batch, "angles")
        ai = _get_tensor(angles, "atomi")
        aj = _get_tensor(angles, "atomj")
        ak = _get_tensor(angles, "atomk")
        if ai is None or aj is None or ak is None:
            h_angle = _empty(self.angle_dim, ref)
        else:
            h_angle = self.angle_context(h_atom, ai, aj, ak)

        # Propers
        propers = _ns(batch, "propers")
        pi = _get_tensor(propers, "atomi")
        pj = _get_tensor(propers, "atomj")
        pk = _get_tensor(propers, "atomk")
        pl = _get_tensor(propers, "atoml")
        if pi is None or pj is None or pk is None or pl is None:
            h_proper = _empty(self.proper_dim, ref)
        else:
            h_proper = self.proper_context(h_atom, pi, pj, pk, pl)

        # Impropers (atomi = center)
        impropers = _ns(batch, "impropers")
        ii = _get_tensor(impropers, "atomi")
        ij = _get_tensor(impropers, "atomj")
        ik = _get_tensor(impropers, "atomk")
        il = _get_tensor(impropers, "atoml")
        if ii is None or ij is None or ik is None or il is None:
            h_improper = _empty(self.improper_dim, ref)
        else:
            h_improper = self.improper_context(h_atom, ii, ij, ik, il)

        return ChemEmbeddings(
            atom=h_atom,
            bond=h_bond,
            angle=h_angle,
            proper=h_proper,
            improper=h_improper,
        )

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
