"""The two shapes of a molrs style's differentiable torch implementation.

A molpot potential is the torch twin of one molrs force-field style
(``"bond:harmonic"``, ``"pair:lj/cut"``, …) and prices exactly what molrs
prices, in molrs's convention — LAMMPS's: no hidden ½, angle-valued parameters
in degrees, the improper dihedral of the atoms as stored. molrs is the
numerical oracle; molnex adds only the autograd.

Parameters come from one of two places, decided per call:

* **passed in** — ``potential(frame, k=..., r0=...)`` with one row per term
  (bonded) or per atom (pair). This is how a learned model hands over the
  tables its heads predicted;
* **carried** — the per-type table read from the force field's types at
  construction (:meth:`from_style`), looked up through the frame's integer
  ``<namespace>.type`` column (``atoms.type`` for pair styles), which indexes
  the style's types in declaration order.

Every potential returns one energy per graph ``(B,)``. Dead rows contribute
zero by multiplication with their ``mask``, never by selection, so shapes stay
static under ``torch.compile``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch
import torch.nn as nn
from tensordict import TensorDictBase

__all__ = ["BondedPotential", "PairPotential"]

#: molrs category → batch namespace. The batch speaks ``propers`` where molrs
#: speaks ``dihedrals``.
NAMESPACE: dict[str, str] = {
    "bond": "bonds",
    "angle": "angles",
    "dihedral": "propers",
    "improper": "impropers",
}

_INDEX_COLUMNS = ("atomi", "atomj", "atomk", "atoml")


def _graph_count(frame: TensorDictBase) -> int:
    """``B``, the graph slots of the batch (dead slots included)."""
    return int(frame["graphs"].batch_size[0])


class BondedPotential(nn.Module):
    """A style whose terms each read ``width`` atoms of one covalent namespace.

    Subclasses state ``style``, ``width`` and ``params`` (the molrs parameter
    names, in order) and implement :meth:`term_energy`.

    Args:
        table: Per-type parameters, name → ``(n_types, ...)``, or ``None`` when
            every call passes its parameters.
    """

    style: ClassVar[str]
    width: ClassVar[int]
    params: ClassVar[tuple[str, ...]]

    def __init__(self, table: Mapping[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.namespace = NAMESPACE[self.style.split(":", 1)[0]]
        self.has_table = table is not None
        if table is not None:
            missing = [name for name in self.params if name not in table]
            if missing:
                raise ValueError(f"{self.style}: table lacks {missing}")
            for name in self.params:
                self.register_buffer(f"table_{name}", torch.as_tensor(table[name]))

    @classmethod
    def from_style(cls, style: Any, **_: Any) -> BondedPotential:
        """The potential of one molpy style, carrying its types' parameters.

        Args:
            style: A molpy ``Style`` of this class's category and name.

        Returns:
            The potential, with a per-type table when the style declares types.
        """
        types = list(style.get_types())
        if not types:
            return cls()
        return cls(cls.table_from_types(types))

    @classmethod
    def table_from_types(cls, types: Sequence[Any]) -> dict[str, torch.Tensor]:
        """Stack the types' parameters into one ``(n_types,)`` row per name."""
        return {
            name: torch.tensor([float(t.params[name]) for t in types], dtype=torch.float64)
            for name in cls.params
        }

    def term_energy(self, x: torch.Tensor, **params: torch.Tensor) -> torch.Tensor:
        """Energy of each term from its gathered atoms.

        Args:
            x: ``(M, width, 3)`` Å, in the stored atom order.
            **params: One row per term, named by :attr:`params`.

        Returns:
            ``(M,)`` in the force field's energy unit.
        """
        raise NotImplementedError

    def index(self, frame: TensorDictBase) -> torch.Tensor:
        """``(width, M)`` atom indices of this style's namespace."""
        block = frame[self.namespace]
        if self.namespace == "bonds" and "bond_index" in block.keys():
            return block["bond_index"].long()
        return torch.stack([block[column].long() for column in _INDEX_COLUMNS[: self.width]])

    def values(
        self, frame: TensorDictBase, params: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """The per-term parameters: the passed ones, else the table's rows."""
        if params:
            missing = [name for name in self.params if name not in params]
            if missing:
                raise KeyError(f"{self.style}: parameters {missing} were not passed")
            return {name: params[name] for name in self.params}
        if not self.has_table:
            raise ValueError(
                f"{self.style}: no parameters were passed and the style declares no types"
            )
        # The collate spells a bond's type ``bond_types`` and every other term's ``type``.
        column = "bond_types" if self.namespace == "bonds" else "type"
        type_id = frame[self.namespace, column].long()
        return {name: getattr(self, f"table_{name}")[type_id] for name in self.params}

    def forward(self, frame: TensorDictBase, **params: torch.Tensor) -> torch.Tensor:
        """Energy per graph ``(B,)``.

        Args:
            frame: Batch with ``atoms.pos``, ``atoms.batch``, ``graphs`` and this
                style's namespace (``mask`` and ``type`` optional).
            **params: Per-term parameters; omitted to use the carried table.
        """
        index = self.index(frame)
        pos = frame["atoms", "pos"]
        values = self.values(frame, params)
        energy = self.term_energy(pos[index.T], **{k: v.to(pos.dtype) for k, v in values.items()})
        block = frame[self.namespace]
        if "mask" in block.keys():
            energy = energy * block["mask"].to(energy.dtype)
        graph = frame["atoms", "batch"].long()[index[0]]
        return energy.new_zeros(_graph_count(frame)).index_add(0, graph, energy)


class PairPotential(nn.Module):
    """A style priced over the cutoff pair list.

    Reads ``pairs.pair_index`` ``(E, 2)``, the lattice remainder
    (``pairs.shifts``, else ``pairs.pair_diff``) and ``pairs.relation``
    (``0`` none, ``1``/``2``/``3`` = 1-2/1-3/1-4). Each real pair counts once
    (``src < tgt``; a periodic self-image ``(i, i, s ≠ 0)`` is dropped by that
    rule — these styles price gas-phase molecules):
    only the ``source < target`` orientation of the bidirectional list is
    live, which also kills the padding self-loops. A pair at relation ``c`` is
    weighted by the force field's ``special_bonds`` entry for that class.

    Args:
        cutoff: Radius beyond which a pair contributes nothing, Å.
        special: ``(w12, w13, w14)`` — this style's ``special_bonds`` weights.
    """

    style: ClassVar[str]

    def __init__(self, *, cutoff: float, special: Sequence[float]) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError(f"{self.style}: cutoff must be positive, got {cutoff}")
        if len(special) != 3:
            raise ValueError(f"{self.style}: special must be (w12, w13, w14), got {special}")
        self.cutoff = float(cutoff)
        self.register_buffer(
            "special",
            torch.tensor([1.0, *(float(w) for w in special)], dtype=torch.float64),
            persistent=False,
        )

    @classmethod
    def special_bonds(cls, forcefield: Any) -> tuple[Sequence[float], Sequence[float]]:
        """The force field's ``(lj, coul)`` special-bond triples, each ``(w12, w13, w14)``.

        Raises:
            ValueError: The force field declares no ``special_bonds``; a pair
                style cannot be priced without them (molpy leaves them unset
                rather than assuming a default).
        """
        declared = forcefield.special_bonds
        if declared is None:
            raise ValueError(
                f"{cls.style}: the force field declares no special_bonds "
                "(ForceField.set_special_bonds)"
            )
        return declared

    def geometry(self, frame: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(pair_index, r, weight)`` at the live positions.

        ``weight`` is the special-bonds weight on live in-range rows and ``0``
        elsewhere; ``r`` is replaced on rows outside the range so no ``1/r``
        there is singular.
        """
        pos = frame["atoms", "pos"]
        pairs = frame["pairs"]
        pair_index = pairs["pair_index"].long()
        src, tgt = pair_index[:, 0], pair_index[:, 1]
        vectors = pos[tgt] - pos[src]
        if "shifts" in pairs.keys():
            vectors = vectors + pairs["shifts"].to(pos.dtype)
        elif "pair_diff" in pairs.keys():
            vectors = vectors + (pairs["pair_diff"].to(pos.dtype) - vectors).detach()
        r = (vectors * vectors).sum(-1).clamp_min(1e-24).sqrt()
        inside = (src < tgt) & (r < self.cutoff)
        # Required, not defaulted: without it every bonded pair would be priced
        # at full strength, a different force field computed in silence.
        relation = pairs["relation"].long()
        weight = self.special.to(pos.dtype)[relation] * inside.to(pos.dtype)
        r = torch.where(inside, r, r.new_full((), self.cutoff))
        return pair_index, r, weight

    def reduce(
        self, frame: TensorDictBase, pair_index: torch.Tensor, per_pair: torch.Tensor
    ) -> torch.Tensor:
        """Sum per-pair energies into ``(B,)`` by the source atom's graph."""
        graph = frame["atoms", "batch"].long()[pair_index[:, 0]]
        return per_pair.new_zeros(_graph_count(frame)).index_add(0, graph, per_pair)
