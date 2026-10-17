"""Each cutoff pair's covalent relation, written once per batch on the host.

A Class-I force field scales its nonbonded terms by how two atoms are bonded:
directly (1-2), through one atom (1-3, the ends of an angle), or through two
(1-4, the ends of a proper torsion). Which of those a pair is, is a property of
the topology, not of the positions, so it is computed here — in the loader,
beside :class:`~molix.neighbor.NeighborList` — and the model reads it as data.
The model then does no host work per batch and the scale policy is a
``special_bonds`` setting of the force field the pair potential is built from
(:class:`molpot.potentials.potential.PairPotential`).

The closest relation wins (LAMMPS ``special_bonds``): on a three-membered ring
a bonded pair is also the two ends of an angle, and it is scaled as 1-2.

Reference:
    LAMMPS ``special_bonds``, https://docs.lammps.org/special_bonds.html
"""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.task import BatchTask

__all__ = ["PairRelation"]

#: ``(namespace, first column, last column, code)``; the code order is the
#: precedence — the smallest code of a pair wins.
_ENDS: tuple[tuple[str, str, str, int], ...] = (
    ("bonds", "atomi", "atomj", 1),
    ("angles", "atomi", "atomk", 2),
    ("propers", "atomi", "atoml", 3),
)


class PairRelation(BatchTask):
    """Write ``pairs.relation``: ``0`` none, ``1`` / ``2`` / ``3`` for 1-2 / 1-3 / 1-4.

    Mount after the neighbour list, which writes ``pairs.pair_index``. The
    column is ``int8`` and one entry per pair row; the padder zero-fills it,
    so dead rows read ``0``.

    Example::

        DataModule(..., batch_tasks=[neighbor_list, PairRelation(), TotalCharge()])

    A pair's relation is fixed by its molecule's topology and its own edges, so
    :meth:`~molix.data.prepared.PreparedRow.capture` may run this task per row
    and the loader then skips it for rows captured with it; rebuilt beside the
    cutoff graph it is 0.14 s a packed batch, vectorised.
    """

    def execute(self, data: TensorDict | dict) -> TensorDict | dict:
        """Compute the relation of every row of ``pairs.pair_index``.

        Args:
            data: Collated batch with ``atoms``, ``pairs.pair_index`` ``(E, 2)``
                and whichever of ``bonds`` (``atomi``/``atomj`` or
                ``bond_index`` ``(2, Nb)``), ``angles`` and ``propers`` it has.

        Returns:
            The same batch, with ``pairs.relation`` ``(E,)`` ``int8``.

        Raises:
            TypeError: *data* is not a ``TensorDict``.
        """
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                "PairRelation writes pairs.relation into the batch it is handed; "
                f"it needs the collated TensorDict, got {type(data).__name__}"
            )
        pair_index = data["pairs", "pair_index"].long()
        n = int(data["atoms"].batch_size[0])
        keys: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        for namespace, first, last, code in _ENDS:
            ends = self._ends(data, namespace, first, last)
            if ends is None:
                continue
            lo = torch.minimum(ends[0], ends[1])
            hi = torch.maximum(ends[0], ends[1])
            keep = lo < hi
            keys.append((lo * n + hi)[keep])
            codes.append(torch.full_like(keys[-1], code))
        relation = torch.zeros(pair_index.shape[0], dtype=torch.int8, device=pair_index.device)
        if keys:
            listed = torch.cat(keys)
            unique, inverse = torch.unique(listed, return_inverse=True)
            closest = torch.full_like(unique, 4)
            closest.scatter_reduce_(0, inverse, torch.cat(codes), reduce="amin")
            lo = torch.minimum(pair_index[:, 0], pair_index[:, 1])
            hi = torch.maximum(pair_index[:, 0], pair_index[:, 1])
            query = lo * n + hi
            if unique.numel():
                slot = torch.searchsorted(unique, query).clamp(max=unique.numel() - 1)
                found = unique[slot] == query
                relation = torch.where(found, closest[slot], 0).to(torch.int8)
        data["pairs", "relation"] = relation
        return data

    @staticmethod
    def _ends(data: TensorDictBase, namespace: str, first: str, last: str) -> torch.Tensor | None:
        """``(2, M)`` end atoms of a valence namespace, or ``None`` when absent."""
        if namespace not in data.keys():
            return None
        block = data[namespace]
        if first in block.keys() and last in block.keys():
            return torch.stack([block[first].long(), block[last].long()])
        if namespace == "bonds" and "bond_index" in block.keys():
            return block["bond_index"].long()
        return None
