"""The bond list indexed by endpoint pair, so a valence term can read its own bonds' features.

An angle ``i-j-k`` is two bonds, a proper ``i-j-k-l`` three, an improper
``c-(i,j,k)`` the three bonds at its centre. ByteFF interleaves the features
of those bonds with the atom features before the term's MLP
(``layers.py:486-510``, ``OutputWithBond``; the per-term bond slots are data,
``mol_data.py:119-122``). A context that reads atom features only has to
decode the central bond order of a torsion from a two-hop atom summary, and
on a vertex-transitive graph it cannot at all: in a Kekulé ring the torsion
about a single bond and the one about a double bond get identical features
while the two bond features differ.

The table is built once per batch from the bond endpoints on the device —
one sort of the pair keys — and answers a lookup with ``searchsorted`` on
static shapes: no host sync, no data-dependent shape, nothing the loader has
to write. A pair that is not a bond (a padded ghost row's, or a malformed
term's) gathers a zero feature rather than raising, because raising on a
tensor value is a host sync; the contexts zero padded rows anyway.
"""

from __future__ import annotations

import torch

__all__ = ["BondTable"]


class BondTable:
    """Bond rows by unordered endpoint pair.

    Args:
        atomi: ``(E,)`` first endpoint of every bond row.
        atomj: ``(E,)`` second endpoint.
        n_atoms: Atom rows ``N`` (a static shape, never read off a tensor);
            the pair key is ``min · N + max``.
    """

    def __init__(self, atomi: torch.Tensor, atomj: torch.Tensor, n_atoms: int) -> None:
        if atomi.shape != atomj.shape or atomi.dim() != 1:
            raise ValueError(
                f"atomi and atomj must be matching (E,) columns, got {tuple(atomi.shape)} "
                f"and {tuple(atomj.shape)}"
            )
        self.n_atoms = int(n_atoms)
        self.keys, self.order = torch.sort(self._key(atomi, atomj))

    def _key(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.long()
        b = b.long()
        return torch.minimum(a, b) * self.n_atoms + torch.maximum(a, b)

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def rows(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The bond row of each pair ``(a, b)`` and whether the pair is a bond.

        Args:
            a: ``(M,)`` atom indices.
            b: ``(M,)`` atom indices.

        Returns:
            ``row`` ``(M,)`` long — a valid row of the bond list for every
            pair (row 0 where the pair is not a bond) — and ``found`` ``(M,)``
            bool.
        """
        query = self._key(a, b)
        if len(self) == 0:
            return torch.zeros_like(query), torch.zeros_like(query, dtype=torch.bool)
        slot = torch.searchsorted(self.keys, query).clamp(max=len(self) - 1)
        return self.order[slot], self.keys[slot] == query

    def gather(self, h_bond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``(M, D)`` bond features of the pairs ``(a, b)``; zero where the pair is not a bond.

        Args:
            h_bond: ``(E, D)`` bond features, one row per bond of the table.
            a: ``(M,)`` atom indices.
            b: ``(M,)`` atom indices.
        """
        if int(h_bond.shape[0]) != len(self):
            raise ValueError(f"h_bond has {int(h_bond.shape[0])} rows, the table {len(self)} bonds")
        if len(self) == 0:
            return h_bond.new_zeros((int(a.shape[0]), int(h_bond.shape[1])))
        row, found = self.rows(a, b)
        return h_bond[row] * found.to(h_bond.dtype).unsqueeze(-1)
