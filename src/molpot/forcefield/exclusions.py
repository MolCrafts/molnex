"""Topological exclusion table for the Class-I nonbonded path.

A Class-I force field (the classical diagonal form: harmonic bonds and angles,
periodic torsions, Lennard-Jones and Coulomb pairs) does not evaluate the
nonbonded terms over every pair. Pairs joined by one, two or three covalent
bonds (1-2, 1-3, 1-4) are removed or damped, because the bonded terms already
describe them. Which factor applies is a property of the **topology** — which
atom is bonded to which, the connectivity a molecule keeps whatever its
coordinates are — not of the geometry, so
:class:`ExclusionTable` is built once per topology
(:meth:`molpot.forcefield.Potentials.bind_topology`) and then answers a whole
fixed-capacity neighbour list per call.

Precedence — the closest relation wins
--------------------------------------

A pair can sit in more than one list at once: on a three-membered ring the
carbons of every C-C bond are also the two ends of an angle, and a 1-4 torsion
end is frequently also a 1-3 angle end. LAMMPS ``special_bonds`` resolves this
by *not re-listing* a pair that a closer list already holds, so the ranking is

    1-2 (bond) ≻ 1-3 (angle end) ≻ 1-4 (torsion end) ≻ unlisted (factor 1).

The table therefore keeps exactly one relation code per unordered pair, the
smallest one, and a pair in no relation reads ``1.0``.

Shape discipline
----------------

:meth:`ExclusionTable.lookup` is on the molecular-dynamics (MD) force path —
the code that runs once per integration step — so it answers **one
value per input row** — dead padding rows of a fixed-capacity list included.
``torch.unique`` and every other data-dependent shape happen once at
construction; the query is ``searchsorted`` + ``torch.where``, with no boolean
indexing, no host sync and no Python branch on a tensor value, so it traces
under ``torch.compile`` with zero graph breaks.

The query table's own **length** is likewise a function of the input column
*shapes*, never of their contents: it is padded to one slot per bonded column
with a sentinel key no query can match. A capacity-padded training batch
(:class:`molix.data.MoleculePadder`) therefore rebinds to the same table
length every batch, where an exactly-sized table would change length with the
molecules drawn and recompile the step once per batch. :attr:`pairs` is the
exception and stays exact — it is the topology's real pair list, read by the
Ewald correction on the single-topology periodic path, not by the packed
training path. A *graph break* is a point
where ``torch.compile`` cannot keep building its traced graph — because an
output shape depends on the data, or a value has to be read back to Python —
and has to fall back to the eager interpreter, which costs most of the
speed-up the compiled region was for.

References:
    LAMMPS ``special_bonds`` command (exclusion / scaling semantics)
    https://docs.lammps.org/special_bonds.html
    Cornell et al. "A Second Generation Force Field …" JACS 117, 5179 (1995)
    https://doi.org/10.1021/ja00124a002
"""

from __future__ import annotations

import torch

from molpot.ir.scaling import NonbondedScaling

__all__ = ["ExclusionTable"]

#: Relation codes, ordered by closeness — the index into an
#: :class:`~molpot.ir.scaling.ExclusionScales` triple ``(s12, s13, s14)``. The
#: numeric order *is* the precedence: the smallest code of a pair wins.
_BONDED, _ANGLE_END, _TORSION_END = 0, 1, 2

#: Key stored in every slot no relation claims — an empty topology's whole
#: table, and the pad that holds a table's length to its shape-derived
#: capacity. Pair keys are ``lo * n_atoms + hi ≥ 0``, so this one can never be
#: matched, which is what lets :meth:`ExclusionTable.lookup` keep a single
#: branch-free body instead of testing an empty table on every call.
_NO_MATCH_KEY = -1


class ExclusionTable:
    """Per-interaction-class 1-2 / 1-3 / 1-4 scale factors of one topology.

    Built once per topology and reused for every energy call on it; a changed
    topology — a different molecule, or a batch whose bond / angle / torsion
    columns differ — needs a **new** table, the way a changed lattice needs a
    new :meth:`~molix.schema.CellBound.bind_cell`. Nothing here is sized by
    geometry, so the table survives every position update, barostat step and
    neighbour-list rebuild of that topology, and a fresh batch of the *same*
    topology keeps it too.

    Nothing rebuilds a table implicitly:
    :meth:`~molpot.forcefield.Potentials.bind_topology` records a shape-level
    fingerprint of the topology it was built from and every later energy call
    compares it, raising rather than scaling a new molecule's pairs by the
    previous molecule's relations.

    Not an ``nn.Module``: it holds no parameter and no buffer a checkpoint
    should carry — the topology is an input, not a learned state.

    A relation is read off the *ends* of each bonded term: a bond ``(i, j)``,
    an angle ``(i, j, k)`` → ``(i, k)``, a proper torsion ``(i, j, k, l)`` →
    ``(i, l)``. Ends are stored as the undirected key ``lo * n_atoms + hi``
    with ``lo < hi``, so a padded column whose two ends coincide contributes
    nothing and self-pairs read the unscaled ``1.0``.

    Args:
        n_atoms: Number of atoms ``N`` the indices below address. Fixes the
            key encoding, so it must be the count the queried ``pair_index``
            is numbered against.
        scaling: Declared interaction classes and their dimensionless
            ``(s12, s13, s14)`` triples.
            :meth:`~molpot.ir.scaling.NonbondedScaling.kinds` is the key set
            :meth:`lookup` accepts.
        bond_index: Bond columns ``(2, Nb)`` — ``None`` or an empty column
            block for a topology with no bond.
        angle_index: Angle columns ``(3, Na)``; ends are rows 0 and 2.
        proper_index: Proper-torsion columns ``(4, Np)``; ends are rows 0
            and 3.
        device: Device the table is built on, and therefore the device
            :meth:`lookup` expects its queries on. Defaults to the device of
            the first index block given (CPU when none is).
    """

    def __init__(
        self,
        *,
        n_atoms: int,
        scaling: NonbondedScaling,
        bond_index: torch.Tensor | None = None,
        angle_index: torch.Tensor | None = None,
        proper_index: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._n_atoms = int(n_atoms)
        columns = (
            (_BONDED, bond_index),
            (_ANGLE_END, angle_index),
            (_TORSION_END, proper_index),
        )
        if device is None:
            device = next(
                (index.device for _, index in columns if index is not None),
                torch.device("cpu"),
            )

        keys: list[torch.Tensor] = []
        relations: list[torch.Tensor] = []
        capacity = 0
        for code, index in columns:
            if index is None or index.numel() == 0:
                continue
            ends = index.to(device=device, dtype=torch.long)
            # One key per column at most, whatever the column holds: the query
            # table's length is a function of the input *shapes* alone.
            capacity += int(ends.shape[-1])
            lo = torch.minimum(ends[0], ends[-1])
            hi = torch.maximum(ends[0], ends[-1])
            # ``lo == hi`` is a degenerate or padded column, never a relation.
            keep = lo < hi
            keys.append((lo * self._n_atoms + hi)[keep])
            relations.append(torch.full_like(keys[-1], code))

        empty = torch.zeros(0, dtype=torch.long, device=device)
        listed = torch.cat(keys) if keys else empty
        codes = torch.cat(relations) if relations else empty
        # ``unique`` (and the scatter below) is the whole data-dependent part of
        # this class, and it runs here — once per topology — not per call.
        unique_keys, inverse = torch.unique(listed, return_inverse=True)
        closest = torch.zeros(unique_keys.shape[0], dtype=torch.long, device=device)
        closest.scatter_reduce_(0, inverse, codes, reduce="amin", include_self=False)

        self._pairs = torch.stack(
            [unique_keys // self._n_atoms, unique_keys % self._n_atoms], dim=1
        )
        # Pad the query table to that shape-derived capacity with the sentinel.
        # How many pairs *survive* the dedup is a property of the molecules in
        # the batch, so an exact table changes length from one packed batch to
        # the next — and its length is a dynamo guard on every energy call
        # ("_keys size mismatch: expected 32990, actual 30097", GH200 job
        # 2462406), which is one recompile per batch on a padded training path.
        # ``_NO_MATCH_KEY`` is negative and every real key is ``>= 0``, so the
        # pad sorts ahead of the table, ``searchsorted`` is unaffected, and the
        # equality in :meth:`lookup` can never select a pad slot.
        n_pad = max(capacity, 1) - int(unique_keys.shape[0])
        if n_pad > 0:
            unique_keys = torch.cat(
                [torch.full((n_pad,), _NO_MATCH_KEY, dtype=torch.long, device=device), unique_keys]
            )
            closest = torch.cat([torch.zeros(n_pad, dtype=torch.long, device=device), closest])
        self._keys = unique_keys
        self._relation = closest
        # One ``(3,)`` row per declared class, indexed by relation code. fp64 so
        # a repeating factor such as 5/6 is not rounded before the caller picks
        # its own working precision.
        self._scales = {
            kind: torch.tensor(scaling.triple(kind), dtype=torch.float64, device=device)
            for kind in scaling.kinds()
        }

    @property
    def pairs(self) -> torch.Tensor:
        """Every scaled pair of the topology, ``(P, 2)`` with ``lo < hi``.

        The **union** of the three relations with each unordered pair present
        exactly once — a pair that is both a bond and an angle end appears as
        one row, under whichever relation won the precedence. Sorted by
        ``lo * n_atoms + hi``, so ``(lo, hi)`` ascends lexicographically.

        Returns:
            Pair endpoints ``(P, 2)``, ``torch.long``. Empty ``(0, 2)`` when
            the topology excludes nothing.
        """
        return self._pairs

    def lookup(
        self,
        pair_index: torch.Tensor,
        kind: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Scale factor of every listed pair, one value per row.

        Fixed shape in, fixed shape out: a dead padding row of a
        capacity-shaped neighbour list is answered (with ``1.0``) rather than
        dropped, so the caller multiplies instead of slicing.

        Args:
            pair_index: Pairs ``(E, 2)``, ``torch.long``, in either order —
                the table is undirected. A self-pair ``(i, i)`` is in no
                relation and reads ``1.0``.
            kind: Interaction class, one of the ``scaling.kinds()`` the table
                was built with (``"charge"``, ``"lj"``).
            dtype: Working precision of the result, **required**. The table
                stores its triples in ``float64`` so a repeating factor such as
                5/6 is not rounded at build time, but the answer is multiplied
                into a pair term carrying the caller's precision — a default
                here would hand an fp32 force path an fp64 vector and promote
                the whole product, which on a compiled path is a recompile.
                The single site that knows the answer names it
                (:meth:`~molpot.forcefield.Potentials._scaled_pairs` passes
                ``pos.dtype``).

        Returns:
            Scale factors ``(E,)``: ``s12`` / ``s13`` / ``s14`` for a pair in
            the 1-2 / 1-3 / 1-4 relation, ``1.0`` for a pair in none.

        Raises:
            KeyError: ``kind`` is not a declared interaction class. Never
                falls back to an unscaled ``1.0``, which is how an unmodelled
                class would escape its exclusions unnoticed.
        """
        scales = self._scales.get(kind)
        if scales is None:
            raise KeyError(
                f"Unknown interaction class {kind!r}; declared kinds are {tuple(self._scales)}"
            )
        scales = scales.to(dtype)
        lo = torch.minimum(pair_index[:, 0], pair_index[:, 1])
        hi = torch.maximum(pair_index[:, 0], pair_index[:, 1])
        query = lo * self._n_atoms + hi
        # ``clamp`` keeps the gather in range for a query past the last key;
        # the equality below is what decides whether the slot actually matched.
        slot = torch.searchsorted(self._keys, query).clamp(max=self._keys.shape[0] - 1)
        found = self._keys[slot] == query
        return torch.where(found, scales[self._relation[slot]], scales.new_ones(()))

    def __repr__(self) -> str:
        return (
            f"ExclusionTable(n_atoms={self._n_atoms}, pairs={self._pairs.shape[0]}, "
            f"kinds={tuple(self._scales)})"
        )
