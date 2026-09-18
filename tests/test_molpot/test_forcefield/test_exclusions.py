"""Topological exclusion table — mirrors ``src/molpot/forcefield/exclusions.py``.

:class:`~molpot.forcefield.exclusions.ExclusionTable` is built **once per
topology** from the bond / angle / proper columns and answers
``lookup(pair_index, kind, dtype=…)`` for a whole fixed-capacity neighbour list
at once.
It replaces ``Potentials._pair_scales`` (deleted with ``composable-a0-02``),
which rebuilt the same ``torch.unique`` keys on every energy call and wrote its
answer through boolean mask assignment — a data-dependent shape, and therefore
a graph break, on the MD force path.

Two properties are pinned here and nowhere else:

1. **Exact behaviour preservation.** The per-kind scale vectors of an OPLS
   butane are hard-coded goldens captured from the *pre-step* implementation.
2. **Fixed shape.** ``lookup`` returns one value per input row, dead padding
   included, so the caller multiplies instead of slicing.

Golden provenance — captured at commit ``29f7f004`` with the then-existing
``Potentials._pair_scales`` (``src/molpot/forcefield/potentials.py``
byte-identical at ``97c36852``), 2026-09-07, ``torch`` 2.10, over the butane
topology of ``conftest.py`` and the full ``triu`` pair list, for both declared
kinds under ``OPLSAAForceField().nonbonded_scaling``. That method was
**deleted in a0-02**, so the capture is not re-runnable and no snippet is
reproduced here for a reader to try: the values are frozen literals in
``tests/test_molpot/test_forcefield/conftest.py``
(:data:`~tests.test_molpot.test_forcefield.conftest.BUTANE_OPLS_SCALES`) and
move only on a new, separately documented capture.
"""

from __future__ import annotations

import pytest
import torch

from molpot.forcefield.exclusions import ExclusionTable
from molpot.ir.scaling import ExclusionScales, NonbondedScaling

DTYPE = torch.float64

#: A scaling whose four possible answers (1-2, 1-3, 1-4, untouched) are all
#: distinct, so a precedence mistake cannot hide behind an equal value.
DISTINCT = NonbondedScaling(
    charge=ExclusionScales(0.125, 0.25, 0.75),
    lj=ExclusionScales(0.375, 0.625, 0.875),
)


def _opls_table(topology: dict[str, torch.Tensor]) -> ExclusionTable:
    """The butane table under ``OPLSAAForceField().nonbonded_scaling``."""
    from molpot.forcefield import OPLSAAForceField

    return ExclusionTable(
        n_atoms=14,
        bond_index=topology["bond_index"],
        angle_index=topology["angle_index"],
        proper_index=topology["proper_index"],
        scaling=OPLSAAForceField().nonbonded_scaling,
    )


def _ring_table() -> ExclusionTable:
    """Cyclopropane carbons — every pair is *both* a bond and an angle end.

    ``0-1``, ``1-2``, ``2-0`` are bonds; the three angles ``0-1-2`` / ``1-2-0``
    / ``2-0-1`` have ends ``(0, 2)`` / ``(1, 0)`` / ``(2, 1)``, i.e. exactly the
    same three unordered pairs. A tree topology can never produce this overlap,
    which is why the 1-2 ≻ 1-3 leg of the precedence rule needs a ring to be
    observable at all.
    """
    return ExclusionTable(
        n_atoms=3,
        bond_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        angle_index=torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=torch.long),
        proper_index=torch.zeros(4, 0, dtype=torch.long),
        scaling=DISTINCT,
    )


class TestExclusionTable:
    def test_butane_lookup_reproduces_the_pre_step_pair_scales(
        self,
        butane_topology: dict[str, torch.Tensor],
        butane_scales: torch.Tensor,
    ) -> None:
        """Hard-coded golden: exact equality with the deleted ``_pair_scales``.

        OPLS-AA scales 1-4 by 0.5 for both ``charge`` and ``lj``, so the two kinds
        answer with the same vector on this topology — the per-kind *keying* is
        pinned separately below.
        """
        table = _opls_table(butane_topology)
        pair_index = butane_topology["pair_index"]
        for kind in ("charge", "lj"):
            scales = table.lookup(pair_index, kind, dtype=DTYPE)
            assert torch.equal(scales, butane_scales), kind

    def test_the_two_kinds_answer_from_their_own_triple(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """``charge`` and ``lj`` are independent columns, not one shared vector.

        Under the Class-I defaults 1-4 electrostatics keep 5/6 while 1-4 LJ
        keeps 1/2; both drop 1-2 and 1-3 and leave everything else at 1.
        """
        table = ExclusionTable(
            n_atoms=14,
            bond_index=butane_topology["bond_index"],
            angle_index=butane_topology["angle_index"],
            proper_index=butane_topology["proper_index"],
            scaling=NonbondedScaling(),
        )
        pair_index = butane_topology["pair_index"]
        q_values = sorted(
            float(v) for v in torch.unique(table.lookup(pair_index, "charge", dtype=DTYPE))
        )
        lj_values = sorted(
            float(v) for v in torch.unique(table.lookup(pair_index, "lj", dtype=DTYPE))
        )
        assert q_values == pytest.approx([0.0, 5.0 / 6.0, 1.0], abs=1e-12)
        assert lj_values == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)

    def test_lookup_output_shape_equals_input_shape_on_a_padded_list(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """Dead padding gets an answer too — capacity in, capacity out.

        A fixed-capacity neighbour list carries dead rows past ``num_pairs``;
        they must not change the output length (that would be a data-dependent
        shape), and a self-pair is in no relation, so it reads 1.
        """
        table = _opls_table(butane_topology)
        live = butane_topology["pair_index"]
        padding = torch.zeros(7, 2, dtype=torch.long)
        padded = torch.cat([live, padding], dim=0)
        scales = table.lookup(padded, "charge", dtype=DTYPE)
        assert scales.shape == (padded.shape[0],)
        assert torch.equal(scales[: live.shape[0]], table.lookup(live, "charge", dtype=DTYPE))
        assert torch.equal(scales[live.shape[0] :], torch.ones(7, dtype=scales.dtype))

    def test_a_pair_that_is_both_one_three_and_one_four_reads_s13(self) -> None:
        """The closer relation wins: ``s14`` lands only where nothing else did.

        Butane is acyclic, so no pair of it is in two relations at once — a
        ring is needed to exercise the precedence rule. Pair ``(0, 2)`` here is
        the angle end of ``0-1-2`` *and* the torsion end of ``0-1-4-2``, so it
        must read ``s13``; ``(0, 4)`` is in no relation and stays at 1. This
        leg the pre-step ``_pair_scales`` already enforced
        (``scale[proper_mask & (scale == 1.0)] = s14``); the 1-2 ≻ 1-3 leg
        below is the one it got wrong.
        """
        table = ExclusionTable(
            n_atoms=6,
            bond_index=torch.zeros(2, 0, dtype=torch.long),
            angle_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
            proper_index=torch.tensor([[0], [1], [4], [2]], dtype=torch.long),
            scaling=DISTINCT,
        )
        pair_index = torch.tensor([[0, 2], [0, 4]], dtype=torch.long)
        scales = table.lookup(pair_index, "charge", dtype=DTYPE)
        assert float(scales[0]) == pytest.approx(0.25, abs=1e-12)
        assert float(scales[1]) == pytest.approx(1.0, abs=1e-12)

    def test_a_ring_pair_that_is_both_bonded_and_an_angle_end_reads_s12(self) -> None:
        """1-2 ≻ 1-3 on a 3-ring, for every declared kind.

        This is a deliberate **fix** of the pre-step ``_pair_scales``, whose
        angle pass overwrote the bond pass unconditionally
        (``scale[angle_mask] = s13`` after ``scale[bond_mask] = s12``). On a
        tree the two sets are disjoint, so butane cannot see it; on
        cyclopropane every C-C pair is both, and the old code answered ``s13``.
        The physical rule is LAMMPS ``special_bonds``: a pair already in a
        closer list is not re-listed in a farther one, so the *closest*
        relation sets the factor. ``charge`` and ``lj`` are checked together because
        the precedence is a property of the table, not of one triple.
        """
        table = _ring_table()
        pair_index = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
        for kind, s12 in (("charge", 0.125), ("lj", 0.375)):
            scales = table.lookup(pair_index, kind, dtype=DTYPE)
            assert scales.shape == (3,)
            assert [float(v) for v in scales] == pytest.approx([s12] * 3, abs=1e-12), kind

    def test_pairs_is_the_topology_fixed_scaled_pair_list(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """13 + 24 + 27 undirected pairs — butane is acyclic, so none overlap.

        ``pairs`` is what ``composable-a0-03`` sums the periodic exclusion
        correction over, so it is the *union* of the three relations with each
        unordered pair present once, ``(P, 2)`` with ``source < target``.
        """
        table = _opls_table(butane_topology)
        pairs = table.pairs
        assert pairs.shape == (13 + 24 + 27, 2)
        assert bool((pairs[:, 0] < pairs[:, 1]).all())
        keys = pairs[:, 0] * 14 + pairs[:, 1]
        assert int(torch.unique(keys).numel()) == pairs.shape[0]
        # every scaled pair is exactly the complement of the untouched ones
        scales = table.lookup(butane_topology["pair_index"], "charge", dtype=DTYPE)
        assert int((scales != 1.0).sum()) == pairs.shape[0]

    def test_pairs_deduplicates_a_pair_that_is_in_two_relations(self) -> None:
        """Three unordered pairs on the 3-ring, not six — the union, not the sum.

        Cyclopropane contributes each C-C pair once as a bond and once as an
        angle end. ``composable-a0-03`` sums the periodic exclusion correction
        over ``pairs``, so a duplicated row there is a doubled correction term
        and a silently wrong electrostatic energy under PBC.
        """
        pairs = _ring_table().pairs
        assert pairs.shape == (3, 2)
        assert bool((pairs[:, 0] < pairs[:, 1]).all())
        keys = sorted(int(row[0]) * 3 + int(row[1]) for row in pairs)
        assert keys == [1, 2, 5]  # (0,1), (0,2), (1,2)

    def test_table_length_follows_the_column_shapes_not_their_contents(self) -> None:
        """Two topologies of one padded shape give one table length — the compile gate.

        How many pairs survive the dedup is a property of the molecules drawn,
        so an exactly-sized table changes length from one packed batch to the
        next; its length is a dynamo guard on every energy call, which is one
        recompile per batch on a padded training path (GH200 job 2462406).
        Same column widths in, same ``searchsorted`` table out.
        """
        # Same widths (3 bonds, 2 angles), different content: a 3-ring, where
        # every angle end is also a bond, and a chain, where none is.
        ring = ExclusionTable(
            n_atoms=6,
            bond_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            angle_index=torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long),
            scaling=DISTINCT,
        )
        chain = ExclusionTable(
            n_atoms=6,
            bond_index=torch.tensor([[0, 1, 2], [1, 2, 3], [0, 0, 0]], dtype=torch.long)[:2],
            angle_index=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
            scaling=DISTINCT,
        )
        assert ring.pairs.shape[0] != chain.pairs.shape[0]  # the fixture is not degenerate
        assert ring._keys.shape == chain._keys.shape == (5,)
        assert ring._relation.shape == chain._relation.shape == (5,)

    def test_padded_slots_never_match_a_query(self) -> None:
        """A pad slot answers the unscaled ``1.0``, exactly as an unlisted pair does."""
        table = ExclusionTable(
            n_atoms=6,
            bond_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            angle_index=torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long),
            scaling=DISTINCT,
        )
        # 3 unique pairs in a 5-slot table: two slots are pad.
        assert table.pairs.shape[0] == 3 and table._keys.shape[0] == 5
        # (4, 5) is in no relation; the query must not land on a pad slot.
        unlisted = torch.tensor([[4, 5], [5, 4]], dtype=torch.long)
        assert torch.equal(
            table.lookup(unlisted, "charge", dtype=DTYPE), torch.ones(2, dtype=DTYPE)
        )
        listed = torch.tensor([[0, 1]], dtype=torch.long)
        assert float(table.lookup(listed, "charge", dtype=DTYPE)[0]) != 1.0

    def test_unknown_kind_raises_listing_the_declared_kinds(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """``induced`` arrives with ``composable-a1``; until then, a loud miss."""
        table = _opls_table(butane_topology)
        with pytest.raises(KeyError) as excinfo:
            table.lookup(butane_topology["pair_index"], "induced", dtype=DTYPE)
        message = str(excinfo.value)
        assert "induced" in message
        assert "charge" in message
        assert "lj" in message

    def test_lookup_takes_its_precision_from_the_caller_not_from_the_table(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """``dtype`` is keyword-**required**: the query names its working precision.

        The table stores its triples in ``float64`` so a repeating factor such
        as 5/6 is not rounded at build time, but the answer is multiplied into
        a pair term whose distances carry the *caller's* precision. An optional
        ``dtype`` let an fp32 force path silently receive an fp64 scale vector
        and promote the whole product — one dtype seam that neither the caller
        nor the table declared, and on a compiled hot path a dtype promotion is
        a real recompile, not a cosmetic one. Making it required moves that
        decision to the single site that knows the answer (``_scaled_pairs``
        passes ``pos.dtype``).

        Both legs belong to the one rule: the named dtype is honoured, and an
        unnamed one is refused rather than defaulted.
        """
        table = _opls_table(butane_topology)
        pair_index = butane_topology["pair_index"]
        assert table.lookup(pair_index, "charge", dtype=torch.float32).dtype is torch.float32
        assert table.lookup(pair_index, "charge", dtype=DTYPE).dtype is DTYPE
        with pytest.raises(TypeError):
            table.lookup(pair_index, "charge")  # type: ignore[call-arg]

    @pytest.mark.slow
    def test_lookup_has_no_graph_breaks_on_a_padded_list(
        self, butane_topology: dict[str, torch.Tensor]
    ) -> None:
        """``searchsorted`` + ``torch.where``, no boolean indexing, no host sync."""
        from molix.compile import Compiler

        table = _opls_table(butane_topology)
        padded = torch.cat(
            [butane_topology["pair_index"], torch.zeros(7, 2, dtype=torch.long)], dim=0
        )
        explanation = Compiler.explain(table.lookup, padded, "charge", dtype=DTYPE)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)
