"""The pair enumeration behind stage 1: dedup, the omitted 1-4, and the gather.

The enumeration decides which quantity the fit compares, so it is asserted
directly rather than through the loss. The fixture is the butane-shaped chain
``0-1-2-3``: three bonds, two angles, one proper — the smallest topology in which
a pair is shared between term types and a proper has a 1-4 pair to omit.
"""

from __future__ import annotations

import pytest
import torch

from molpot.forcefield.partial_hessian import PartialHessianPairs, TermHessian

#: ``0-1-2-3``, as the COO index columns of each term type.
CHAIN: dict[str, torch.Tensor] = {
    "bonds": torch.tensor([[0, 1, 2], [1, 2, 3]]),
    "angles": torch.tensor([[0, 1], [1, 2], [2, 3]]),
    "propers": torch.tensor([[0], [1], [2], [3]]),
}
#: Every ordered pair of the chain except the 1-4 pair ``(0, 3)``.
CHAIN_PAIRS: int = 10


def _labelled_hessian(n_atoms: int) -> torch.Tensor:
    """A Hessian whose every entry names its own position.

    ``H[3a + c, 3b + d] = 100a + 10b + 3c + d``, so a gathered block is readable
    by eye and a transposed pair or a swapped Cartesian axis cannot pass.
    """
    atom = torch.arange(n_atoms).repeat_interleave(3)
    axis = torch.arange(3).repeat(n_atoms)
    rows = (100 * atom + 3 * axis).unsqueeze(1)
    cols = (10 * atom + axis).unsqueeze(0)
    return (rows + cols).to(torch.float64)


class TestPartialHessianPairs:
    """What the enumeration promises: one slot per pair, and the 1-4 left out."""

    def test_a_pair_shared_by_two_term_types_gets_one_slot(self) -> None:
        """The MM block for a pair is a sum over terms, so the slot must be shared.

        ``(1, 2)`` is a bond, sits in both angles and in the proper. All four
        occurrences have to name the same slot, or the accumulation would compare
        each term's contribution against the whole quantum block separately.
        """
        layout = PartialHessianPairs(CHAIN)
        assert len(layout) == CHAIN_PAIRS
        bond_slot = int(layout.slots["bonds"][(0, 1)][1])  # bond (1, 2), i<j
        assert int(layout.slots["angles"][(1, 2)][0]) == bond_slot
        assert int(layout.slots["angles"][(0, 1)][1]) == bond_slot
        assert int(layout.slots["propers"][(1, 2)][0]) == bond_slot
        assert tuple(layout.pairs[bond_slot].tolist()) == (1, 2)

    def test_the_one_four_pair_of_a_proper_is_not_enumerated(self) -> None:
        """ByteFF omits it, so we omit it — otherwise the two fits differ.

        ``(0, 3)`` is reachable only through the proper, so its absence from the
        pair list is the whole observable consequence of the skip.
        """
        layout = PartialHessianPairs(CHAIN)
        pairs = {tuple(pair) for pair in layout.pairs.tolist()}
        assert (0, 3) not in pairs
        assert (3, 0) not in pairs
        assert (0, 2) in pairs  # the 1-3 pair of the same proper is kept
        assert (0, 3) not in layout.slots["propers"]
        assert (3, 0) not in layout.slots["propers"]

    def test_every_term_records_one_slot_per_choice(self) -> None:
        """A rec vector is per-term, so it is as long as that namespace."""
        layout = PartialHessianPairs(CHAIN)
        for namespace, index in CHAIN.items():
            n_terms = index.shape[1]
            for record in layout.slots[namespace].values():
                assert record.shape == (n_terms,)

    def test_gather_takes_the_block_at_the_pair_in_that_order(self) -> None:
        """``(a, b)`` reads rows of ``a`` against columns of ``b``, not the reverse."""
        layout = PartialHessianPairs(CHAIN)
        blocks = layout.gather(_labelled_hessian(4))
        assert blocks.shape == (CHAIN_PAIRS, 9)
        slot = int(layout.slots["bonds"][(0, 1)][1])  # the pair (1, 2)
        expected = torch.tensor(
            [100 * 1 + 10 * 2 + 3 * c + d for c in range(3) for d in range(3)],
            dtype=torch.float64,
        )
        torch.testing.assert_close(blocks[slot], expected)

    def test_an_index_of_the_wrong_arity_raises(self) -> None:
        """An angle index handed in as bonds is a silently wrong label otherwise."""
        with pytest.raises(ValueError, match=r"bonds index must be \(2, n_terms\)"):
            PartialHessianPairs({"bonds": CHAIN["angles"]})

    def test_a_term_naming_an_atom_outside_the_hessian_raises(self) -> None:
        """A topology and a Hessian from different molecules must not gather."""
        layout = PartialHessianPairs(CHAIN)
        with pytest.raises(ValueError, match="covers 3 atoms"):
            layout.gather(_labelled_hessian(3))

    def test_a_molecule_with_no_bonded_terms_enumerates_nothing(self) -> None:
        """A lone ion has no valence pairs; the empty gather keeps the 9 columns."""
        layout = PartialHessianPairs({})
        assert len(layout) == 0
        assert layout.gather(_labelled_hessian(1)).shape == (0, 9)


def _harmonic_bond_block(xi: torch.Tensor, xj: torch.Tensor, k: float, r0: float) -> torch.Tensor:
    """Analytic ``d2E / dx_i dx_j`` of ``E = 1/2 k (r - r0)^2``.

    ``-[k u u^T + (k (r - r0) / r)(I - u u^T)]`` with ``u`` the unit bond vector:
    the stiff part along the bond, the restoring part across it.
    """
    diff = xj - xi
    r = diff.norm()
    u = diff / r
    along = torch.outer(u, u)
    across = torch.eye(3, dtype=xi.dtype) - along
    return -(k * along + (k * (r - r0) / r) * across)


class TestTermHessian:
    """The per-term Hessian of a term-separable energy."""

    def test_a_harmonic_bond_matches_its_analytic_block(self) -> None:
        """The whole method in one case where the answer is known in closed form."""
        k, r0 = 310.0, 1.526
        positions = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], [[0.0, 2.0, 0.0], [1.0, 2.9, 0.3]]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def energy(local: torch.Tensor) -> torch.Tensor:
            diff = local[:, 1] - local[:, 0]
            return (0.5 * k * (diff.norm(dim=-1) - r0) ** 2).sum()

        blocks = TermHessian().blocks(energy, positions)
        assert blocks.shape == (2, 2, 3, 2, 3)
        for term in range(2):
            torch.testing.assert_close(
                blocks[term, 0, :, 1, :],
                _harmonic_bond_block(positions[term, 0], positions[term, 1], k, r0),
            )

    def test_a_term_reading_only_its_own_atoms_gives_no_cross_term_coupling(self) -> None:
        """Separability is the reason a gathered copy works.

        Two terms sharing an atom hold independent copies of it, so the returned
        blocks are each term's own curvature — never one polluted by the other's.
        Here both terms are literally the same atom pair; their blocks must be
        equal, which they cannot be if the shared atom were differentiated once.
        """
        pair = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64)
        positions = torch.stack([pair, pair]).requires_grad_(True)

        def energy(local: torch.Tensor) -> torch.Tensor:
            return (local[:, 1] - local[:, 0]).norm(dim=-1).pow(2).sum()

        blocks = TermHessian().blocks(energy, positions)
        torch.testing.assert_close(blocks[0], blocks[1])

    def test_the_parameters_keep_their_graph_through_the_double_backward(self) -> None:
        """A loss on the blocks has to reach the force constants that made them.

        Geometry is differentiated twice and detached; the parameters are not
        differentiated but must stay attached, or stage 1 would compute a target
        with no gradient and train nothing.
        """
        k = torch.tensor([310.0, 340.0], dtype=torch.float64, requires_grad=True)
        positions = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], [[0.0, 2.0, 0.0], [1.1, 2.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def energy(local: torch.Tensor) -> torch.Tensor:
            diff = local[:, 1] - local[:, 0]
            return (0.5 * k * (diff.norm(dim=-1) - 1.5) ** 2).sum()

        TermHessian().blocks(energy, positions).pow(2).sum().backward()
        assert k.grad is not None
        assert bool(torch.isfinite(k.grad).all())
        assert bool((k.grad != 0).all())

    def test_positions_of_the_wrong_rank_raise(self) -> None:
        """A flat ``(n*w, 3)`` gather is the mistake this guards."""
        with pytest.raises(ValueError, match=r"must be \(n_terms, width, 3\)"):
            TermHessian().blocks(lambda x: x.sum(), torch.zeros(6, 3, requires_grad=True))

    def test_positions_that_do_not_require_grad_raise(self) -> None:
        """It cannot mark them itself: doing so inside a compiled region does not trace.

        Raising here rather than marking is what turns a silent compile failure
        deep in dynamo into a message naming the prologue that should have run.
        """
        with pytest.raises(ValueError, match="does not require grad"):
            TermHessian().blocks(lambda x: x.sum(), torch.zeros(2, 2, 3))
