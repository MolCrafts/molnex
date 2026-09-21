"""Which atom pairs a Class-I bonded term couples, and the Hessian blocks on them.

A *partial Hessian* is the set of 3x3 mixed-derivative blocks
``d2E / dx_a dx_b`` for the atom pairs ``(a, b)`` that share a bonded term. It is
the target of ByteFF's first fitting stage: the molecular-mechanics blocks are
compared against the quantum blocks pair by pair, which constrains the harmonic
force constants without ever forming the full ``(3N, 3N)`` Hessian.

**Why the pairs must be deduplicated.** One atom pair can sit in several terms at
once — a 1-2 pair is in a bond and in every angle and torsion that contains it —
and the MM block for that pair is the **sum** of every term's contribution.
Comparing term by term would compare a different quantity, so the pairs are
enumerated once, each term records which slot its ``(i, j)`` belongs to, and the
MM contributions are accumulated into those slots before the comparison.

**Which pairs.** Every ordered pair of distinct positions within a term, except
``|i - j| >= 3`` in a proper torsion — the 1-4 pair. ByteFF omits it and says why:
"In 4-members rings, 1-4 interaction of proper torsions should be taken into
consideration. We omit this term for simplicity, which could result in an error of
~1 kcal/mol/A^2." This module keeps that choice so the two fits see the same
target, and states it here rather than leaving it as an unexplained bound.

Ordered, not unordered: ``(a, b)`` and ``(b, a)`` are separate slots. The full
Hessian is symmetric, so the blocks are transposes of each other, but keeping both
lets a term's contribution be scattered without a per-pair orientation test.

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." *Chem. Sci.* (2025).
    https://doi.org/10.48550/arXiv.2408.12817
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import torch

__all__ = ["PartialHessianPairs", "TermHessian", "TERM_WIDTHS"]

#: Atoms each Class-I bonded term couples, by batch namespace.
TERM_WIDTHS: dict[str, int] = {
    "bonds": 2,
    "angles": 3,
    "propers": 4,
    "impropers": 4,
}


class PartialHessianPairs:
    """The deduplicated atom pairs of a molecule's bonded terms, and their slots.

    Args:
        indices: Per namespace, the ``(width, n_terms)`` COO atom indices of that
            term type. A namespace may be absent or empty.

    Attributes:
        pairs: ``(n_pairs, 2)`` ordered atom pairs, in first-seen order.
        slots: ``{namespace: {(i, j): (n_terms,) slot index}}`` — where each
            term's ``(i, j)`` block accumulates.

    Example::

        layout = PartialHessianPairs({"bonds": bond_index, "angles": angle_index})
        blocks = layout.gather(hessian)      # (n_pairs, 9) QM label
    """

    def __init__(self, indices: dict[str, torch.Tensor]) -> None:
        pair_slot: dict[tuple[int, int], int] = {}
        pairs: list[tuple[int, int]] = []
        slots: dict[str, dict[tuple[int, int], torch.Tensor]] = {}

        for namespace, width in TERM_WIDTHS.items():
            index = indices.get(namespace)
            if index is None or index.numel() == 0:
                continue
            if index.shape[0] != width:
                raise ValueError(
                    f"{namespace} index must be ({width}, n_terms); got {tuple(index.shape)}"
                )
            terms = index.T.tolist()
            per_choice: dict[tuple[int, int], list[int]] = {}
            for i, j in permutations(range(width), 2):
                if namespace == "propers" and abs(i - j) >= 3:
                    continue
                per_choice[(i, j)] = []
            for atoms in terms:
                for (i, j), record in per_choice.items():
                    key = (int(atoms[i]), int(atoms[j]))
                    slot = pair_slot.get(key)
                    if slot is None:
                        slot = len(pairs)
                        pair_slot[key] = slot
                        pairs.append(key)
                    record.append(slot)
            slots[namespace] = {
                choice: torch.tensor(record, dtype=torch.long)
                for choice, record in per_choice.items()
            }

        self.pairs = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        self.slots = slots

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def gather(self, hessian: torch.Tensor) -> torch.Tensor:
        """The ``(n_pairs, 9)`` blocks of *hessian* on these pairs.

        Args:
            hessian: ``(3N, 3N)`` Cartesian Hessian, flat index ``3a + c``.

        Returns:
            One row-major flattened 3x3 block per pair, in :attr:`pairs` order.

        Raises:
            ValueError: *hessian* is not square, or an index exceeds its size.
        """
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError(f"hessian must be square (3N, 3N); got {tuple(hessian.shape)}")
        if len(self) == 0:
            return hessian.new_zeros((0, 9))
        limit = hessian.shape[0] // 3
        if int(self.pairs.max()) >= limit:
            raise ValueError(
                f"a bonded term names atom {int(self.pairs.max())} but the Hessian covers "
                f"{limit} atoms"
            )
        rows = (3 * self.pairs[:, 0]).unsqueeze(1) + torch.arange(3, device=hessian.device)
        cols = (3 * self.pairs[:, 1]).unsqueeze(1) + torch.arange(3, device=hessian.device)
        return hessian[rows.unsqueeze(2), cols.unsqueeze(1)].reshape(-1, 9)


class TermHessian:
    """Each bonded term's own ``(w, 3, w, 3)`` Hessian, from the term's own energy.

    **Why a gathered copy.** The trick is to differentiate not the batch
    positions but a *fresh copy* of each term's own atoms, ``(n_terms, w, 3)``.
    Two terms sharing an atom then hold two independent copies of it, so the
    total energy is separable over terms and its Hessian is block diagonal by
    construction — one ``(w, 3, w, 3)`` block per term, exact, with no
    cancellation to arrange and no per-term loop.

    **Why not a second energy expression.** The energy is passed in as a
    callable, so this class holds no physics: it calls the potential that the
    force field already uses. A local ``E = ½k(r−r0)²`` written here instead
    would be a second truth that stage 1 could fit while stages 2 and 3 priced
    something else.

    Only geometry is differentiated twice — the parameters keep their graph, so
    a loss on these blocks reaches the force constants that produced them.

    **The gathered positions must already require grad, and this never calls
    ``requires_grad_`` itself.** Creating a leaf inside a compiled region is
    ``Unsupported: returning intermediate with requires_grad_()`` — AOTAutograd
    drops the effect on a graph output, so dynamo refuses rather than return a
    wrong one. It cost stage 1 every cell of its first calibration grid and is
    invisible eagerly. The caller marks ``atoms.pos`` on the host instead, the
    same prologue discipline :class:`molix.autograd.Derivative` uses.

    Differentiating the *gather* rather than a private copy is what keeps the
    blocks per-term: ``x = pos[index]`` is one tensor node and ``dE/dx`` treats its
    entries as independent, whatever sharing happens upstream in ``pos``. Two terms
    on the same atom still get their own block, and no copy is made.
    """

    def blocks(
        self,
        energy: Callable[[torch.Tensor], torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-term Hessian of a term-separable *energy*.

        Args:
            energy: Takes the gathered positions ``(n_terms, w, 3)`` and returns
                the scalar total energy of those terms. It must be separable
                over the first axis — true of every bonded term, whose energy
                reads only its own atoms.
            positions: Gathered term positions ``(n_terms, w, 3)``, gathered from
                a ``pos`` that already requires grad.

        Returns:
            ``(n_terms, w, 3, w, 3)`` — ``[t, a, c, b, d]`` is
            ``d2E_t / dx[a][c] dx[b][d]``.

        Raises:
            ValueError: *positions* is not ``(n_terms, w, 3)``, or does not require
                grad — there is then nothing to differentiate, and marking it here
                would break the compile.
        """
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(f"positions must be (n_terms, width, 3); got {tuple(positions.shape)}")
        if not positions.requires_grad:
            raise ValueError(
                "positions does not require grad, so their curvature cannot be taken. "
                "Mark atoms.pos on the host before the traced forward — this cannot "
                "mark it, because creating a leaf inside a compiled region is "
                "Unsupported: returning intermediate with requires_grad_()."
            )
        n_terms, width = int(positions.shape[0]), int(positions.shape[1])
        local = positions
        with torch.enable_grad():
            total = energy(local)
            (gradient,) = torch.autograd.grad(total, local, create_graph=True)
            # One reverse pass per Cartesian component of the term's own atoms —
            # 6 for a bond, 12 for a torsion, each over every term at once. The
            # width is a Python int off a static shape, so the trip count is
            # fixed at trace time.
            rows = [
                torch.autograd.grad(
                    gradient[:, a, c].sum(),
                    local,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for a in range(width)
                for c in range(3)
            ]
        return torch.stack(rows, dim=1).reshape(n_terms, width, 3, width, 3)
