"""The Class-I functional form Rondo fits.

Spec: ``src/molzoo/specs/rondo.md`` §1.1, §3.6.
"""

from __future__ import annotations

from molpot.forcefield import ClassIForceField
from molpot.forcefield.container import ExclusionScales, NonbondedScaling

__all__ = ["RondoForceField"]


class RondoForceField(ClassIForceField):
    """The force field a Rondo model parameterizes.

    Harmonic bonds and angles, the **general periodic torsion** for propers
    and impropers, Lennard-Jones 12-6 and an open Coulomb sum. Choosing the
    general Class-I form over a named product keeps a learned field from
    being born inside one vendor's functional shape: OPLS-AA is the special
    case where every torsion type carries ``n = 1,2,3,4`` and
    ``gamma = 0, pi, 0, pi``, a manifold the exporters detect — it is not a
    separate style to fit in.

    **The field owns the radius it prices.** ``neighbor_list()`` derives the
    pair list from this object, so molpot charges for exactly the pairs the
    batch carries and never enumerates its own. Handing the model one field
    and the loader another, equal-looking one is the same number tracked by
    hand in two places (ledger finding **F21**); hand the *same* object to
    both, or read it back as ``model.template``.

    **The open Coulomb radius.** These are isolated gas-phase molecules with
    no cell, so the Coulomb term is
    :class:`~molpot.forcefield.pair_terms.CoulombOpenPotential`, whose ``1/r``
    sum is unbounded unless a radius is declared — and ``neighbor_list()``
    refuses to invent one.

    **One radius, stated once.** ``cutoff`` is re-declared on *both* pair
    styles. ``neighbor_list()`` derives the list radius from the largest pair
    cutoff on the field, and ``pair:lj/cut`` carries a default of its own; a
    ``cutoff`` that only reached the Coulomb style would therefore be
    *silently ignored* whenever it is below that default — the field would
    say 9 Å and price 10 Å. Setting both is what makes the declared radius the
    one actually charged for.

    Args:
        name: Force-field name.
        units: Unit set passed to molpy. Positional so
            :meth:`~molpot.forcefield.ForceField.clone` — which reconstructs
            through ``type(self)(name, units)`` — can rebuild this subclass;
            the clone then overwrites the styles registered here, so the
            declared radius travels with them rather than with this default.
        cutoff: Nonbonded radius, Å.
        scale_14_charge: 1-4 Coulomb exclusion scale; 1-2 and 1-3 are ``0``.
        scale_14_lj: 1-4 Lennard-Jones exclusion scale.
        mixing: Lennard-Jones combination rule for ``pair:lj/cut``.
    """

    def __init__(
        self,
        name: str = "rondo",
        units: str = "real",
        *,
        cutoff: float = 10.0,
        scale_14_charge: float = 0.5,
        scale_14_lj: float = 0.5,
        mixing: str = "geometric",
    ) -> None:
        super().__init__(
            name,
            units,
            nonbonded_scaling=NonbondedScaling(
                charge=ExclusionScales(0.0, 0.0, scale_14_charge),
                lj=ExclusionScales(0.0, 0.0, scale_14_lj),
            ),
            mixing=mixing,
        )
        self.def_pairstyle("lj/cut", mixing=mixing, cutoff=cutoff)
        self.def_pairstyle("coulomb", cutoff=cutoff)
