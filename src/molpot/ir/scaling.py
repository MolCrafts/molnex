"""Nonbonded 1–2 / 1–3 / 1–4 scaling factors for Class-I force fields.

A *Class-I* force field is the classical diagonal form — harmonic bonds and
angles, a periodic proper torsion, an improper term, and the nonbonded
Lennard-Jones and Coulomb pair terms — with no cross terms coupling one
internal coordinate to another. Two atoms of one molecule are called *1–2*
when a single covalent bond separates them, *1–3* when two bonds do (the two
ends of a bond angle) and *1–4* when three do (the two ends of a proper
torsion). Those pairs are already described by the bonded terms, so the
nonbonded terms multiply them by a scale factor: ``0`` removes the pair
outright, ``1`` keeps it in full, and the fraction in between is the usual
1–4 attenuation. Every factor here is a pure number (dimensionless); it
multiplies an energy already carrying the force field's own units.

Scaling is keyed by *interaction class*: one :class:`ExclusionScales` triple
per class, looked up through :meth:`NonbondedScaling.kinds` /
:meth:`NonbondedScaling.triple` rather than through flat attribute names. A
new permanent or induced class is then one extra field, and every consumer
that iterates ``kinds()`` keeps working unchanged.

The defaults follow the conventions of the AMBER family of biomolecular force
fields, of its small-molecule extension GAFF (General AMBER Force Field), and
of the Open Force Field Initiative's SMIRNOFF format:

| relation | q     | lj  |
|----------|-------|-----|
| 1–2      | 0     | 0   |
| 1–3      | 0     | 0   |
| 1–4      | 5/6   | 0.5 |

References:
    LAMMPS ``special_bonds`` command — the 1–2 / 1–3 / 1–4 weight semantics
    these triples carry.
    OpenMM user guide, nonbonded exceptions — the same three weights, listed
    per exception pair rather than per relation.
    SMIRNOFF specification, nonbonded section — the ``scale14`` conventions
    the defaults above follow.
    Cornell et al. "A Second Generation Force Field for the Simulation of
    Proteins, Nucleic Acids, and Organic Molecules" JACS 117, 5179 (1995)
    https://doi.org/10.1021/ja00124a002
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["ExclusionScales", "NonbondedScaling"]


@dataclass(frozen=True)
class ExclusionScales:
    """Bonded-path scale factors for one interaction class.

    Frozen: a topology-fixed exclusion table is built from these once per
    system and caches its answers, so a later mutation would silently
    disagree with the table already in use.

    Attributes:
        s12: Dimensionless scale for 1–2 (directly bonded) pairs.
            Default ``0`` — the pair is removed.
        s13: Dimensionless scale for 1–3 (angle-end) pairs. Default ``0``.
        s14: Dimensionless scale for 1–4 (proper-torsion-end) pairs.
            Default ``1`` — unscaled, i.e. this bare triple applies no 1–4
            attenuation; the force-field defaults that do live on
            :class:`NonbondedScaling`.
    """

    s12: float = 0.0
    s13: float = 0.0
    s14: float = 1.0


@dataclass(frozen=True)
class NonbondedScaling:
    """Per-interaction-class exclusion scales for a Class-I force field.

    Each field is one interaction class carrying its own
    :class:`ExclusionScales` triple; the field names are the keys reported by
    :meth:`kinds`. Adding a class (e.g. ``induced=`` for a polarizable model)
    is a new field on this type — the ``kinds()`` / :meth:`triple` contract
    that consumers are written against does not change.

    Attributes:
        q: Coulomb (partial-charge) scales, dimensionless. Default
            ``(0, 0, 5/6)`` — the AMBER 1–4 electrostatic factor.
        lj: Lennard-Jones scales, dimensionless. Default ``(0, 0, 0.5)``.
    """

    q: ExclusionScales = ExclusionScales(0.0, 0.0, 5.0 / 6.0)
    lj: ExclusionScales = ExclusionScales(0.0, 0.0, 0.5)

    def kinds(self) -> tuple[str, ...]:
        """Return the declared interaction-class keys, in declaration order.

        Returns:
            ``("q", "lj")`` — plus any class added by a later model (e.g.
            ``"induced"``), so consumers iterate this instead of hard-coding
            the key set.
        """
        return tuple(f.name for f in fields(self))

    def triple(self, kind: str) -> tuple[float, float, float]:
        """Return ``(s12, s13, s14)`` for one interaction class.

        Args:
            kind: An interaction-class key from :meth:`kinds`.

        Returns:
            The dimensionless 1–2 / 1–3 / 1–4 scale factors for ``kind``.

        Raises:
            KeyError: ``kind`` is not a declared class. Never falls back to
                an unscaled ``1.0``, which is how an unmodelled term would
                escape exclusion unnoticed.
        """
        if kind not in self.kinds():
            raise KeyError(f"Unknown interaction class {kind!r}; declared kinds are {self.kinds()}")
        scales: ExclusionScales = getattr(self, kind)
        return (float(scales.s12), float(scales.s13), float(scales.s14))

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Return the scales nested by interaction class.

        Returns:
            ``{"q": {"s12": …, "s13": …, "s14": …}, "lj": {…}}``. Export
            adapters flatten this into their own backend key names; this type
            does not own a flat spelling.
        """
        return {
            kind: dict(zip(("s12", "s13", "s14"), self.triple(kind), strict=True))
            for kind in self.kinds()
        }
