"""Unit and form convention table for Class-I IR → backend force-spec.

IR units (learnable-classical-ff-01): **kcal/mol, Å, e, rad**.
OpenMM standard MD units: **kJ/mol, nm, e, rad**.

Load-bearing goldens:

1. Bond force constant for ``E = ½ k (r − r₀)²``::

       k = 100 kcal mol⁻¹ Å⁻²
       → k_omm = 100 × 4.184 / (0.1 nm)² = 41840 kJ mol⁻¹ nm⁻²

2. AMBER proper torsion barrier ``Vn = 2`` kcal/mol maps to OpenMM
   PeriodicTorsion ``k = 4.184`` kJ/mol because AMBER uses
   ``E = (Vn/2)[1 + cos(...)]`` while OpenMM (and the molnex IR form
   ``E = (k/s)[1 + cos(...)]``) absorb the half-barrier into ``k``.
   Thus ``k_ir = Vn/2 = 1`` and ``k_omm = 1 × 4.184``.

OpenMM PeriodicTorsionForce (User Guide §19) energy::

    E = k [1 + cos(n φ − γ)]

References:
    OpenMM User Guide §19 "Forces"
    SMIRNOFF unit conventions
    Spec: learnable-classical-ff-08-ff-export
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from molix.ff_export.cases import TranslationCase
from molix.units import units

__all__ = [
    "KCAL_PER_MOL_TO_KJ_PER_MOL",
    "ANGSTROM_TO_NM",
    "BOND_K_IR_TO_OPENMM",
    "ANGLE_K_IR_TO_OPENMM",
    "ConventionRow",
    "ConventionTable",
]

KCAL_PER_MOL_TO_KJ_PER_MOL: float = (
    (1.0 * units.kilocalorie_per_mole).to(units.kilojoule_per_mole).magnitude
)
ANGSTROM_TO_NM: float = (1.0 * units.angstrom).to(units.nanometer).magnitude
BOND_K_IR_TO_OPENMM: float = (
    (1.0 * units.parse("kilocalorie_per_mole / angstrom ** 2"))
    .to("kilojoule_per_mole / nanometer ** 2")
    .magnitude
)
ANGLE_K_IR_TO_OPENMM: float = KCAL_PER_MOL_TO_KJ_PER_MOL


@dataclass(frozen=True)
class ConventionRow:
    """One IR term's translation policy for a backend.

    Attributes:
        ir_term: Bag / term name (e.g. ``"bond_harmonic"``).
        case: :class:`TranslationCase` outcome.
        unit_factors: Named scale factors applied to IR fields.
        notes: Human-readable mapping notes (backend form, caveats).
    """

    ir_term: str
    case: TranslationCase
    unit_factors: Mapping[str, float]
    notes: str

    def scale(self, field: str, value: float) -> float:
        """Apply this row's unit factor for ``field`` to ``value``."""
        try:
            return value * self.unit_factors[field]
        except KeyError:
            raise KeyError(f"{self.ir_term!r} has no unit factor {field!r}") from None


class ConventionTable:
    """Frozen lookup of :class:`ConventionRow` by IR term name.

    Build with :meth:`default_openmm` for the Class-I → OpenMM matrix.
    Conversions live on the table (and its rows), not as free functions.
    """

    def __init__(self, rows: Mapping[str, ConventionRow]) -> None:
        self._rows: Mapping[str, ConventionRow] = MappingProxyType(dict(rows))

    def __contains__(self, ir_term: object) -> bool:
        return ir_term in self._rows

    def __getitem__(self, ir_term: str) -> ConventionRow:
        return self._rows[ir_term]

    def row(self, ir_term: str) -> ConventionRow:
        """Return the row for ``ir_term`` or raise :class:`KeyError`."""
        return self._rows[ir_term]

    def terms(self) -> tuple[str, ...]:
        """All registered IR term names (sorted)."""
        return tuple(sorted(self._rows))

    def scale(self, ir_term: str, field: str, value: float) -> float:
        """Scale ``value`` by the named unit factor on ``ir_term``."""
        return self.row(ir_term).scale(field, value)

    def scale_amber_vn(self, vn_kcal: float) -> float:
        """Map AMBER full barrier ``Vn`` (kcal/mol) to OpenMM PeriodicTorsion ``k``.

        AMBER: ``E = (Vn/2)[1 + cos(...)]``. OpenMM: ``E = k[1 + cos(...)]``.
        Golden: ``ConventionTable.default_openmm().scale_amber_vn(2) == 4.184``.
        """
        return self.scale("proper_periodic", "k", vn_kcal / 2.0)

    @classmethod
    def default_openmm(cls) -> ConventionTable:
        """Class-I IR → OpenMM §19 convention matrix."""
        rows = {
            "bond_harmonic": ConventionRow(
                ir_term="bond_harmonic",
                case=TranslationCase.DIRECT_UNIT_SCALE,
                unit_factors=MappingProxyType({"k": BOND_K_IR_TO_OPENMM, "r0": ANGSTROM_TO_NM}),
                notes=(
                    "OpenMM HarmonicBondForce: E = ½ k (r − r0)² with k in "
                    "kJ/mol/nm², r0 in nm. Matches IR ½ k form; scale k by "
                    "4.184/0.01=418.4 (golden: 100 → 41840)."
                ),
            ),
            "angle_harmonic": ConventionRow(
                ir_term="angle_harmonic",
                case=TranslationCase.DIRECT_UNIT_SCALE,
                unit_factors=MappingProxyType({"k": ANGLE_K_IR_TO_OPENMM, "theta0": 1.0}),
                notes=(
                    "OpenMM HarmonicAngleForce: E = ½ k (θ − θ0)²; θ in rad "
                    "both sides; only energy unit on k changes (×4.184)."
                ),
            ),
            "proper_periodic": ConventionRow(
                ir_term="proper_periodic",
                case=TranslationCase.FORM_REPARAMETERIZE,
                unit_factors=MappingProxyType({"k": KCAL_PER_MOL_TO_KJ_PER_MOL}),
                notes=(
                    "OpenMM PeriodicTorsionForce: E = k[1+cos(nφ−γ)]. "
                    "IR stores E = (k/idivf)[1+cos]; export k_omm = "
                    "(k/idivf)×4.184 (idivf absorption = form reparameterize). "
                    "Multi-term bags use DECOMPOSE at emit time. "
                    "AMBER Vn full barrier: k_omm = table.scale_amber_vn(Vn); "
                    "golden Vn=2 → 4.184 kJ/mol."
                ),
            ),
            "lj": ConventionRow(
                ir_term="lj",
                case=TranslationCase.DIRECT_UNIT_SCALE,
                unit_factors=MappingProxyType(
                    {
                        "epsilon": KCAL_PER_MOL_TO_KJ_PER_MOL,
                        "sigma": ANGSTROM_TO_NM,
                    }
                ),
                notes=(
                    "OpenMM NonbondedForce LJ: ε in kJ/mol, σ in nm "
                    "(σ convention; combining rules left to consumer)."
                ),
            ),
            "charge": ConventionRow(
                ir_term="charge",
                case=TranslationCase.DIRECT_UNIT_SCALE,
                unit_factors=MappingProxyType({"q": 1.0}),
                notes="Elementary charge e is identical in IR and OpenMM.",
            ),
            "improper_harmonic": ConventionRow(
                ir_term="improper_harmonic",
                case=TranslationCase.UNSUPPORTED,
                unit_factors=MappingProxyType({}),
                notes=(
                    "OpenMM has no built-in harmonic improper force matching "
                    "IR ImproperHarmonicBag; refuse rather than silent drop."
                ),
            ),
            "improper_periodic": ConventionRow(
                ir_term="improper_periodic",
                case=TranslationCase.FORM_REPARAMETERIZE,
                unit_factors=MappingProxyType({"k": KCAL_PER_MOL_TO_KJ_PER_MOL}),
                notes=(
                    "Periodic improper uses the same PeriodicTorsionForce "
                    "form as propers; particle ordering is adapter-specific "
                    "(molrs center-first vs OpenFF trefoil)."
                ),
            ),
        }
        return cls(rows)
