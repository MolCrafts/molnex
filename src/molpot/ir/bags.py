"""Typed parameter bags for the Class-I Potential Intermediate Representation.

Bags hold parameter tables only — they do not evaluate energy. Index
conventions (bond_index [2, N], proper_index [4, N], …) live on the kernel
side; bags validate only internal field-shape consistency.

References:
    OpenMM User Guide §19 "Forces"
    SMIRNOFF specification (OpenFF) — proper / improper / nonbonded sections
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "BondBag",
    "AngleBag",
    "ProperTorsionBag",
    "ImproperPeriodicBag",
    "ImproperHarmonicBag",
    "LJBag",
    "ChargeBag",
]


def _require_equal_1d(a: torch.Tensor, b: torch.Tensor, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have equal shape, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def _broadcast_periodicity(k: torch.Tensor, periodicity: torch.Tensor) -> torch.Tensor:
    """Per-type multiplicities, broadcasting a shared ``[n_terms]`` row.

    The canonical table is per type and per term so AMBER and CHARMM torsions
    fit; one row shared by every type (OPLS ``n = 1,2,3,4``) is that table's
    special case, so it is accepted here and expanded rather than refused.
    """
    if periodicity.ndim == 1 and k.ndim == 2 and periodicity.shape[0] == k.shape[1]:
        return periodicity.expand(k.shape[0], -1).contiguous()
    return periodicity


def _validate_cosine_term_tables(
    *,
    k: torch.Tensor,
    periodicity: torch.Tensor,
    phase: torch.Tensor,
    bag_name: str,
) -> None:
    """Validate multi-term cosine torsion tables.

    Expected shapes
    ---------------
    k, phase, periodicity : ``[n_types, n_terms]``
    """
    if k.ndim != 2:
        raise ValueError(f"{bag_name}.k must be [n_types, n_terms], got shape {tuple(k.shape)}")
    if phase.shape != k.shape:
        raise ValueError(
            f"{bag_name}.phase must match k shape {tuple(k.shape)}, got {tuple(phase.shape)}"
        )
    if periodicity.shape != k.shape:
        raise ValueError(
            f"{bag_name}.periodicity must match k shape {tuple(k.shape)} "
            f"(or be [n_terms] and broadcast), got {tuple(periodicity.shape)}"
        )
    if not bool((periodicity > 0).all()):
        raise ValueError(f"{bag_name}.periodicity entries must be > 0, got {periodicity.tolist()}")


@dataclass
class BondBag:
    """Harmonic bond parameters: ``E = ½ k (r − r₀)²``.

    Attributes:
        k: Force constants ``[n_types]`` (or per-interaction).
        r0: Equilibrium lengths ``[n_types]`` (same shape as ``k``).
    """

    k: torch.Tensor
    r0: torch.Tensor

    def __post_init__(self) -> None:
        _require_equal_1d(self.k, self.r0, "k", "r0")


@dataclass
class AngleBag:
    """Harmonic angle parameters: ``E = ½ k (θ − θ₀)²``.

    Attributes:
        k: Force constants ``[n_types]``.
        theta0: Equilibrium angles in radians ``[n_types]``.
    """

    k: torch.Tensor
    theta0: torch.Tensor

    def __post_init__(self) -> None:
        _require_equal_1d(self.k, self.theta0, "k", "theta0")


@dataclass
class ProperTorsionBag:
    """Class-I multi-term cosine proper torsion parameters.

    Energy per interaction type ``t`` and term ``m``::

        E = Σ_m k[t,m] * [1 + cos(n[m] * φ − γ[t,m])]

    Storage is the cosine table ``(k, phase, periodicity)`` only — no
    ``c0``, no persisted ``a``/``b``, and no AMBER divisor: a reader divides
    ``k`` by ``IDIVF`` on the way in (spec ff-params-01). Cartesian Fourier
    amplitudes convert through :meth:`from_cartesian`.

    Units (``CLASS_I_CANONICAL``): ``k`` in kcal/mol; ``phase`` in rad.

    Attributes:
        k: Barrier heights ``[n_types, n_terms]`` (kcal/mol).
        periodicity: Integer periodicities ``[n_types, n_terms]`` — per type and
            per term, so AMBER and CHARMM torsions fit as well as OPLS.
        phase: Phase offsets γ ``[n_types, n_terms]`` (radians).
    """

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor

    def __post_init__(self) -> None:
        self.periodicity = _broadcast_periodicity(self.k, self.periodicity)
        _validate_cosine_term_tables(
            k=self.k,
            periodicity=self.periodicity,
            phase=self.phase,
            bag_name="ProperTorsionBag",
        )

    @classmethod
    def from_cartesian(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> ProperTorsionBag:
        """Build cosine-table storage from Cartesian Fourier amplitudes.

        Maps ``k = hypot(a, b)`` and ``phase = atan2(b, a)`` (radians).
        Zero amplitude uses the IEEE convention ``atan2(0, 0) = 0``.
        There is no ``c0`` field and no divisor.

        Units (``CLASS_I_CANONICAL``): ``a``, ``b``, ``k`` in kcal/mol;
        ``phase`` in rad.

        Args:
            a: Cosine amplitudes ``[n_types, n_terms]`` (kcal/mol).
            b: Sine amplitudes ``[n_types, n_terms]`` (kcal/mol).
            periodicity: Integer periodicities ``[n_types, n_terms]``.

        Returns:
            Bag with fields ``k``, ``phase``, ``periodicity``.

        Raises:
            ValueError: If ``a``/``b`` are not 2-D or their shapes differ.
        """
        k, phase = cls.polar_from_cartesian(a, b)
        return cls(k=k, periodicity=periodicity, phase=phase)

    @staticmethod
    def polar_from_cartesian(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``k = hypot(a, b)``, ``phase = atan2(b, a)`` — tensor math only.

        Separate from :meth:`from_cartesian` because the parameterized energy
        path converts *inside* the compiled region, and building a bag there
        would run :func:`_validate_cosine_term_tables` — whose positivity check
        is a ``bool()`` on a tensor, i.e. a host sync and a data-dependent
        guard dynamo refuses. The shape checks below are Python-level and
        stay: they are free under compile.

        Units (``CLASS_I_CANONICAL``): ``a``, ``b``, ``k`` in kcal/mol;
        ``phase`` in rad.

        Args:
            a: Cosine amplitudes ``[n_types, n_terms]`` (kcal/mol).
            b: Sine amplitudes ``[n_types, n_terms]`` (kcal/mol).

        Returns:
            ``(k, phase)``, each ``[n_types, n_terms]``.

        Raises:
            ValueError: If ``a``/``b`` are not 2-D or their shapes differ.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"a and b must be 2-D [n_types, n_terms], got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if a.shape != b.shape:
            raise ValueError(
                f"a and b must have equal shape, got {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        # The map is singular at the origin: ``d hypot / d a = a / hypot`` and
        # ``d atan2 / d a = -b / (a² + b²)`` are both ``0 / 0`` there, so a row
        # of exactly ``(0, 0)`` returns a finite ``k = 0`` and a NaN gradient.
        # That row is not hypothetical — every padded torsion slot is exactly
        # zero, because the mask zeroes the parameters rather than the energy
        # (``Potentials._mask_term_tables``). Left alone it hands NaN to every
        # encoder weight while the loss itself still reads finite.
        #
        # Evaluate the branch on substituted inputs and select afterwards: at a
        # zero row both ``k`` and its gradient are 0, and ``phase`` carries the
        # gradient of ``atan2(b, 1)``, which the caller multiplies by
        # ``k = 0`` (``dE/dγ = k · sin(nφ − γ)``). Live rows are untouched —
        # exact values, no epsilon bias.
        degenerate = (a == 0) & (b == 0)
        ones = torch.ones_like(a)
        square = torch.where(degenerate, ones, a * a + b * b)
        k = torch.where(degenerate, torch.zeros_like(a), torch.sqrt(square))
        phase = torch.atan2(b, torch.where(degenerate, ones, a))
        return k, phase

    @classmethod
    def _cosine_fields(cls, pp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return polar cosine fields, converting cartesian ``(a, b)`` when present.

        Raises:
            ValueError: If both ``(a, b)`` and ``(k, phase)`` are present.
        """
        has_cartesian = "a" in pp or "b" in pp
        has_polar = "k" in pp or "phase" in pp
        if has_cartesian and has_polar:
            raise ValueError("propers params cannot mix cartesian (a, b) with polar (k, phase)")
        if "a" in pp and "b" in pp:
            k, phase = cls.polar_from_cartesian(pp["a"], pp["b"])
            return {
                "k": k,
                "phase": phase,
                "periodicity": _broadcast_periodicity(k, pp["periodicity"]),
            }
        return pp


@dataclass
class ImproperPeriodicBag:
    """Class-I multi-term cosine improper torsion parameters.

    Same field layout as :class:`ProperTorsionBag`. The central atom is
    ``atomi`` / row 0 of ``improper_index`` (molnex center-first).

    Attributes:
        k: Barrier heights ``[n_types, n_terms]``.
        periodicity: Integer periodicities ``[n_types, n_terms]``.
        phase: Phase offsets γ ``[n_types, n_terms]`` (radians).
    """

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor

    def __post_init__(self) -> None:
        self.periodicity = _broadcast_periodicity(self.k, self.periodicity)
        _validate_cosine_term_tables(
            k=self.k,
            periodicity=self.periodicity,
            phase=self.phase,
            bag_name="ImproperPeriodicBag",
        )


@dataclass
class ImproperHarmonicBag:
    """Harmonic improper parameters: ``E = ½ k (χ − χ₀)²``.

    Attributes:
        k: Force constants ``[n_types]``.
        chi0: Equilibrium improper angles in radians ``[n_types]``.
    """

    k: torch.Tensor
    chi0: torch.Tensor

    def __post_init__(self) -> None:
        _require_equal_1d(self.k, self.chi0, "k", "chi0")


@dataclass
class LJBag:
    """Lennard-Jones 12-6 atom (or type) parameters.

    Attributes:
        epsilon: Well depths ``[n]``.
        sigma: Collision diameters ``[n]`` (same shape as ``epsilon``).
    """

    epsilon: torch.Tensor
    sigma: torch.Tensor

    def __post_init__(self) -> None:
        _require_equal_1d(self.epsilon, self.sigma, "epsilon", "sigma")


@dataclass
class ChargeBag:
    """Partial charges.

    Attributes:
        charge: Charges in elementary charge units ``[n_atoms]`` (or per type).
    """

    charge: torch.Tensor
