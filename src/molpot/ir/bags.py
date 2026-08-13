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


def _validate_cosine_term_tables(
    *,
    k: torch.Tensor,
    periodicity: torch.Tensor,
    phase: torch.Tensor,
    idivf: torch.Tensor,
    bag_name: str,
) -> None:
    """Validate multi-term cosine torsion tables.

    Expected shapes
    ---------------
    k, phase : ``[n_types, n_terms]``
    periodicity : ``[n_terms]``
    idivf : ``[n_types]``
    """
    if k.ndim != 2:
        raise ValueError(f"{bag_name}.k must be [n_types, n_terms], got shape {tuple(k.shape)}")
    if phase.shape != k.shape:
        raise ValueError(
            f"{bag_name}.phase must match k shape {tuple(k.shape)}, got {tuple(phase.shape)}"
        )
    n_types, n_terms = k.shape
    if periodicity.ndim != 1 or periodicity.shape[0] != n_terms:
        raise ValueError(
            f"{bag_name}.periodicity must be [n_terms]={n_terms}, "
            f"got shape {tuple(periodicity.shape)}"
        )
    if idivf.ndim != 1 or idivf.shape[0] != n_types:
        raise ValueError(
            f"{bag_name}.idivf must be [n_types]={n_types}, got shape {tuple(idivf.shape)}"
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

        E = Σ_m (k[t,m] / idivf[t]) * [1 + cos(n[m] * φ − γ[t,m])]

    Storage is the cosine table ``(k, phase, periodicity, idivf)`` only —
    no ``c0`` and no persisted ``a``/``b``. Cartesian Fourier amplitudes
    convert through :meth:`from_cartesian`.

    Units (``CLASS_I_CANONICAL``): ``k`` in kcal/mol; ``phase`` in rad.

    Attributes:
        k: Barrier heights ``[n_types, n_terms]`` (kcal/mol).
        periodicity: Integer periodicities ``[n_terms]`` (shared across types).
        phase: Phase offsets γ ``[n_types, n_terms]`` (radians).
        idivf: AMBER scale / identity divisor ``s`` per type ``[n_types]``.
    """

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor
    idivf: torch.Tensor

    def __post_init__(self) -> None:
        _validate_cosine_term_tables(
            k=self.k,
            periodicity=self.periodicity,
            phase=self.phase,
            idivf=self.idivf,
            bag_name="ProperTorsionBag",
        )

    @classmethod
    def from_cartesian(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        periodicity: torch.Tensor,
        idivf: torch.Tensor | None = None,
    ) -> ProperTorsionBag:
        """Build cosine-table storage from Cartesian Fourier amplitudes.

        Maps ``k = hypot(a, b)`` and ``phase = atan2(b, a)`` (radians).
        Zero amplitude uses the IEEE convention ``atan2(0, 0) = 0``.
        ``idivf`` defaults to ones. There is no ``c0`` field.

        Units (``CLASS_I_CANONICAL``): ``a``, ``b``, ``k`` in kcal/mol;
        ``phase`` in rad.

        Args:
            a: Cosine amplitudes ``[n_types, n_terms]`` (kcal/mol).
            b: Sine amplitudes ``[n_types, n_terms]`` (kcal/mol).
            periodicity: Integer periodicities ``[n_terms]``.
            idivf: Identity divisor ``[n_types]``. Defaults to ones.

        Returns:
            Bag with fields ``k``, ``phase``, ``periodicity``, ``idivf``.

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
        k = torch.hypot(a, b)
        phase = torch.atan2(b, a)
        if idivf is None:
            idivf = torch.ones(a.shape[0], device=a.device, dtype=k.dtype)
        return cls(k=k, periodicity=periodicity, phase=phase, idivf=idivf)

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
            bag = cls.from_cartesian(pp["a"], pp["b"], pp["periodicity"], pp.get("idivf"))
            return {
                "k": bag.k,
                "phase": bag.phase,
                "periodicity": bag.periodicity,
                "idivf": bag.idivf,
            }
        return pp


@dataclass
class ImproperPeriodicBag:
    """Class-I multi-term cosine improper torsion parameters.

    Same field layout as :class:`ProperTorsionBag`. The central atom is
    ``atomi`` / row 0 of ``improper_index`` (molnex center-first).

    Attributes:
        k: Barrier heights ``[n_types, n_terms]``.
        periodicity: Integer periodicities ``[n_terms]``.
        phase: Phase offsets γ ``[n_types, n_terms]`` (radians).
        idivf: Scale / identity divisor ``s`` per type ``[n_types]``.
    """

    k: torch.Tensor
    periodicity: torch.Tensor
    phase: torch.Tensor
    idivf: torch.Tensor

    def __post_init__(self) -> None:
        _validate_cosine_term_tables(
            k=self.k,
            periodicity=self.periodicity,
            phase=self.phase,
            idivf=self.idivf,
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
        q: Charges in elementary charge units ``[n_atoms]`` (or per type).
    """

    q: torch.Tensor
