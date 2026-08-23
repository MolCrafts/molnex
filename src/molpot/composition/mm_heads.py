"""Continuous Class-I MM parameter heads (learnable classical force fields).

Maps per-interaction (or per-atom) feature vectors to Class-I IR parameter
dicts in CLASS_I_CANONICAL units (kcal/mol, Å, e, rad). Heads are pure MLPs;
endpoint symmetry for bonds/angles/propers is the caller's responsibility via
order-invariant feature pooling (see module notes below).

Endpoint symmetry (documented contract)
--------------------------------------
Bond features that reverse atom order ``(i,j)↔(j,i)``, angle features that
reverse ``(i,j,k)↔(k,j,i)``, and proper torsion features that reverse
``(i,j,k,l)↔(l,k,j,i)`` must yield **identical** parameters. Enforce this by
symmetric feature construction at the call site (e.g. sum/mean of endpoint
embeddings). These heads do **not** reorder atoms — they stay pure MLPs.

Units (CLASS_I_CANONICAL)
-------------------------
- Bond ``k``: kcal/mol/Å²; ``r0``: Å
- Angle ``k``: kcal/mol/rad²; ``theta0``: rad ∈ (0, π)
- Proper Cartesian ``a``, ``b``: kcal/mol (unconstrained)
- Proper OPLS ``k1..k4``: kcal/mol (LAMMPS ``Kn``, unconstrained)
- Improper periodic ``k``: kcal/mol; ``phase``: rad
- Improper harmonic ``k``: kcal/mol/rad²; ``chi0``: rad

References:
    Spec: learnable-classical-ff-03-mm-heads
    Spec: oplsaa-parameterizer-02-heads
    OpenMM User Guide §19 "Forces"
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config

__all__ = [
    "BondParamHead",
    "AngleParamHead",
    "ProperTorsionParamHead",
    "ImproperParamHead",
]


def _mlp(feature_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, dtype=config.ftype),
    )


class BondParamHead(nn.Module):
    """Map bond features to harmonic bond parameters.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        min_k: Positive floor for force constants (kcal/mol/Å²).
        min_r0: Positive floor for equilibrium lengths (Å).

    Forward:
        features: ``(N_bonds, D)`` → ``{"k": (N,), "r0": (N,)}`` with
        ``k > min_k`` and ``r0 > min_r0`` via softplus.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        min_k: float = 1e-4,
        min_r0: float = 1e-4,
    ) -> None:
        super().__init__()
        self.min_k = min_k
        self.min_r0 = min_r0
        self.mlp = _mlp(feature_dim, hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict bond ``k``, ``r0`` from features.

        Args:
            features: Bond features ``(N_bonds, feature_dim)``. Prefer
                order-invariant pooling so ``(i,j)`` and ``(j,i)`` match.

        Returns:
            Dict with ``k`` ``(N,)`` (kcal/mol/Å²) and ``r0`` ``(N,)`` (Å).
        """
        raw = self.mlp(features)
        k = F.softplus(raw[:, 0]) + self.min_k
        r0 = F.softplus(raw[:, 1]) + self.min_r0
        return {"k": k, "r0": r0}


class AngleParamHead(nn.Module):
    """Map angle features to harmonic angle parameters.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        min_k: Positive floor for force constants (kcal/mol/rad²).

    Forward:
        features: ``(N_angles, D)`` → ``{"k": (N,), "theta0": (N,)}`` with
        ``k > min_k`` and ``theta0 ∈ (0, π)`` via scaled sigmoid.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        min_k: float = 1e-4,
    ) -> None:
        super().__init__()
        self.min_k = min_k
        self.mlp = _mlp(feature_dim, hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict angle ``k``, ``theta0`` from features.

        Args:
            features: Angle features ``(N_angles, feature_dim)``. Prefer
                endpoint-symmetric pooling for ``(i,j,k)`` vs ``(k,j,i)``.

        Returns:
            Dict with ``k`` ``(N,)`` (kcal/mol/rad²) and ``theta0`` ``(N,)``
            in radians, strictly inside ``(0, π)``.
        """
        raw = self.mlp(features)
        k = F.softplus(raw[:, 0]) + self.min_k
        # Open interval (0, π): epsilon margin avoids exact 0/π singularities.
        eps = 1e-4
        theta0 = eps + (math.pi - 2.0 * eps) * torch.sigmoid(raw[:, 1])
        return {"k": k, "theta0": theta0}


class ProperTorsionParamHead(nn.Module):
    """Map proper-torsion features to Cartesian or OPLS-AA amplitudes.

    Two output forms (``form``):

    * ``"cartesian"`` (default): unconstrained Fourier ``(a_n, b_n)`` with
      default periodicities ``n = 1…6``. Mapping onto the cosine table
      ``(k, γ, periodicity, idivf)`` happens at IR / ``to_potentials`` /
      ``project`` export, not in this head.
    * ``"opls"``: unconstrained LAMMPS ``dihedral_style opls`` coefficients
      ``k1..k4`` (each ``(N,)``). Aligns with ``PotentialMeta``
      ``dihedral:opls`` ParamSpec.

    Units (``CLASS_I_CANONICAL``): ``a``, ``b``, and OPLS ``k1..k4`` in
    kcal/mol; Cartesian ``idivf`` dimensionless. There is no ``c0`` /
    heat-of-formation offset.

    **OPLS ``Kn`` vs Class-I half-amplitudes:** the OPLS-AA / LAMMPS energy

    .. math::

        E = \\tfrac12 K_1[1+\\cos\\phi]
          + \\tfrac12 K_2[1-\\cos 2\\phi]
          + \\tfrac12 K_3[1+\\cos 3\\phi]
          + \\tfrac12 K_4[1-\\cos 4\\phi]

    already includes the factor :math:`\\tfrac12`. Keys ``k1..k4`` **are**
    the LAMMPS ``dihedral_coeff`` ``Kn`` values — do **not** multiply or
    divide by :math:`\\tfrac12` again. Class-I multi-cosine IR on the OPLS
    manifold uses half-amplitudes :math:`k_m = K_n/2` with
    :math:`n=(1,2,3,4)` and :math:`\\gamma=(0,\\pi,0,\\pi)`; this head's
    OPLS path emits :math:`K_n`, not :math:`k_m`.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        n_terms: Number of Fourier terms ``T`` (Cartesian only). Defaults
            to 6. Incompatible with ``form="opls"`` when not left at the
            Cartesian default.
        periodicity: Integer periodicities of length ``T`` (Cartesian
            only). Defaults to ``(1, 2, …, T)``. Must be omitted for
            ``form="opls"``.
        default_idivf: Default AMBER-style identity divisor per interaction
            (Cartesian only).
        form: ``"cartesian"`` or ``"opls"``. Defaults to ``"cartesian"``.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        n_terms: int = 6,
        periodicity: tuple[int, ...] | list[int] | None = None,
        default_idivf: float = 1.0,
        form: Literal["cartesian", "opls"] = "cartesian",
    ) -> None:
        super().__init__()
        if form not in ("cartesian", "opls"):
            raise ValueError(f"form must be 'cartesian' or 'opls', got {form!r}")
        self.form = form
        self.default_idivf = default_idivf

        if form == "opls":
            if n_terms != 6:
                raise ValueError(
                    f"form='opls' does not accept n_terms (got {n_terms}; OPLS emits fixed k1..k4)"
                )
            if periodicity is not None:
                raise ValueError(
                    "form='opls' does not accept periodicity "
                    "(OPLS emits k1..k4 only, no periodicity buffer)"
                )
            self.n_terms = 4
            self.mlp = _mlp(feature_dim, hidden_dim, 4)
            # No periodicity buffer — OPLS emits k1..k4 only; an empty
            # register_buffer would enter state_dict and break empty-.max().
            return

        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}")
        if periodicity is None:
            periodicity = tuple(range(1, n_terms + 1))
        if len(periodicity) != n_terms:
            raise ValueError(f"periodicity length {len(periodicity)} must equal n_terms={n_terms}")
        self.n_terms = n_terms
        self.mlp = _mlp(feature_dim, hidden_dim, 2 * n_terms)
        self.register_buffer(
            "periodicity",
            torch.tensor(list(periodicity), dtype=torch.long),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict proper-torsion parameters from features.

        Args:
            features: Proper features ``(N_propers, feature_dim)``. Prefer
                order-invariant pooling so ``(i,j,k,l)`` and ``(l,k,j,i)``
                match.

        Returns:
            For ``form="cartesian"``: dict with

            - ``a``: ``(N, T)`` cosine amplitudes (kcal/mol), unconstrained
            - ``b``: ``(N, T)`` sine amplitudes (kcal/mol), unconstrained
            - ``periodicity``: ``(T,)`` integer buffer
            - ``idivf``: ``(N,)`` identity divisors

            Does not emit ``k``, ``phase``, or ``c0``.

            For ``form="opls"``: dict with exactly ``k1``, ``k2``, ``k3``,
            ``k4`` — each ``(N,)`` unconstrained LAMMPS ``Kn`` (kcal/mol).
            Does not emit ``a``/``b``/``k``/``phase``/``periodicity``/``idivf``.
        """
        if self.form == "opls":
            raw = self.mlp(features)
            return {
                "k1": raw[:, 0],
                "k2": raw[:, 1],
                "k3": raw[:, 2],
                "k4": raw[:, 3],
            }

        n = features.shape[0]
        raw = self.mlp(features)
        t = self.n_terms
        a = raw[:, :t]
        b = raw[:, t : 2 * t]
        idivf = torch.full(
            (n,),
            self.default_idivf,
            dtype=features.dtype,
            device=features.device,
        )
        return {
            "a": a,
            "b": b,
            "periodicity": self.periodicity,
            "idivf": idivf,
        }


class ImproperParamHead(nn.Module):
    """Map improper features to harmonic and/or periodic improper parameters.

    Config flags select which parameter families to emit (not a multi-method
    switch over unrelated kernels — both families are Class-I improper terms).

    When both modes are enabled, keys are namespaced
    (``k_harmonic`` / ``chi0`` and ``k_periodic`` / ``phase`` / …) to avoid
    collisions. Single-mode outputs use the short IR field names (``k``,
    ``chi0`` or ``k``, ``phase``, ``periodicity``, ``idivf``).

    Universal-v0 defaults emit the periodic branch only (``n = 1, 2``)
    in polar ``(k, phase)`` coordinates — not Cartesian. Harmonic
    impropers remain available via ``include_harmonic=True``.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        include_harmonic: Emit harmonic improper params.
        include_periodic: Emit multi-term cosine improper params.
        n_terms: Fourier terms for the periodic branch. Defaults to 2.
        periodicity: Integer periodicities for the periodic branch.
            Defaults to ``(1, 2, …, n_terms)``.
        min_k: Softplus floor for force constants / barriers.
        default_idivf: Identity divisor for the periodic branch.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        include_harmonic: bool = False,
        include_periodic: bool = True,
        n_terms: int = 2,
        periodicity: tuple[int, ...] | list[int] | None = None,
        min_k: float = 1e-4,
        default_idivf: float = 1.0,
    ) -> None:
        super().__init__()
        if not include_harmonic and not include_periodic:
            raise ValueError("ImproperParamHead requires include_harmonic and/or include_periodic")
        self.include_harmonic = include_harmonic
        self.include_periodic = include_periodic
        self.min_k = min_k
        self.default_idivf = default_idivf
        self.n_terms = n_terms

        out_dim = 0
        if include_harmonic:
            out_dim += 2  # k, chi0
        if include_periodic:
            if n_terms < 1:
                raise ValueError(f"n_terms must be >= 1, got {n_terms}")
            if periodicity is None:
                periodicity = tuple(range(1, n_terms + 1))
            if len(periodicity) != n_terms:
                raise ValueError(
                    f"periodicity length {len(periodicity)} must equal n_terms={n_terms}"
                )
            out_dim += 2 * n_terms  # k, phase per term
            self.register_buffer(
                "periodicity",
                torch.tensor(list(periodicity), dtype=torch.long),
            )
        else:
            # Placeholder so attribute always exists for type checkers.
            self.register_buffer("periodicity", torch.zeros(0, dtype=torch.long))

        self.mlp = _mlp(feature_dim, hidden_dim, out_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict improper parameters from features.

        Args:
            features: Improper features ``(N_impropers, feature_dim)``.

        Returns:
            Dict of parameter tensors. See class docstring for key naming when
            both harmonic and periodic modes are enabled.
        """
        n = features.shape[0]
        raw = self.mlp(features)
        out: dict[str, torch.Tensor] = {}
        offset = 0
        both = self.include_harmonic and self.include_periodic

        if self.include_harmonic:
            k_h = F.softplus(raw[:, offset]) + self.min_k
            chi0 = raw[:, offset + 1]
            offset += 2
            if both:
                out["k_harmonic"] = k_h
                out["chi0"] = chi0
            else:
                out["k"] = k_h
                out["chi0"] = chi0

        if self.include_periodic:
            t = self.n_terms
            k_p = F.softplus(raw[:, offset : offset + t]) + self.min_k
            phase = raw[:, offset + t : offset + 2 * t]
            idivf = torch.full(
                (n,),
                self.default_idivf,
                dtype=features.dtype,
                device=features.device,
            )
            if both:
                out["k_periodic"] = k_p
            else:
                out["k"] = k_p
            out["phase"] = phase
            out["periodicity"] = self.periodicity
            out["idivf"] = idivf

        return out
