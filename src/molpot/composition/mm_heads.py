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
    """Map proper-torsion features to the general Fourier amplitudes.

    Emits unconstrained Cartesian ``(a_n, b_n)`` with periodicities
    ``n = 1…T`` (default 6). The mapping onto the canonical cosine table
    ``(k, γ, periodicity)`` happens at IR / ``to_potentials`` / ``project``
    export, not here — ``k = hypot(a, b)`` and ``γ = atan2(b, a)``.

    **Why Cartesian and not ``(k, γ)`` directly.** A barrier that passes
    through zero is ordinary during a fit, and at ``k = 0`` the phase is
    undefined: ``(k, γ)`` has a coordinate singularity there and its gradient
    does not exist, while ``(a, b)`` is smooth through the origin. The polar
    pair is the storage form; the Cartesian pair is the trainable one.

    **No vendor dialect.** There is no OPLS output form: OPLS is this series
    at ``n = 1,2,3,4`` with ``γ = 0, π, 0, π``, so a fit that wants it reaches
    it, and the exporters detect that manifold and write ``dihedral_style
    opls`` when it holds (spec ff-params-01 § 5). Emitting ``k1..k4`` here
    would build one vendor's constraint into the model's shape.

    Units (``CLASS_I_CANONICAL``): ``a``, ``b`` in kcal/mol. There is no
    ``c0`` / heat-of-formation offset.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        n_terms: Number of Fourier terms ``T``. Defaults to 6.
        periodicity: Integer periodicities of length ``T``. Defaults to
            ``(1, 2, …, T)``.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        n_terms: int = 6,
        periodicity: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
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
            Dict with

            - ``a``: ``(N, T)`` cosine amplitudes (kcal/mol), unconstrained
            - ``b``: ``(N, T)`` sine amplitudes (kcal/mol), unconstrained
            - ``periodicity``: ``(N, T)`` integer multiplicities, one row per
              interaction

            Does not emit ``k``, ``phase``, ``c0`` or any ``k1..k4``.
        """
        raw = self.mlp(features)
        t = self.n_terms
        a = raw[:, :t]
        b = raw[:, t : 2 * t]
        return {
            "a": a,
            "b": b,
            # One row per interaction: the canonical table is per type,
            # per term, so a head that predicts N rows declares N of them.
            "periodicity": self.periodicity.expand(a.shape[0], -1),
        }


class ImproperParamHead(nn.Module):
    """Map improper features to harmonic and/or periodic improper parameters.

    Config flags select which parameter families to emit (not a multi-method
    switch over unrelated kernels — both families are Class-I improper terms).

    When both modes are enabled, keys are namespaced
    (``k_harmonic`` / ``chi0`` and ``k_periodic`` / ``phase`` / …) to avoid
    collisions. Single-mode outputs use the short IR field names (``k``,
    ``chi0`` or ``k``, ``phase``, ``periodicity``).

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
    ) -> None:
        super().__init__()
        if not include_harmonic and not include_periodic:
            raise ValueError("ImproperParamHead requires include_harmonic and/or include_periodic")
        self.include_harmonic = include_harmonic
        self.include_periodic = include_periodic
        self.min_k = min_k
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
            if both:
                out["k_periodic"] = k_p
            else:
                out["k"] = k_p
            out["phase"] = phase
            out["periodicity"] = self.periodicity.expand(k_p.shape[0], -1)

        return out
