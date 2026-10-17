"""Class-I parameter heads: features → force-field parameters, as deviations from a reference.

Every head here maps one interaction's feature vector to that interaction's
parameters in molrs's conventions — LAMMPS ``K`` (no ½), angles in degrees,
energies in kcal/mol, lengths in Å — through one MLP per parameter whose last
layer starts at zero. The pre-activation ``raw`` is dimensionless and
``raw = 0`` **is** a reference value the caller supplies per row: the bond
length of two covalent radii, the VSEPR angle of the centre's hybridization,
the GAFF2 Lennard-Jones well of the atom's type. The model therefore begins as
a chemically sensible uniform field and one learning rate moves every
parameter by a comparable physical amount (a unit of ``raw`` is a factor ``e``
on a force constant, 10 % on a bond length, 20° on an angle).

Why the reference is an input and not a constant: a single tetrahedral angle
held sp carbon at 146° against a measured 177.5° after 7 800 steps because
the head had to climb seven units of a saturating sigmoid; with the centre's
own reference it starts there.

Every multiplicative parameter is **bounded** to a window around its reference
(:class:`ExpAround`): force constants to ``[1/8, 8] · ref`` (an improper's
amplitude down to ``1/64``, since the strain objective legitimately sends a
pyramidal centre's to zero), lengths and σ to ``[0.7, 1.4] · ref``; an angle's
``θ0`` to ``[0.5°, 180°]`` (:class:`LinearAround`). The window is reached by
saturation, ByteFF's ``tanh`` (``mlparams.py:104-110``): on each side of the
reference the pre-activation is bent as ``w · tanh(x / w)`` with ``w`` the
distance to that bound, so the slope at the reference is exactly one and the
gradient fades as a bound is approached. Not a straight-through clamp: that
holds the value but keeps passing the full gradient, and a loss that
legitimately wants a row *at* the bound (the improper strain objective at a
pyramidal centre, where ``k → 0``) then pushes forever — ``k · ∂/∂raw`` is
self-limiting when ``k`` may vanish and perpetual when it is held at a floor.
With a tenth of the improper rows held at the floor for 26 000 steps, the
improper stream's LayerNorm gain drifted to an RMS of 5.9 and the handoff
gate refused the checkpoint. Without a ceiling one outlier Hessian row can
carry a ``K`` to 1e4-1e5 — the loss drops the row
(``PartialHessianLoss(blowup=)``) but the head was free to go there.

Reference:
    Zheng et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." (ByteFF) — parameters as
    ``mean + std · tanh(raw)`` around GAFF2 statistics, ``layers.py:470-478``,
    ``definitions.py:106-117``. https://arxiv.org/abs/2408.12817
    Wang et al. "Development and testing of a general amber force field."
    J. Comput. Chem. 25, 1157 (2004). https://doi.org/10.1002/jcc.20035
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn as nn
from molpy import Element

from molix import config
from molpot.heads._common import scalar_mlp

__all__ = [
    "AngleHead",
    "BondHead",
    "CovalentRadiusSum",
    "CvffHead",
    "ExpAround",
    "IMPROPER_K_BOUNDS",
    "K_BOUNDS",
    "LENGTH_BOUNDS",
    "LJHead",
    "LinearAround",
    "PeriodicHead",
]

#: A force constant (bond, angle, ε) stays within these ratios of its reference.
K_BOUNDS: tuple[float, float] = (1.0 / 8.0, 8.0)
#: An improper amplitude may go lower: the strain objective sends a pyramidal
#: centre's to zero (p10 0.003 kcal/mol per row on a passing gate run).
IMPROPER_K_BOUNDS: tuple[float, float] = (1.0 / 64.0, 8.0)
#: A length (r0, σ) stays within these ratios of its reference. Two covalent
#: radii overestimate a triple bond by 1/0.79 (C≡C 1.52 vs 1.20 Å) and the
#: ratio never reaches 1.4 on a bonded pair.
LENGTH_BOUNDS: tuple[float, float] = (0.7, 1.4)


def _zero_last(mlp: nn.Sequential) -> nn.Sequential:
    """Zero the output layer so ``raw = 0`` and the head starts at its reference.

    Zeroing does not freeze the layer: the weight gradient is ``dL/draw · h`` and
    stays non-zero, so the head begins as a uniform field and learns to tell
    interactions apart.
    """
    last = mlp[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return mlp


def _side(x: torch.Tensor, width: float | torch.Tensor) -> torch.Tensor:
    """``width · tanh(x / width)``: ``x`` bent to saturate at *width*, slope one at zero.

    A tensor *width* is per row and may be zero (a reference sitting on its
    bound): the value is then held at the bound exactly and its gradient is
    zero, with no division by zero.
    """
    if isinstance(width, torch.Tensor):
        return width * torch.tanh(x / width.clamp_min(1e-6))
    return width * torch.tanh(x / width)


def _bend(
    x: torch.Tensor, down: float | torch.Tensor | None, up: float | torch.Tensor | None
) -> torch.Tensor:
    """``x`` bent smoothly into ``(-down, up)``, each side by :func:`_side`.

    ByteFF's ``tanh`` with the window as its asymptotes: the slope at ``0`` is
    one on both sides, the gradient fades as a bound is approached and is never
    passed through a held value. ``None`` leaves that side open (the identity).
    """
    if down is None and up is None:
        return x
    above = x if up is None else _side(x, up)
    below = x if down is None else _side(x, down)
    return torch.where(x >= 0, above, below)


class ExpAround(nn.Module):
    """``value = ref · exp(gain · raw)``: positive, no flat region, one factor ``e`` per unit.

    With *lo* / *hi* set the exponent saturates toward ``log lo`` / ``log hi``
    (:func:`_bend`), so the value stays inside ``[lo, hi] · ref``, a unit of
    ``raw`` is still a factor ``e^gain`` at the reference, and a row driven to
    a bound stops pulling on the features that feed it (ByteFF ``tanh``,
    ``mlparams.py:104-110``; see the module docstring for why not a
    straight-through clamp).

    Args:
        gain: Pre-activation scale. ``1`` for a force constant (a unit of
            ``raw`` is a factor ``e``), ``0.1`` for a length (a unit is 10 %).
        lo: Lower bound as a ratio to the reference, in ``(0, 1)``; ``None``
            leaves that side open.
        hi: Upper bound as a ratio to the reference, ``> 1``; ``None`` leaves
            that side open.
    """

    def __init__(
        self, gain: float = 1.0, *, lo: float | None = None, hi: float | None = None
    ) -> None:
        super().__init__()
        if gain <= 0.0:
            raise ValueError(f"gain must be positive, got {gain}")
        if lo is not None and lo <= 0.0:
            raise ValueError(f"lo is a ratio to the reference and must be positive, got {lo}")
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
        if (lo is not None and lo >= 1.0) or (hi is not None and hi <= 1.0):
            raise ValueError(
                f"the window must contain the reference (lo < 1 < hi), got lo={lo}, hi={hi}"
            )
        self.gain = float(gain)
        self.log_lo = None if lo is None else math.log(float(lo))
        self.log_hi = None if hi is None else math.log(float(hi))

    def forward(self, raw: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        down = None if self.log_lo is None else -self.log_lo
        exponent = _bend(self.gain * raw, down, self.log_hi)
        return ref * torch.exp(exponent)


class LinearAround(nn.Module):
    """``value = ref + gain · raw``, saturating toward ``lo`` / ``hi``.

    Linear at the reference — a unit of ``raw`` is *gain* there, on every range
    (ByteFF reaches ±149° per unit; a sigmoid's slope collapsed near 180°) —
    and bent by :func:`_bend` so the value never crosses the physical range
    and a row driven to a bound stops pulling on its features. A reference on
    a bound (a linear centre's 180°) is held there exactly on that side.

    Args:
        gain: Value per unit of ``raw``, in the parameter's unit.
        lo: Lower bound, same unit.
        hi: Upper bound, same unit.
    """

    def __init__(self, gain: float, lo: float, hi: float) -> None:
        super().__init__()
        if gain <= 0.0 or hi <= lo:
            raise ValueError(f"need gain > 0 and hi > lo, got gain={gain}, lo={lo}, hi={hi}")
        self.gain = float(gain)
        self.lo = float(lo)
        self.hi = float(hi)

    def forward(self, raw: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        below = (ref - self.lo).clamp_min(0.0)
        above = (self.hi - ref).clamp_min(0.0)
        return ref + _bend(self.gain * raw, below, above)


class CovalentRadiusSum(nn.Module):
    """``r_i + r_j`` from molpy's covalent radii — the bond head's reference length, Å.

    Args:
        num_elements: Table size (atomic numbers ``< num_elements``); row 0 is
            not an element and holds ``0``.
    """

    def __init__(self, num_elements: int = 119, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        radii = [0.0] + [float(Element(z).covalent) for z in range(1, int(num_elements))]
        # Derived from molpy, not learned: kept out of the state dict.
        self.register_buffer("covalent_radius", torch.tensor(radii, dtype=dtype), persistent=False)

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        """``(M,)`` reference lengths for bonds between atomic numbers ``zi`` and ``zj``."""
        return self.covalent_radius[zi.long()] + self.covalent_radius[zj.long()]


class _Head(nn.Module):
    """One MLP per parameter, every output layer zeroed.

    Subclasses declare :attr:`namespace` and :attr:`produces`; the forward
    takes the interaction's features and the per-row references and returns
    the parameters **and** their references (``<name>_ref``) so a prior can
    read both from the same table.
    """

    #: The batch namespace the head's parameters belong to.
    namespace: ClassVar[str]
    #: The parameters, one MLP each.
    produces: ClassVar[tuple[str, ...]]
    #: The reference columns ``forward`` returns beside them (``<name>_ref``).
    references: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        widths: dict[str, int],
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        #: The build dtype, for the constant buffers a subclass registers.
        self.dtype = dtype
        self.mlps = nn.ModuleDict(
            {
                name: _zero_last(scalar_mlp(feature_dim, [hidden_dim], width, dtype=dtype))
                for name, width in widths.items()
            }
        )

    def raw(self, name: str, features: torch.Tensor) -> torch.Tensor:
        """The dimensionless pre-activation of parameter *name*, ``(M,)`` or ``(M, T)``."""
        out = self.mlps[name](features)
        return out[:, 0] if out.shape[-1] == 1 else out


class BondHead(_Head):
    """Harmonic bond ``k`` (kcal/mol/Å², LAMMPS ``K``) and ``r0`` (Å).

    Args:
        feature_dim: Bond feature width.
        hidden_dim: Hidden width of each parameter MLP.
        k_ref: Reference force constant every bond starts at. GAFF2's bond
            ``K`` run 300-700 kcal/mol/Å² (``gaff2.dat``: c3-c3 300, c3-hc 330,
            c=o 640); the THEMol Hessian labels' 1-2 mixed-block trace median
            is −932 = −2K near equilibrium.
        k_gain: ``raw`` scale for ``k`` (a factor ``e`` per unit).
        r0_gain: ``raw`` scale for ``r0`` (a 10 % change per unit).
    """

    namespace = "bonds"
    produces = ("k", "r0")
    references = ("k_ref", "r0_ref")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        k_ref: float = 450.0,
        k_gain: float = 1.0,
        r0_gain: float = 0.1,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(feature_dim, hidden_dim, {"k": 1, "r0": 1}, dtype=dtype)
        if k_ref <= 0.0:
            raise ValueError(f"k_ref must be positive, got {k_ref}")
        self.k_ref = float(k_ref)
        self.k_map = ExpAround(k_gain, lo=K_BOUNDS[0], hi=K_BOUNDS[1])
        self.r0_map = ExpAround(r0_gain, lo=LENGTH_BOUNDS[0], hi=LENGTH_BOUNDS[1])

    def forward(self, features: torch.Tensor, *, r0_ref: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            features: ``(M, D)`` bond features.
            r0_ref: ``(M,)`` reference lengths (:class:`CovalentRadiusSum`).

        Returns:
            ``k``, ``r0``, ``k_ref``, ``r0_ref``, each ``(M,)``.
        """
        k_ref = r0_ref.new_full(r0_ref.shape, self.k_ref)
        return {
            "k": self.k_map(self.raw("k", features), k_ref),
            "r0": self.r0_map(self.raw("r0", features), r0_ref),
            "k_ref": k_ref,
            "r0_ref": r0_ref,
        }


class AngleHead(_Head):
    """Harmonic angle ``k`` (kcal/mol/rad², LAMMPS ``K``) and ``theta0`` (degrees).

    Args:
        feature_dim: Angle feature width.
        hidden_dim: Hidden width of each parameter MLP.
        k_ref: Reference force constant (GAFF2 angle ``K`` run 35-70).
        k_gain: ``raw`` scale for ``k``.
        theta0_gain_deg: Degrees per unit of ``raw``.
        theta0_lo_deg: Lower clamp, degrees.
        theta0_hi_deg: Upper clamp, degrees. **180° inclusive**: with ``θ0 =
            180°`` the harmonic ``K (θ − θ0)²`` is a smooth isotropic bend at a
            linear centre (the C^∞ kernel of ``molpot.potentials.geom`` is
            defined there), whereas any ``θ0 < 180°`` makes the linear geometry
            the top of a Mexican hat whose curvature ``2K (θ − θ0) ∂²θ/∂x²`` grows
            as ``1 / sin θ`` — the partial-Hessian fit then has no way to match
            the QM block but ``K → 0`` (an sp-carbon angle ``K`` fell 47 → 1.1
            over 40 000 steps with the ceiling at 179.5°).
    """

    namespace = "angles"
    produces = ("k", "theta0")
    references = ("k_ref", "theta0_ref")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        k_ref: float = 40.0,
        k_gain: float = 1.0,
        theta0_gain_deg: float = 20.0,
        theta0_lo_deg: float = 0.5,
        theta0_hi_deg: float = 180.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(feature_dim, hidden_dim, {"k": 1, "theta0": 1}, dtype=dtype)
        if k_ref <= 0.0:
            raise ValueError(f"k_ref must be positive, got {k_ref}")
        self.k_ref = float(k_ref)
        self.k_map = ExpAround(k_gain, lo=K_BOUNDS[0], hi=K_BOUNDS[1])
        self.theta0_map = LinearAround(theta0_gain_deg, theta0_lo_deg, theta0_hi_deg)

    def forward(
        self, features: torch.Tensor, *, theta0_ref: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Args:
            features: ``(M, D)`` angle features.
            theta0_ref: ``(M,)`` reference angles, **degrees** (the centre's
                VSEPR angle, :meth:`molrep.chem.Hybridization.angle_reference`).

        Returns:
            ``k``, ``theta0``, ``k_ref``, ``theta0_ref``, each ``(M,)``.
        """
        k_ref = theta0_ref.new_full(theta0_ref.shape, self.k_ref)
        return {
            "k": self.k_map(self.raw("k", features), k_ref),
            "theta0": self.theta0_map(self.raw("theta0", features), theta0_ref),
            "k_ref": k_ref,
            "theta0_ref": theta0_ref,
        }


class PeriodicHead(_Head):
    """Proper torsion as signed ``k_n`` (kcal/mol) at fixed phases — ``dihedral periodic``.

    Term ``n`` is ``k_n [1 + cos(nφ − γ_n)]`` with ByteFF's phases ``γ_n = 0``
    for odd ``n`` and ``180°`` for even ``n`` (``classical_forcefield.py:230-235``),
    and ``k_n`` free in sign — which is how GAFF2 and OPLS tables are written
    (``K_n = 2 k_n`` for the OPLS form). Every term is even in ``φ``, so a
    mirror image, which reverses every dihedral, has the same energy; a free
    phase or a sine term would not be. The amplitude per unit of ``raw`` is
    ``1/n`` kcal/mol, so a unit adds ``n`` (not ``n²``) kcal/mol/rad² of
    curvature, and the gradient ``∂E/∂k_n = 1 + cos(nφ − γ_n)`` does not vanish
    at the zero start (an ``|a|`` parameterization would).

    Args:
        feature_dim: Proper feature width.
        hidden_dim: Hidden width of the MLP.
        periodicity: The ``n`` of each term, default ``1..6``.
    """

    namespace = "propers"
    produces = ("k",)

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        periodicity: Sequence[int] = (1, 2, 3, 4, 5, 6),
        dtype: torch.dtype | None = None,
    ) -> None:
        n = tuple(int(p) for p in periodicity)
        if not n or min(n) < 1:
            raise ValueError(f"periodicity must be non-empty positive integers, got {n}")
        super().__init__(feature_dim, hidden_dim, {"k": len(n)}, dtype=dtype)
        self.register_buffer("periodicity", torch.tensor(n, dtype=torch.long))
        self.register_buffer(
            "phase",
            torch.tensor([180.0 * ((m + 1) % 2) for m in n], dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "amplitude_scale", 1.0 / torch.tensor(n, dtype=self.dtype), persistent=False
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"k": (M, T)}`` signed amplitudes; phases are :attr:`phase`, degrees."""
        return {"k": self.mlps["k"](features) * self.amplitude_scale}


class CvffHead(_Head):
    """Improper torsion amplitude ``k`` (kcal/mol) — ``improper cvff`` with ``sign = −1``.

    ``k [1 − cos(nψ)]`` at ``n = 2`` is ByteFF's improper (``k [1 + cos(2ψ − π)]``,
    ``classical_forcefield.py:261-284``): even in ``ψ``, zero in the plane. ``k``
    is positive by construction, ``k_ref · exp(k_gain · raw)`` like the bond and
    angle ``K``: the term's strain on a QM minimum, ``k (1 − cos nψ)``, is linear
    in ``k``, so an objective that minimises it (ByteFF ``ImproperTorsion_Energy``)
    is bounded below only for ``k ≥ 0`` — a signed ``k`` ran to −∞ on it (the
    strain fell 0 → −4.4 kcal/mol/term in 1 400 steps and every N3 ``θ0`` went
    to 163° to cancel the force). A negative ``k`` also puts the
    minimum at ``ψ = 90°``, which no improper means. One row of the ``impropers``
    namespace is one of the three cyclic leg orders of a centre (ByteFF splits
    ``k/3`` over them); the centre's amplitude is the sum over its rows.

    Args:
        feature_dim: Improper feature width.
        hidden_dim: Hidden width of the MLP.
        k_ref: Reference amplitude per row every improper starts at. ByteFF's
            GAFF2 statistics put the per-centre improper ``k`` at 4.29 ± 3.71
            kcal/mol (``definitions.py:113``); a third of that per row.
        k_gain: ``raw`` scale for ``k`` (a factor ``e`` per unit).
        periodicity: The ``n`` of the term, default ``2``.
    """

    namespace = "impropers"
    produces = ("k",)
    references = ("k_ref",)

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        k_ref: float = 4.291 / 3.0,
        k_gain: float = 1.0,
        periodicity: int = 2,
        dtype: torch.dtype | None = None,
    ) -> None:
        if periodicity < 1:
            raise ValueError(f"periodicity must be >= 1, got {periodicity}")
        if k_ref <= 0.0:
            raise ValueError(f"k_ref must be positive, got {k_ref}")
        super().__init__(feature_dim, hidden_dim, {"k": 1}, dtype=dtype)
        self.k_ref = float(k_ref)
        self.k_map = ExpAround(k_gain, lo=IMPROPER_K_BOUNDS[0], hi=IMPROPER_K_BOUNDS[1])
        self.register_buffer("periodicity", torch.tensor(int(periodicity), dtype=torch.long))
        # The cvff form ``k [1 + d cos(nψ)]`` with ``d = −1``: the head owns its form.
        self.register_buffer("sign", torch.tensor(-1.0, dtype=self.dtype), persistent=False)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"k": (M,), "k_ref": (M,)}``, positive amplitudes and their reference."""
        raw = self.raw("k", features)
        k_ref = raw.new_full(raw.shape, self.k_ref)
        return {"k": self.k_map(raw, k_ref), "k_ref": k_ref}


class LJHead(_Head):
    """Lennard-Jones ``sigma`` (Å) and ``epsilon`` (kcal/mol) around a per-atom reference.

    The reference is the atom's GAFF2 type's well, carried by the batch as
    ``atoms.gaff2_sigma`` / ``atoms.gaff2_epsilon`` (ByteFF regresses its
    predicted σ/ε to the same labels, ``loss.py:410-417``). With the output
    layer zeroed the model starts exactly on GAFF2; gas-phase intramolecular
    data gives these parameters little signal, so a training stage names them
    only when it also carries a prior toward the reference.

    Args:
        feature_dim: Atom feature width.
        hidden_dim: Hidden width of each parameter MLP.
        sigma_gain: ``raw`` scale for σ (10 % per unit).
        epsilon_gain: ``raw`` scale for ε (a factor ``e`` per unit).
    """

    namespace = "atoms"
    produces = ("sigma", "epsilon")
    references = ("sigma_ref", "epsilon_ref")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        sigma_gain: float = 0.1,
        epsilon_gain: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(feature_dim, hidden_dim, {"sigma": 1, "epsilon": 1}, dtype=dtype)
        self.sigma_map = ExpAround(sigma_gain, lo=LENGTH_BOUNDS[0], hi=LENGTH_BOUNDS[1])
        self.epsilon_map = ExpAround(epsilon_gain, lo=K_BOUNDS[0], hi=K_BOUNDS[1])

    def forward(
        self, features: torch.Tensor, *, sigma_ref: torch.Tensor, epsilon_ref: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Args:
            features: ``(N, D)`` atom features.
            sigma_ref: ``(N,)`` reference σ, Å.
            epsilon_ref: ``(N,)`` reference ε, kcal/mol.

        Returns:
            ``sigma``, ``epsilon``, ``sigma_ref``, ``epsilon_ref``, each ``(N,)``.
        """
        return {
            "sigma": self.sigma_map(self.raw("sigma", features), sigma_ref),
            "epsilon": self.epsilon_map(self.raw("epsilon", features), epsilon_ref),
            "sigma_ref": sigma_ref,
            "epsilon_ref": epsilon_ref,
        }
