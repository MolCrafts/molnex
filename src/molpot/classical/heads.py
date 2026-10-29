"""Class-I parameter heads: features → force-field parameters.

Two families live here.

* **the published model's bonded heads** (:class:`SymmetricHead`, :class:`TanhWindow`).
  A bonded term's input is assembled from the atom and bond features of its
  own atoms and bonds, in every order that names the same term (a bond and its
  reverse, an angle and its reverse, a proper and its reverse, the three legs
  of an improper). One MLP reads each order, the results are summed — so the
  parameters are invariant under those permutations by construction — and a
  second MLP and one MLP per parameter follow. The parameter is
  ``tanh_output · tanh(raw) · std + mean``, clamped to a range with a
  straight-through gradient. The constants are the published model's (GAFF2 statistics).
  Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md`` §3.
* **Rondo's D2 Lennard-Jones head** (:class:`LJHead`): σ and ε as a bounded
  window around the atom's GAFF2 well. One MLP per parameter whose last layer
  starts at zero, so the model starts exactly on GAFF2; the window is reached by
  saturation (:class:`ExpAround`), not by a clamp, because a loss that wants a
  row at a bound must stop pulling on the features that feed it.

Every head writes molrs's conventions (LAMMPS ``K`` without ½, degrees,
kcal/mol, Å); the conversion from the published model's ``½k`` happens in the term that
owns the style (:mod:`molpot.classical.terms`).

Reference:
    Zheng et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817 — reference implementation
    ``ByteDance-Seed/byteff2@v1.1.0``,
    ``model/ff_layers/mm_bonded.py:PreMMBonded``.
    Wang et al. "Development and testing of a general amber force field."
    J. Comput. Chem. 25, 1157 (2004). https://doi.org/10.1002/jcc.20035
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch
import torch.nn as nn

from molpot.heads._common import scalar_mlp
from molrep.embedding.mlp import DenseMLP

__all__ = [
    "TanhWindow",
    "ExpAround",
    "K_BOUNDS",
    "LENGTH_BOUNDS",
    "LJHead",
    "SymmetricHead",
]

#: A force constant (ε) stays within these ratios of its reference.
K_BOUNDS: tuple[float, float] = (1.0 / 8.0, 8.0)
#: A length (σ) stays within these ratios of its reference.
LENGTH_BOUNDS: tuple[float, float] = (0.7, 1.4)


class TanhWindow(nn.Module):
    """The published model's output transform of one parameter.

    Reference ``mm_bonded.py:PreMMBonded.forward``.

    ``value = clamp(tanh_output · tanh(raw) · std + mean, lo, hi)``, the clamp
    passing the gradient straight through (``mm_bonded.py:CustomClamp``).

    Args:
        std: Scale of the saturated output.
        mean: Value at ``raw = 0``.
        lo: Lower clamp.
        hi: Upper clamp.
        tanh_output: Saturation of the ``tanh``.
    """

    def __init__(
        self, std: float, mean: float, lo: float, hi: float, *, tanh_output: float
    ) -> None:
        super().__init__()
        self.std = float(std)
        self.mean = float(mean)
        self.lo = float(lo)
        self.hi = float(hi)
        self.tanh_output = float(tanh_output)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """The parameter, same shape as *raw*."""
        value = self.tanh_output * torch.tanh(raw) * self.std + self.mean
        # Forward is the clamped value exactly; the gradient is the identity.
        return value.clamp(self.lo, self.hi).detach() + (value - value.detach())


class SymmetricHead(nn.Module):
    """the published model's head of one bonded term (``PreMMBonded._symmetric_pooling``).

    ``y = post(Σ_copies pre(copy))``, then one MLP per parameter on ``y``. The
    caller passes every ordering of the term's atom and bond features; the sum
    makes the output invariant under them.

    Args:
        copy_dim: Width of one assembled copy (atoms · atom width + bonds ·
            bond width).
        params: Output width of each parameter.
        hidden_dim: Hidden width of ``pre`` and ``post``.
        width: Output width of ``pre`` and ``post``.
        layers: Linears in ``pre`` and in ``post``.
        out_hidden_dim: Hidden width of every parameter MLP.
        out_layers: Linears in every parameter MLP.
        dtype: Floating dtype of every parameter.
    """

    def __init__(
        self,
        copy_dim: int,
        params: Mapping[str, int],
        *,
        hidden_dim: int,
        width: int,
        layers: int,
        out_hidden_dim: int,
        out_layers: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.pre = DenseMLP(copy_dim, hidden_dim, width, layers, dtype=dtype)
        self.post = DenseMLP(width, hidden_dim, width, layers, dtype=dtype)
        self.out = nn.ModuleDict(
            {
                name: DenseMLP(width, out_hidden_dim, size, out_layers, dtype=dtype)
                for name, size in params.items()
            }
        )

    def raw(self, copies: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """The pre-transform output of every parameter, ``(M, width)`` each."""
        pooled = self.pre(copies[0])
        for copy in copies[1:]:
            pooled = pooled + self.pre(copy)
        y = self.post(pooled)
        return {name: mlp(y) for name, mlp in self.out.items()}


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

    the published model's ``tanh`` with the window as its asymptotes: the slope at ``0`` is
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
    a bound stops pulling on the features that feed it (the published ``tanh``,
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
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
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


class LJHead(_Head):
    """Lennard-Jones ``sigma`` (Å) and ``epsilon`` (kcal/mol) around a per-atom reference.

    The reference is the atom's GAFF2 type's well, carried by the batch as
    ``atoms.gaff2_sigma`` / ``atoms.gaff2_epsilon`` (the published model regresses its
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
        dtype: torch.dtype,
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
