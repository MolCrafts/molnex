"""Angular embedding modules for molrep encoders.

The spherical harmonics are evaluated in **pure PyTorch** rather than through
``cuequivariance_torch.SphericalHarmonics``. cuEquivariance's spherical-harmonics
kernel is a custom ``autograd.Function`` that does not implement ``setup_context``,
so it cannot be traced by functorch transforms (``torch.func.grad`` / ``vmap``) —
which is exactly how molnex computes forces (``molpot.derivation.ForceDerivation``)
and how the force path is made ``torch.compile``-able. A pure-torch evaluation is
functorch- and ``torch.compile``-traceable, and is built from cuEquivariance's own
symbolic polynomials (``cue.descriptors.sympy_spherical_harmonics``) so it matches
the cuEquivariance convention (including ``normalize``) to machine precision.

Reference:
    Real spherical harmonics in the cuEquivariance ``O3``/``SO3`` convention; the
    coefficients are taken verbatim from
    ``cuequivariance.descriptors.sympy_spherical_harmonics``.
"""

from __future__ import annotations

import cuequivariance as cue
import sympy
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from molix import config

Key = str | tuple[str, ...]


class SphericalHarmonicsSpec(BaseModel):
    """Specification for spherical harmonics computation.

    Defines parameters for computing spherical harmonics Y_l^m from 3D vectors.

    Attributes:
        l_max: Maximum angular momentum number. Must be non-negative.
            Output will contain all orders from l=0 to l=l_max.
        normalize: Whether to use normalized spherical harmonics. Defaults to True.
            Normalized harmonics satisfy orthonormality on the unit sphere.
    """

    l_max: int = Field(..., ge=0)
    normalize: bool = True

    @property
    def ls(self) -> list[int]:
        """Return list of angular momentum orders.

        Returns:
            List [0, 1, 2, ..., l_max].
        """
        return list(range(self.l_max + 1))

    @property
    def output_dim(self) -> int:
        """Calculate total output dimension.

        Returns:
            Sum of (2*l + 1) for l in [0, l_max], which equals (l_max + 1)^2.
        """
        return (self.l_max + 1) ** 2


def _monomial_table(ls: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the spherical-harmonics monomials from cuEquivariance's polynomials.

    For every ``l`` in ``ls`` and every component ``m`` the symbolic polynomial
    ``Y_l^m(x0, x1, x2)`` (cuEquivariance convention) is expanded into monomials
    ``c * x0^a * x1^b * x2^d``. Returns a flat term table.

    Args:
        ls: Angular momentum orders to include.

    Returns:
        Tuple ``(exps, coeffs, out_idx)`` where ``exps`` is ``(T, 3)`` integer
        exponents per term, ``coeffs`` is ``(T,)`` term coefficients, and
        ``out_idx`` is ``(T,)`` the output channel ``l**2 + m`` each term sums into.
    """
    gens = sympy.symbols("x0 x1 x2")
    exps: list[tuple[int, int, int]] = []
    coeffs: list[float] = []
    out_idx: list[int] = []
    for l in ls:
        _, arr = cue.descriptors.sympy_spherical_harmonics(cue.SO3(1), l)
        arr = sympy.Array(arr)
        for m in range(2 * l + 1):
            poly = sympy.Poly(sympy.expand(arr[m]), *gens)
            for monom, coeff in poly.terms():
                exps.append((int(monom[0]), int(monom[1]), int(monom[2])))
                coeffs.append(float(coeff))
                out_idx.append(l * l + m)
    return (
        torch.tensor(exps, dtype=torch.long),
        # Use the configured runtime float dtype, not the global torch default
        # (float32). The SH coefficients carry irrational factors (sqrt(15), …);
        # truncating them to float32 caps l>=2 rotation equivariance at ~1e-8,
        # which no later `.double()` can recover (the bits are already gone).
        torch.tensor(coeffs, dtype=config.ftype),
        torch.tensor(out_idx, dtype=torch.long),
    )


class SphericalHarmonics(nn.Module):
    """Spherical harmonics computation module (pure-torch, functorch-safe).

    Computes spherical harmonics Y_l^m(v) for input 3D vectors v. The output
    contains all spherical harmonics from l=0 to l=l_max, ordered as
    ``[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ...]`` — the cuEquivariance
    ``ir_mul`` channel order, so it is a drop-in for the previous
    ``cuequivariance_torch.SphericalHarmonics`` backend.

    For l=0: 1 component (s-orbital). For l=1: 3 (p). For l=2: 5 (d). Total
    dimension ``(l_max + 1)^2``.

    The evaluation uses a pow-free monomial table (per-component power tables built
    by cumulative multiplication, then gathered) so the gradient is well-defined at
    the origin and the whole forward is ``torch.compile`` / functorch traceable.

    Attributes:
        config: SphericalHarmonicsSpec configuration.
        l_max: Maximum angular momentum number.
        normalize: Whether the input direction is normalised before evaluation.
    """

    def __init__(
        self,
        *,
        l_max: int,
        normalize: bool = True,
    ):
        """Initialize spherical harmonics module.

        Args:
            l_max: Maximum angular momentum number.
            normalize: Whether to use normalized spherical harmonics.
        """
        super().__init__()

        self.config = SphericalHarmonicsSpec(l_max=l_max, normalize=normalize)
        self.l_max = int(self.config.l_max)
        self.normalize = bool(self.config.normalize)
        self._out_dim = self.config.output_dim
        self._max_exp = max(1, self.l_max)

        exps, coeffs, out_idx = _monomial_table(self.config.ls)
        self.register_buffer("_exp_x", exps[:, 0].contiguous(), persistent=False)
        self.register_buffer("_exp_y", exps[:, 1].contiguous(), persistent=False)
        self.register_buffer("_exp_z", exps[:, 2].contiguous(), persistent=False)
        self.register_buffer("_coeffs", coeffs, persistent=False)
        self.register_buffer("_out_idx", out_idx, persistent=False)

    def _power_table(self, comp: torch.Tensor) -> torch.Tensor:
        """Stack ``comp**0 .. comp**max_exp`` via cumulative product (no ``pow``).

        Args:
            comp: A single coordinate component ``(...,)``.

        Returns:
            Power table ``(..., max_exp + 1)``.
        """
        powers = [torch.ones_like(comp)]
        cur = powers[0]
        for _ in range(self._max_exp):
            cur = cur * comp
            powers.append(cur)
        return torch.stack(powers, dim=-1)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics from 3D vectors.

        Args:
            vectors: Input 3D vectors ``(..., 3)``. Need not be normalized when
                ``normalize=True``.

        Returns:
            Spherical harmonics ``(..., (l_max + 1)^2)``.
        """
        v = vectors
        if self.normalize:
            # Clamp the norm so a zero-length vector (the origin) yields v=0
            # rather than 0/0 = NaN. The module contract promises origin-safety;
            # real edges are never zero-length, so this only guards the degenerate
            # input. l=0 stays its constant; l>0 monomials vanish at v=0.
            norm = torch.linalg.norm(v, dim=-1, keepdim=True)
            v = v / norm.clamp_min(torch.finfo(v.dtype).tiny)

        x_pow = self._power_table(v[..., 0])
        y_pow = self._power_table(v[..., 1])
        z_pow = self._power_table(v[..., 2])

        mono = (
            torch.index_select(x_pow, -1, self._exp_x)
            * torch.index_select(y_pow, -1, self._exp_y)
            * torch.index_select(z_pow, -1, self._exp_z)
            * self._coeffs
        )
        out = torch.zeros(
            *v.shape[:-1], self._out_dim, dtype=v.dtype, device=v.device
        )
        return out.index_add(-1, self._out_idx, mono)
