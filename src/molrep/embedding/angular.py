"""Spherical harmonics of edge directions, on cuEquivariance.

:class:`SphericalHarmonics` wraps ``cuequivariance_torch.SphericalHarmonics``
(``O3`` convention, ``cue.ir_mul`` channel order
``[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, …]``). Forces go through the
autograd :class:`molix.autograd.Derivative` (first and second derivatives —
force training is a double backward), which the cuEquivariance module
supports; the earlier pure-torch monomial evaluation existed only for
functorch transforms molnex no longer uses.

Reference:
    Real spherical harmonics in the cuEquivariance ``O3`` convention
    (``cuequivariance_torch.SphericalHarmonics``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from molrep import cueq


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


class SphericalHarmonics(nn.Module):
    """Spherical harmonics ``Y_l^m`` up to ``l_max`` of 3-vectors (cuEquivariance).

    Args:
        l_max: Maximum angular momentum.
        normalize: Normalise the input direction before evaluation.

    Attributes:
        config: :class:`SphericalHarmonicsSpec`.
        l_max: Maximum angular momentum.
        normalize: Whether the input direction is normalised.
    """

    def __init__(self, *, l_max: int, normalize: bool = True, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SphericalHarmonicsSpec(l_max=l_max, normalize=normalize)
        self.l_max = int(self.config.l_max)
        self.normalize = bool(self.config.normalize)
        self.sh = cueq.SphericalHarmonics(
            self.config.ls, normalize=self.normalize, math_dtype=dtype
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Evaluate on ``vectors`` ``(..., 3)``; returns ``(..., (l_max + 1)²)``."""
        flat = vectors.reshape(-1, 3)
        return self.sh(flat).reshape(*vectors.shape[:-1], -1)
