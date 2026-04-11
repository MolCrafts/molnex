"""Angular embedding modules for molrep encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from cuequivariance_torch import SphericalHarmonics as CueSphericalHarmonics
from pydantic import BaseModel, Field

Key = str | tuple[str, ...]


class SphericalHarmonicsSpec(BaseModel):
    """Specification for spherical harmonics computation.

    Defines parameters for computing spherical harmonics Y_l^m from 3D vectors
    using cuEquivariance as the backend.

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
    """Spherical harmonics computation module.

    Computes spherical harmonics Y_l^m(v) for input 3D vectors v using
    cuEquivariance as the backend. This is a wrapper around
    cuequivariance_torch.SphericalHarmonics.

    The output contains all spherical harmonics from l=0 to l=l_max,
    ordered as [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ...].

    For l=0: 1 component (s-orbital)
    For l=1: 3 components (p-orbitals)
    For l=2: 5 components (d-orbitals)
    Total dimensions: (l_max + 1)^2

    Attributes:
        config: SphericalHarmonicsSpec configuration.
        sh: cuEquivariance SphericalHarmonics backend module.
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

        self.config = SphericalHarmonicsSpec(
            l_max=l_max,
            normalize=normalize,
        )
        self.l_max = int(self.config.l_max)

        # Initialize cuEquivariance SphericalHarmonics backend
        self.sh = CueSphericalHarmonics(ls=self.config.ls, normalize=self.config.normalize)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics from 3D vectors.

        Args:
            vectors: Input 3D vectors. Shape: (..., 3)

        Returns:
            Spherical harmonics. Output shape: (..., (l_max + 1)^2)

        Note:
            Input vectors do not need to be normalized; cuEquivariance handles
            normalization internally.
        """
        return self.sh(vectors)
