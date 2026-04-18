from __future__ import annotations

import math

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class BesselRBFSpec(BaseModel):
    """Specification for Bessel radial basis function.

    Defines parameters for computing Bessel RBF features from distance values.
    The Bessel RBF provides a smooth, localized representation of distances
    that is commonly used in message-passing neural networks.

    Attributes:
        r_cut: Cutoff radius. Distances are normalized by this value.
            Must be positive.
        num_radial: Number of radial basis functions. Must be positive.
        eps: Small constant to avoid division by zero. Defaults to 1e-8.
        normalize: Whether to apply shift+scale normalization per the Allegro
            SI. When True, each basis function is standardized to zero mean /
            unit variance assuming r ~ Uniform([0, r_cut]). Defaults to True.
        normalize_samples: Number of quadrature samples used to estimate μ_n
            and σ_n at init time. Defaults to 4096.
    """

    r_cut: float = Field(..., gt=0)
    num_radial: int = Field(..., gt=0)
    eps: float = 1e-8
    normalize: bool = True
    normalize_samples: int = Field(4096, gt=0)


class BesselRBF(nn.Module):
    """Bessel radial basis function module.

    Computes Bessel RBF features from distance values using the formula:
        phi_n(r) = sqrt(2/r_cut) * sin(n*pi*r/r_cut) / (r + eps)

    When ``normalize=True`` (Allegro SI convention) the basis is additionally
    shifted and scaled so that each channel has zero mean and unit variance
    under the assumption r ~ Uniform([0, r_cut]):
        B_n(r) = (phi_n(r) - μ_n) / σ_n

    The statistics μ_n, σ_n are computed numerically once at construction time
    via a dense Riemann sum and stored as non-trainable buffers.

    Attributes:
        config: BesselRBFSpec configuration.
        freqs: Buffer storing frequency values n*pi/r_cut.
        prefactor: Buffer storing normalization constant sqrt(2/r_cut).
        eps: Small constant for numerical stability.
        normalize: Whether shift+scale normalization is applied.
        mu, sigma: Buffers with per-channel statistics (only when normalize).

    Input shape:
        r: (...,) tensor of distance values.

    Output shape:
        phi: (..., num_radial) tensor of RBF features.
    """

    def __init__(
        self,
        *,
        r_cut: float,
        num_radial: int,
        eps: float = 1e-8,
        normalize: bool = True,
        normalize_samples: int = 4096,
    ) -> None:
        """Initialize Bessel RBF module.

        Args:
            r_cut: Cutoff radius for normalization.
            num_radial: Number of radial basis functions.
            eps: Small constant to avoid division by zero. Defaults to 1e-8.
            normalize: Apply Allegro-SI shift+scale normalization.
            normalize_samples: Grid size for μ_n / σ_n estimation.
        """
        super().__init__()

        self.config = BesselRBFSpec(
            r_cut=r_cut,
            num_radial=num_radial,
            eps=eps,
            normalize=normalize,
            normalize_samples=normalize_samples,
        )

        self.r_cut = float(self.config.r_cut)
        num = int(self.config.num_radial)

        freqs = torch.arange(1, num + 1, dtype=torch.float32) * (math.pi / self.r_cut)
        self.register_buffer("freqs", freqs, persistent=False)
        self.freqs: torch.Tensor

        prefactor = torch.tensor(math.sqrt(2.0 / self.r_cut), dtype=torch.float32)
        self.register_buffer("prefactor", prefactor, persistent=False)
        self.prefactor: torch.Tensor

        self.eps = float(self.config.eps)
        self.normalize = bool(self.config.normalize)

        if self.normalize:
            mu, sigma = self._compute_stats(self.config.normalize_samples)
        else:
            mu = torch.zeros(num, dtype=torch.float32)
            sigma = torch.ones(num, dtype=torch.float32)
        self.register_buffer("mu", mu, persistent=False)
        self.register_buffer("sigma", sigma, persistent=False)
        self.mu: torch.Tensor
        self.sigma: torch.Tensor

    def _raw_basis(self, r: torch.Tensor) -> torch.Tensor:
        """Compute raw (un-normalised) Bessel basis."""
        rr = r.unsqueeze(-1)
        return self.prefactor * torch.sin(rr * self.freqs) / (rr + self.eps)

    @torch.no_grad()
    def _compute_stats(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate μ_n and σ_n under r ~ Uniform([eps, r_cut]).

        Uniform sampling over ``[eps, r_cut]`` avoids the r=0 singularity while
        matching the Allegro SI's stated assumption to within ``eps``.
        """
        r = torch.linspace(self.eps, self.r_cut, n_samples, dtype=torch.float32)
        phi = self._raw_basis(r)
        mu = phi.mean(dim=0)
        sigma = phi.std(dim=0).clamp(min=1e-8)
        return mu, sigma

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute Bessel RBF features from distances.

        Args:
            r: Input distances. Expected shape: (...,)

        Returns:
            RBF features. Output shape: (..., num_radial)
        """
        r = r.float()
        phi = self._raw_basis(r)
        if self.normalize:
            phi = (phi - self.mu) / self.sigma
        return phi
