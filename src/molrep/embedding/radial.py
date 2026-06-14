from __future__ import annotations

import math

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from molix import config


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
        phi = self._raw_basis(r)
        if self.normalize:
            phi = (phi - self.mu) / self.sigma
        return phi


class PolynomialBasisSpec(BaseModel):
    """Configuration for polynomial radial basis."""

    powers: list[int] = Field(..., min_length=1)


class PolynomialBasis(nn.Module):
    """Polynomial radial basis: ``fc ** power`` for each power in ``[1..n_basis]``.

    A cutoff vector ``fc`` is required — unlike BesselRBF, the polynomial basis
    itself does not apply a cutoff; it only shapes the envelope passed in.
    """

    def __init__(self, n_basis: int) -> None:
        super().__init__()
        powers = list(range(1, int(n_basis) + 1))
        if not powers:
            raise ValueError("n_basis must define at least one polynomial power.")
        self.config = PolynomialBasisSpec(powers=powers)
        self.register_buffer("powers", torch.tensor(powers, dtype=config.ftype), persistent=False)
        self.powers: torch.Tensor

    @property
    def output_dim(self) -> int:
        """Number of radial basis channels."""
        return int(self.powers.numel())

    def forward(self, r: torch.Tensor, *, fc: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial basis channels.

        Args:
            r: Distances ``(E,)``. Unused — present for API symmetry.
            fc: Cutoff values ``(E,)``.

        Returns:
            Basis tensor ``(E, output_dim)``.
        """
        del r
        return fc.unsqueeze(-1).pow(self.powers)


class GaussianBasisSpec(BaseModel):
    """Configuration for Gaussian radial basis."""

    centers: list[float] = Field(..., min_length=1)
    gamma: list[float] = Field(..., min_length=1)


class GaussianBasis(nn.Module):
    """Gaussian radial basis: ``exp(-gamma * (r - center)^2)``.

    Centers are auto-spaced linearly over ``[0, r_cut]`` when not provided.
    An optional cutoff ``fc`` can be multiplied in.
    """

    def __init__(
        self,
        *,
        center: float | list[float] | None = None,
        gamma: float | list[float] = 3.0,
        r_cut: float,
        n_basis: int,
    ) -> None:
        super().__init__()
        if center is None:
            centers = torch.linspace(0.0, float(r_cut), int(n_basis), dtype=config.ftype)
        elif isinstance(center, (int, float)):
            centers = torch.full((int(n_basis),), float(center), dtype=config.ftype)
        else:
            centers = torch.tensor(list(center), dtype=config.ftype)
        if centers.numel() != int(n_basis):
            raise ValueError(
                f"center must contain n_basis={n_basis} values, got {centers.numel()}."
            )

        if isinstance(gamma, (int, float)):
            gammas = torch.full_like(centers, float(gamma))
        else:
            gammas = torch.tensor(list(gamma), dtype=config.ftype)
            if gammas.numel() == 1:
                gammas = gammas.expand_as(centers).clone()
        if gammas.numel() != centers.numel():
            raise ValueError(
                f"gamma must be scalar or contain {centers.numel()} values, got {gammas.numel()}."
            )

        self.config = GaussianBasisSpec(
            centers=[float(v) for v in centers.tolist()],
            gamma=[float(v) for v in gammas.tolist()],
        )
        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("gamma", gammas, persistent=False)
        self.centers: torch.Tensor
        self.gamma: torch.Tensor

    @property
    def output_dim(self) -> int:
        """Number of radial basis channels."""
        return int(self.centers.numel())

    def forward(self, r: torch.Tensor, *, fc: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate Gaussian basis channels.

        Args:
            r: Distances ``(E,)``.
            fc: Optional cutoff values ``(E,)``.

        Returns:
            Basis tensor ``(E, output_dim)``.
        """
        rr = r.to(dtype=self.centers.dtype).unsqueeze(-1)
        basis = torch.exp(-self.gamma * (rr - self.centers).pow(2))
        if fc is not None:
            basis = basis * fc.unsqueeze(-1)
        return basis
