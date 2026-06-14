"""Potential built from cubic spline interpolation of tabulated data."""

from typing import Optional

import torch

from molpot.potentials.elec.lib.splines import (
    CubicSpline,
    CubicSplineReciprocal,
    compute_second_derivatives,
    compute_spline_ft,
)
from molpot.potentials.elec.potentials.potential import Potential


class SplinePotential(Potential):
    """Potential built from spline-interpolated tabulated values.

    The potential is assumed to have only a long-range part, but a
    short-range part can be added by subclassing. Real-space values
    come from a cubic spline; Fourier-domain values are computed
    numerically from the spline coefficients.

    Args:
        r_grid: Radial grid points for real-space ``(n_grid,)``.
        y_grid: Potential values on the radial grid ``(n_grid,)``.
        k_grid: Radial grid for k-space; auto-computed from ``r_grid`` if absent.
        yhat_grid: Potential values in k-space; auto-computed if absent.
        reciprocal: If True, spline on :math:`1/r` axis (for long-range tails).
        y_at_zero: Value for :math:`r \\to 0` with reciprocal spline.
        yhat_at_zero: Value for :math:`k \\to 0` in the k-space kernel.
        smearing: Length scale for range separation.
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff.
        prefactor: Potential prefactor.
    """

    def __init__(
        self,
        r_grid: torch.Tensor,
        y_grid: torch.Tensor,
        k_grid: Optional[torch.Tensor] = None,
        yhat_grid: Optional[torch.Tensor] = None,
        reciprocal: Optional[bool] = False,
        y_at_zero: Optional[float] = None,
        yhat_at_zero: Optional[float] = None,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__(
            smearing=smearing,
            exclusion_radius=exclusion_radius,
            exclusion_degree=exclusion_degree,
            prefactor=prefactor,
        )

        if len(y_grid) != len(r_grid):
            raise ValueError("Length of radial grid and value array mismatch.")

        self.register_buffer("r_grid", r_grid)
        self.register_buffer("y_grid", y_grid)

        if reciprocal:
            if torch.min(r_grid) <= 0.0:
                raise ValueError(
                    "Positive-valued radial grid is needed for reciprocal axis spline."
                )
            self._spline = CubicSplineReciprocal(r_grid, y_grid, y_at_zero=y_at_zero)
        else:
            self._spline = CubicSpline(r_grid, y_grid)

        if k_grid is None:
            if reciprocal:
                k_grid = torch.pi * 2 * torch.reciprocal(r_grid).flip(dims=[0])
            else:
                k_grid = r_grid.clone().detach()

        self.register_buffer("k_grid", k_grid)

        if yhat_grid is None:
            yhat_grid = compute_spline_ft(
                k_grid,
                r_grid,
                y_grid,
                compute_second_derivatives(r_grid, y_grid),
            )

        self.register_buffer("yhat_grid", yhat_grid)

        if reciprocal:
            self._krn_spline = CubicSplineReciprocal(k_grid**2, yhat_grid, y_at_zero=yhat_at_zero)
        else:
            self._krn_spline = CubicSpline(k_grid**2, yhat_grid)

        if y_at_zero is None:
            self._y_at_zero = self._spline(self.r_grid.new_zeros(1))
        else:
            self._y_at_zero = torch.tensor(
                y_at_zero, dtype=self.r_grid.dtype, device=self.r_grid.device
            )

        if yhat_at_zero is None:
            self._yhat_at_zero = self._krn_spline(self.k_grid.new_zeros(1))
        else:
            self._yhat_at_zero = torch.tensor(
                yhat_at_zero, dtype=self.k_grid.dtype, device=self.k_grid.device
            )

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full potential from spline (LR + SR).

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Potential values ``(n_edges,)``.
        """
        return self.prefactor * (
            self.lr_from_dist(dist, pair_mask) + self.sr_from_dist(dist, pair_mask)
        )

    def sr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Short-range part (zero by default for spline potentials).

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Zero tensor matching ``dist`` shape.
        """
        return 0.0 * dist

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Long-range part from the real-space spline.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Long-range potential values ``(n_edges,)``.
        """
        return self.prefactor * self._spline(dist)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Fourier-domain kernel from the k-space spline.

        Args:
            k_sq: Squared k-vector norms ``(...)``.

        Returns:
            Fourier-domain potential values.
        """
        return self.prefactor * self._krn_spline(k_sq)

    def self_contribution(self) -> torch.Tensor:
        """Self-interaction correction from spline at r=0.

        Returns:
            Scalar correction.
        """
        return self.prefactor * self._y_at_zero

    def background_correction(self) -> torch.Tensor:
        """Background correction (zero for spline potentials).

        Returns:
            Scalar zero.
        """
        return self.prefactor * torch.zeros(1)
