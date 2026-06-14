"""Cubic spline utilities for real-space and reciprocal-space interpolation."""

from typing import Optional

import torch


class CubicSpline(torch.nn.Module):
    """Cubic spline interpolator for a real-valued function.

    Given ``(x, y)`` grid points, constructs a natural cubic spline
    and evaluates it at arbitrary query points.

    Args:
        x_points: Abscissas ``(n_points,)``.
        y_points: Ordinates ``(n_points,)``.
    """

    def __init__(self, x_points: torch.Tensor, y_points: torch.Tensor):
        super().__init__()

        self.x_points = x_points
        self.y_points = y_points
        self.d2y_points = compute_second_derivatives(x_points, y_points)
        self._intervals = self.x_points[1:] - self.x_points[:-1]
        self._h2over6 = self._intervals**2 / 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the spline at query points.

        Args:
            x: Query positions ``(...)``.

        Returns:
            Interpolated values matching ``x`` shape.
        """
        i = torch.searchsorted(self.x_points, x, right=True) - 1
        i = torch.clamp(i, 0, len(self.x_points) - 2)

        h = self._intervals[i]
        a = (self.x_points[i + 1] - x) / h
        b = (x - self.x_points[i]) / h
        h2over6 = self._h2over6[i]
        return a * (self.y_points[i] + (a * a - 1) * self.d2y_points[i] * h2over6) + b * (
            self.y_points[i + 1] + (b * b - 1) * self.d2y_points[i + 1] * h2over6
        )


class CubicSplineReciprocal(torch.nn.Module):
    """Cubic spline on a :math:`1/x` axis.

    Splines on an inverse grid, extending smoothly to zero as
    :math:`x \\to \\infty`. Suitable for long-range potential tails.

    Args:
        x_points: Abscissas ``(n_points,)``, must be strictly positive.
        y_points: Ordinates ``(n_points,)``.
        y_at_zero: Value at :math:`x = 0`. Defaults to ``y_points[0]``.
    """

    def __init__(
        self,
        x_points: torch.Tensor,
        y_points: torch.Tensor,
        y_at_zero: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        ix_points = torch.cat(
            [
                torch.zeros((1,), dtype=x_points.dtype, device=x_points.device),
                torch.reciprocal(torch.flip(x_points, dims=[0])),
            ],
            dim=0,
        )
        iy_points = torch.cat(
            [
                torch.zeros((1,), dtype=x_points.dtype, device=x_points.device),
                torch.flip(y_points, dims=[0]),
            ],
            dim=0,
        )
        self._rev_spline = CubicSpline(ix_points, iy_points)

        if y_at_zero is None:
            y_at_zero = y_points[0]
        self._y_at_zero = y_at_zero
        self._zero_spline = CubicSpline(
            torch.tensor(
                [0.0, x_points[0], x_points[1]],
                dtype=x_points.dtype,
                device=x_points.device,
            ),
            torch.tensor(
                [self._y_at_zero, y_points[0], y_points[1]],
                dtype=x_points.dtype,
                device=x_points.device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the reciprocal spline.

        Args:
            x: Query positions ``(...)``.

        Returns:
            Interpolated values matching ``x`` shape.
        """
        safe_x = torch.where(x < self._zero_spline.x_points[1], self._zero_spline.x_points[1], x)
        return torch.where(
            x < self._zero_spline.x_points[1],
            self._zero_spline(x),
            self._rev_spline(torch.reciprocal(safe_x)),
        )


def _solve_tridiagonal(a, b, c, d):
    """Solve a tridiagonal linear system.

    Args:
        a: Sub-diagonal ``(n,)``.
        b: Main diagonal ``(n,)``.
        c: Super-diagonal ``(n,)``.
        d: Right-hand side ``(n,)``.

    Returns:
        Solution vector ``(n,)``.
    """
    n = len(d)
    c_prime = torch.zeros_like(d)
    d_prime = torch.zeros_like(d)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x = torch.zeros_like(d)
    x[-1] = d_prime[-1]
    for i in reversed(range(n - 1)):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def compute_second_derivatives(
    x_points: torch.Tensor,
    y_points: torch.Tensor,
) -> torch.Tensor:
    """Compute second derivatives for cubic spline construction.

    Uses natural spline boundary conditions (zero at endpoints).

    Args:
        x_points: Abscissas ``(n_points,)``.
        y_points: Ordinates ``(n_points,)``.

    Returns:
        Second derivatives ``(n_points,)``.
    """
    x = x_points
    y = y_points

    intervals = x[1:] - x[:-1]
    dy = (y[1:] - y[:-1]) / intervals

    n = len(x)
    a = torch.zeros_like(x)
    b = torch.zeros_like(x)
    c = torch.zeros_like(x)
    d = torch.zeros_like(x)

    b[0] = 1
    d[0] = 0
    b[-1] = 1
    d[-1] = 0

    for i in range(1, n - 1):
        a[i] = intervals[i - 1] / 6
        b[i] = (intervals[i - 1] + intervals[i]) / 3
        c[i] = intervals[i] / 6
        d[i] = dy[i] - dy[i - 1]

    return _solve_tridiagonal(a, b, c, d)


def compute_spline_ft(
    k_points: torch.Tensor,
    x_points: torch.Tensor,
    y_points: torch.Tensor,
    d2y_points: torch.Tensor,
) -> torch.Tensor:
    """Compute the Fourier transform of a splined radial function.

    Evaluates:

    .. math::

        \\hat{f}(k) = 4\\pi \\int dr \\, \\frac{\\sin(kr)}{k} \\, r f(r)

    Includes a tail correction beyond the last splined point.

    Args:
        k_points: Target k values ``(n_k,)``.
        x_points: Spline abscissas ``(n_points,)``.
        y_points: Spline ordinates ``(n_points,)``.
        d2y_points: Second derivatives ``(n_points,)``.

    Returns:
        Fourier-transformed values ``(n_k,)``.
    """
    try:
        import scipy.special
    except ImportError as err:
        raise ImportError(
            "Computing the Fourier-domain kernel based on a spline requires scipy"
        ) from err

    dtype = x_points.dtype

    k = k_points.reshape(-1, 1).to(dtype)
    ri = x_points[torch.newaxis, :-1].to(dtype)
    yi = y_points[torch.newaxis, :-1].to(dtype)
    d2yi = d2y_points[torch.newaxis, :-1].to(dtype)
    dr = (x_points[torch.newaxis, 1:] - x_points[torch.newaxis, :-1]).to(dtype)
    dy = (y_points[torch.newaxis, 1:] - y_points[torch.newaxis, :-1]).to(dtype)
    dd2y = (d2y_points[torch.newaxis, 1:] - d2y_points[torch.newaxis, :-1]).to(dtype)
    coskx = torch.cos(k * ri)
    sinkx = torch.sin(k * ri)
    dcoskx = 2 * torch.sin(k * dr / 2) * torch.sin(k * (dr / 2 + ri))
    dsinkx = -2 * torch.sin(k * dr / 2) * torch.cos(k * (dr / 2 + ri))

    ft_interval = 24 * dcoskx * dd2y + k * (
        6 * dsinkx * (3 * d2yi * dr + dd2y * (4 * dr + ri))
        - 24 * dd2y * dr * sinkx
        + k
        * (
            6 * coskx * dr * (3 * d2yi * dr + dd2y * (2 * dr + ri))
            - 2 * dcoskx * (6 * dy + dr * ((6 * d2yi + 5 * dd2y) * dr + 3 * (d2yi + dd2y) * ri))
            + k
            * (
                dr
                * (12 * dy + 3 * d2yi * dr * (dr + 2 * ri) + dd2y * dr * (2 * dr + 3 * ri))
                * sinkx
                + dsinkx
                * (
                    -6 * dy * ri
                    - 3 * d2yi * dr**2 * (dr + ri)
                    - 2 * dd2y * dr**2 * (dr + ri)
                    - 6 * dr * (2 * dy + yi)
                )
                + k
                * (
                    6 * dcoskx * dr * (dr + ri) * (dy + yi)
                    + coskx * (6 * dr * ri * yi - 6 * dr * (dr + ri) * (dy + yi))
                )
            )
        )
    )

    tail_d2y = compute_second_derivatives(
        torch.tensor([0, 1 / x_points[-1], 1 / x_points[-2]]),
        torch.tensor([0, y_points[-1], y_points[-2]]),
    )

    r0 = x_points[-1]
    y0 = y_points[-1]
    d2y0 = tail_d2y[1]

    cosint = torch.from_numpy(scipy.special.sici((k * r0).detach().cpu().numpy())[1]).to(
        dtype=dr.dtype, device=dr.device
    )

    tail = (
        -2
        * torch.pi
        * (
            (d2y0 - 6 * r0**2 * y0) * torch.cos(k * r0)
            + d2y0 * k * r0 * (k * r0 * cosint - torch.sin(k * r0))
        )
    ) / (3.0 * r0)

    ft_sum = torch.pi * 2 / 3 * torch.sum(ft_interval / dr, dim=1).reshape(-1, 1)
    ft_limit = torch.sum(
        -(
            dr
            * torch.pi
            * (
                3 * d2yi * dr**2 * (3 * dr**2 + 10 * dr * ri + 10 * ri**2)
                + dd2y * dr**2 * (5 * dr**2 + 16 * dr * ri + 15 * ri**2)
                - 30
                * (
                    6 * ri**2 * (dy + 2 * yi)
                    + 4 * dr * ri * (2 * dy + 3 * yi)
                    + dr**2 * (3 * dy + 4 * yi)
                )
            )
        )
        / 90,
        dim=1,
    )

    safe_k = torch.where(k == 0, 1.0, k)
    return (
        torch.where(
            k == 0,
            ft_limit,
            ft_sum / safe_k**6 + tail / safe_k**2,
        )
        .reshape(k_points.shape)
        .to(k_points.dtype)
    )
