"""Custom math functions: gamma, exponential integral, gammainc helpers."""

import torch
from torch.special import gammaln


def gamma(x: torch.Tensor) -> torch.Tensor:
    """Complete Gamma function :math:`\\Gamma(x)`.

    PyTorch does not expose the complete Gamma function natively.
    This wraps :func:`torch.special.gammaln`.

    Args:
        x: Input tensor of any shape.

    Returns:
        Gamma function values matching ``x`` shape.
    """
    return torch.exp(gammaln(x))


class _CustomExp1(torch.autograd.Function):
    """Custom autograd function for the exponential integral E1."""

    @staticmethod
    def forward(ctx, x):
        SCIPY_EULER = 0.577215664901532860606512090082402431
        inf = torch.inf

        result = torch.full_like(x, inf)
        mask = x > 0

        x_small = x[mask & (x <= 1)]
        if x_small.numel() > 0:
            e1 = torch.ones_like(x_small)
            r = torch.ones_like(x_small)
            for k in range(1, 26):
                r = -r * k * x_small / (k + 1.0) ** 2
                e1 += r
                if torch.all(torch.abs(r) <= torch.abs(e1) * 1e-15):
                    break
            result[mask & (x <= 1)] = -SCIPY_EULER - torch.log(x_small) + x_small * e1

        x_large = x[mask & (x > 1)]
        if x_large.numel() > 0:
            m = 20 + (80.0 / x_large).to(torch.int32)
            t0 = torch.zeros_like(x_large)
            for k in range(m.max(), 0, -1):
                t0 = k / (1.0 + k / (x_large + t0))
            t = 1.0 / (x_large + t0)
            result[mask & (x > 1)] = torch.exp(-x_large) * t

        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return -grad_output * torch.exp(-x) / x


def exp1(x: torch.Tensor) -> torch.Tensor:
    """Exponential integral :math:`E_1(x) = \\int_x^\\infty e^{-t}/t \\, dt`.

    Args:
        x: Input tensor (must be > 0).

    Returns:
        Exponential integral values matching ``x`` shape.
    """
    return _CustomExp1.apply(x)


def gammaincc_over_powerlaw(exponent: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete gamma function complement for integer exponents.

    Args:
        exponent: Integer exponent of the power law.
        z: Evaluation points.

    Returns:
        Function values matching ``z`` shape.

    Raises:
        ValueError: If ``exponent`` is not in [1, 6].
    """
    if exponent == 1:
        return torch.exp(-z) / z
    if exponent == 2:
        return torch.sqrt(torch.pi / z) * torch.erfc(torch.sqrt(z))
    if exponent == 3:
        return exp1(z)
    if exponent == 4:
        return 2 * (torch.exp(-z) - torch.sqrt(torch.pi * z) * torch.erfc(torch.sqrt(z)))
    if exponent == 5:
        return torch.exp(-z) - z * exp1(z)
    if exponent == 6:
        return (
            (2 - 4 * z) * torch.exp(-z)
            + 4 * torch.sqrt(torch.pi * z**3) * torch.erfc(torch.sqrt(z))
        ) / 3
    raise ValueError(f"Unsupported exponent: {exponent}")
