from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

Key = str | tuple[str, ...]


class CosineCutoffSpec(BaseModel):
    """Specification for cosine cutoff function.

    Defines parameters for a smooth cosine cutoff function that transitions
    from 1 to 0 over the range [0, r_cut].

    Attributes:
        r_cut: Cutoff radius. Values are 0 for r >= r_cut. Must be positive.
    """

    r_cut: float = Field(..., gt=0)


class CosineCutoff(nn.Module):
    """Cosine cutoff function module.

    Applies a smooth cosine cutoff to distance values:
        c(r) = 0.5 * (cos(pi * r / r_cut) + 1)  for r < r_cut
        c(r) = 0                                for r >= r_cut

    This provides a smooth transition to zero at the cutoff radius, which is
    important for avoiding discontinuities in neural network potentials.

    Attributes:
        config: CosineCutoffSpec configuration.
        r_cut: Buffer storing cutoff radius.
        _pi: Buffer storing pi constant.
    """

    def __init__(self, *, r_cut: float):
        """Initialize cosine cutoff module.

        Args:
            r_cut: Cutoff radius.
        """
        super().__init__()

        self.config = CosineCutoffSpec(
            r_cut=r_cut,
        )

        # Register buffers with type annotations
        r_cut_tensor = torch.tensor(float(self.config.r_cut))
        self.register_buffer("r_cut", r_cut_tensor, persistent=False)
        self.r_cut: torch.Tensor

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply cosine cutoff to distances.

        Args:
            r: Input distances.

        Returns:
            Cutoff values. Values range from 1.0 (at r=0) to 0.0 (at r>=r_cut).
        """
        r = r.float()

        # Create mask for distances within cutoff
        mask = r < self.r_cut

        # Compute cosine cutoff
        x = r / self.r_cut
        c = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        c = torch.where(mask, c, torch.zeros_like(c))

        return c


class PolynomialCutoffSpec(BaseModel):
    """Specification for polynomial cutoff function.

    Defines parameters for a smooth polynomial cutoff function with
    continuous derivatives up to a specified order.

    Attributes:
        r_cut: Cutoff radius. Values are 0 for r >= r_cut. Must be positive.
        exponent: Polynomial exponent controlling smoothness. Higher values
            give smoother cutoffs. Defaults to 6. Must be positive.
    """

    r_cut: float = Field(..., gt=0)
    exponent: int = Field(6, gt=0)


class PolynomialCutoff(nn.Module):
    """Polynomial cutoff envelope used in NequIP / Allegro.

    Applies the Klicpera-et-al. (2020) polynomial envelope to distance values:
        u(r) = 1 - ((p+1)(p+2)/2) * x^p
                 + p * (p+2)       * x^(p+1)
                 - (p(p+1)/2)      * x^(p+2)           for x = r/r_cut < 1
        u(r) = 0                                        for r >= r_cut

    For the default p=6 this gives ``1 - 28 x^6 + 48 x^7 - 21 x^8``.

    The envelope vanishes smoothly with continuous derivatives up to order
    ``(p-1)`` at ``r = r_cut``, which is necessary for stable autograd forces.

    Reference:
        Klicpera, Groß, Günnemann, "Directional Message Passing for Molecular
        Graphs", ICLR 2020.  Used as the standard cutoff in NequIP/Allegro.

    Attributes:
        config: PolynomialCutoffSpec configuration.
        r_cut: Buffer storing cutoff radius.
        exponent: Polynomial exponent ``p``.
    """

    def __init__(self, *, r_cut: float, exponent: int = 6):
        """Initialize polynomial cutoff module.

        Args:
            r_cut: Cutoff radius.
            exponent: Polynomial exponent controlling smoothness. Defaults to 6.
        """
        super().__init__()

        self.config = PolynomialCutoffSpec(
            r_cut=r_cut,
            exponent=exponent,
        )

        r_cut_tensor = torch.tensor(float(self.config.r_cut))
        self.register_buffer("r_cut", r_cut_tensor, persistent=False)
        self.r_cut: torch.Tensor

        self.exponent = int(self.config.exponent)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply polynomial cutoff to distances.

        Args:
            r: Input distances.

        Returns:
            Cutoff values. Values range from 1.0 (at r=0) to 0.0 (at r>=r_cut).
        """
        r = r.float()
        x = r / self.r_cut
        mask = x < 1.0

        p = float(self.exponent)
        c_p = (p + 1.0) * (p + 2.0) / 2.0
        c_p1 = p * (p + 2.0)
        c_p2 = p * (p + 1.0) / 2.0

        x_p = torch.pow(x, p)
        x_p1 = x_p * x
        x_p2 = x_p1 * x

        c = 1.0 - c_p * x_p + c_p1 * x_p1 - c_p2 * x_p2
        return torch.where(mask, c, torch.zeros_like(c))
