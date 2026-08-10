"""Generic tensor products and irreps helpers for equivariant message passing.

Exposed here:

- :class:`EquivariantPolynomialTP`: general wrapper around an arbitrary
  ``cue.EquivariantPolynomial``.  Lets callers build custom descriptors via
  ``cue.SegmentedTensorProduct.from_subscripts(...)``.  It supports
  gather/scatter via ``indices_1``/``indices_2``/``indices_out``/``size_out``,
  mirroring the signature of ``cuet.ChannelWiseTensorProduct``.
- ``irreps_from_l_max`` / ``sh_irreps_from_l_max``: irreps-string builders
  shared by MACE, Allegro and the MACE readouts.

The MACE-specific :class:`~molrep.interaction.mace.conv.ConvTP` wrapper moved to
:mod:`molrep.interaction.mace.conv`; it is re-exported at the bottom of this
module so ``from molrep.interaction.product import ConvTP`` keeps working.

Model-specific descriptor builders (e.g. Allegro's per-channel ``"u,iu,ju,ku+ijk"``
kernel) live in the corresponding ``molzoo`` module, not here.
"""

from __future__ import annotations

from typing import Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn


def irreps_from_l_max(l_max: int, hidden_dim: int) -> str:
    """Generate irreps string with uniform multiplicities for optimal cuEquivariance performance.

    Args:
        l_max: Maximum angular momentum.
        hidden_dim: Uniform multiplicity across all l values.

    Returns:
        Irreps string (e.g., "32x0e + 32x1o + 32x2e").

    Note:
        Uniform multiplicities enable cuEquivariance's fast uniform_1d method (~5-10x speedup).
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"{hidden_dim}x{l}{parity}")
    return " + ".join(irreps_list)


def sh_irreps_from_l_max(l_max: int) -> str:
    """Generate spherical harmonics irreps string from l_max.

    Args:
        l_max: Maximum angular momentum.

    Returns:
        Irreps string (e.g., "1x0e + 1x1o + 1x2e").
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"1x{l}{parity}")
    return " + ".join(irreps_list)


# ===========================================================================
# Generalized TP module
# ===========================================================================


class EquivariantPolynomialTP(nn.Module):
    """Execute an arbitrary ``cue.EquivariantPolynomial`` as an nn.Module.

    This generalises :class:`ConvTP` — instead of being hard-coded to the
    channelwise descriptor, it wraps *any* polynomial built via
    ``cue.descriptors.xxx`` or a hand-rolled
    ``cue.SegmentedTensorProduct.from_subscripts(...)``.

    Typical callsites:

    >>> poly = cue.descriptors.channelwise_tensor_product(irreps_in, irreps_sh)
    >>> tp = EquivariantPolynomialTP(poly, internal_weights=False)
    >>> out = tp(lhs, rhs, weight=w_per_edge)

    Model-specific descriptors (e.g. ``"u,iu,ju,ku+ijk"`` for Allegro) are
    built in the corresponding ``molzoo`` module and passed in here.

    Args:
        polynomial: The equivariant polynomial to execute.  Its first input
            (by convention) is the weights; the remaining inputs are the
            operands in order.
        shared_weights: If True, weights are shared across the batch — store
            a single ``(1, weight_numel)`` parameter.  If False, weights must
            be supplied at each forward call with shape
            ``(batch, weight_numel)``.
        internal_weights: If True, allocate an internal ``nn.Parameter`` for
            the weights.  Must be ``True`` only when ``shared_weights=True``.
            Defaults to ``shared_weights``.
        method: Dispatch method for ``cuet.SegmentedPolynomial``.  If None,
            auto-select based on segment shapes.
        math_dtype: Forwarded to ``cuet.SegmentedPolynomial``.
        dtype: Data type for the internal weight parameter.
    """

    def __init__(
        self,
        polynomial: cue.EquivariantPolynomial,
        *,
        shared_weights: bool = False,
        internal_weights: Optional[bool] = None,
        method: Optional[str] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.polynomial = polynomial
        self.num_inputs = polynomial.polynomial.num_inputs
        self.num_outputs = polynomial.polynomial.num_outputs

        # Weights are always the first input operand.
        self.weight_numel = int(polynomial.inputs[0].irreps.dim)

        self.shared_weights = shared_weights
        if internal_weights is None:
            internal_weights = shared_weights
        self.internal_weights = internal_weights

        if internal_weights and not shared_weights:
            raise ValueError("internal_weights=True requires shared_weights=True")

        if internal_weights:
            self.weight = nn.Parameter(
                torch.randn(1, self.weight_numel, device=device, dtype=dtype)
            )
        else:
            self.weight = None

        self.f = cuet.SegmentedPolynomial(
            polynomial.polynomial, method=method, math_dtype=math_dtype
        ).to(device)

    @property
    def irreps_out(self) -> cue.Irreps:
        return self.polynomial.outputs[0].irreps

    def forward(
        self,
        *operands: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        indices_1: Optional[torch.Tensor] = None,
        indices_2: Optional[torch.Tensor] = None,
        indices_out: Optional[torch.Tensor] = None,
        size_out: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the polynomial.

        Args:
            *operands: The ``num_inputs - 1`` non-weight operands.  Shape
                ``(batch, irreps_input.dim)`` each.
            weight: External weight tensor; required iff
                ``internal_weights=False``.
            indices_1: Optional gather indices for operand 1.
            indices_2: Optional gather indices for operand 2.
            indices_out: Optional scatter indices for the output.
            size_out: Required when ``indices_out`` is set.

        Returns:
            Output tensor ``(batch, irreps_out.dim)``.
        """
        weight = self.weight if self.internal_weights else weight
        assert weight is not None, "weight must be provided when internal_weights=False"

        indices_in: dict[int, torch.Tensor] = {}
        if indices_1 is not None:
            indices_in[1] = indices_1
        if indices_2 is not None:
            indices_in[2] = indices_2

        output_indices = {0: indices_out} if indices_out is not None else None
        sizes_out = (
            {0: torch.empty(size_out, 1, device=operands[0].device)}
            if indices_out is not None
            else {}
        )

        return self.f(
            [weight, *operands],
            input_indices=indices_in,
            output_shapes=sizes_out,
            output_indices=output_indices,
        )[0]


# Back-compat re-export of the MACE convolution, moved to
# ``molrep.interaction.mace.conv`` by mace-subpackage-restructure-01. Kept at the
# module tail on purpose: ``mace.block`` imports the irreps helpers above from
# here, so this module must be fully populated before the mace package is pulled
# in. Removed in 06-wire.
from molrep.interaction.mace.conv import ConvTP, ConvTPSpec  # noqa: E402,F401
