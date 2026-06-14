"""Tensor product convolution for equivariant message passing.

Two tensor-product wrappers are exposed here:

- :class:`ConvTP`: thin wrapper around ``cuet.ChannelWiseTensorProduct``
  (subscripts ``"uv,iu,jv,kuv+ijk"``).  Used by MACE-style encoders.
- :class:`EquivariantPolynomialTP`: general wrapper around an arbitrary
  ``cue.EquivariantPolynomial``.  Lets callers build custom descriptors via
  ``cue.SegmentedTensorProduct.from_subscripts(...)``.

Both wrappers support gather/scatter via ``indices_1``/``indices_2``/
``indices_out``/``size_out``, mirroring the signature of
``cuet.ChannelWiseTensorProduct``.

Model-specific descriptor builders (e.g. Allegro's per-channel ``"u,iu,ju,ku+ijk"``
kernel) live in the corresponding ``molzoo`` module, not here.
"""

from __future__ import annotations

from typing import Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConvTPSpec(BaseModel):
    r"""Specification for tensor product convolution layer.

    One-particle basis:
    $\phi_{ij} = \sum_{l_1,l_2,m_1,m_2} c_{l_3 m_3}^{l_1 m_1, l_2 m_2}
    R(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) h_j^{l_2 m_2}$

    Attributes:
        in_irreps: Input irreps.
        out_irreps: Output irreps.
        sh_irreps: Spherical harmonics irreps.
    """

    in_irreps: str
    out_irreps: str
    sh_irreps: str


class ConvTP(nn.Module):
    r"""Channelwise tensor product for equivariant message passing.

    Computes messages via tensor product:
    $$\phi_{ij} = \sum_{l_1,l_2,m_1,m_2} c_{l_3 m_3}^{l_1 m_1, l_2 m_2}
    R(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) h_j^{l_2 m_2}$$

    Attributes:
        config: ConvTPSpec configuration.
        cue_tp: ChannelWiseTensorProduct layer.
        weight_numel: Number of elements in TP weights.
    """

    def __init__(
        self,
        *,
        in_irreps: str,
        out_irreps: str,
        sh_irreps: str,
    ):
        """Initialize channelwise tensor product layer.

        Args:
            in_irreps: Input irreps for node features.
            out_irreps: Output irreps for messages.
            sh_irreps: Irreps for spherical harmonics.
        """
        super().__init__()

        self.config = ConvTPSpec(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            sh_irreps=sh_irreps,
        )

        irreps_in = cue.Irreps("O3", in_irreps)
        irreps_sh = cue.Irreps("O3", sh_irreps)
        irreps_out = cue.Irreps("O3", out_irreps)

        self.cue_tp = cuet.ChannelWiseTensorProduct(  # type: ignore
            irreps_in,
            irreps_sh,
            irreps_out,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_numel = self.cue_tp.weight_numel

    def forward(
        self,
        node_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_index: torch.Tensor,
        tp_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tensor product messages with integrated gather/scatter.

        Args:
            node_features: Node features.
            edge_angular: Spherical harmonics.
            edge_index: Edge indices ``(E, 2)``.
            tp_weights: TP weights.

        Returns:
            Computed messages (n_edges, out_irreps_dim).
        """
        indices_1 = edge_index[:, 0]
        indices_out = edge_index[:, 1]

        messages = self.cue_tp(
            node_features,
            edge_angular,
            tp_weights,
            indices_1=indices_1,
            indices_out=indices_out,
            size_out=node_features.shape[0],
        )

        return messages


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
