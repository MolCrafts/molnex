"""Allegro layers: the type embedding, weighted channels and the Contracter.

Model-specific, so they live with the model (only generic pieces — the
``ScalarMLPFunction`` MLP, ``SincBesselRBF``, cutoffs, spherical harmonics,
edge geometry — are in :mod:`molrep`):

- :class:`ProductTypeEmbedding` — ``[E_c(t_s) ‖ E_n(t_t)] ⊙ W b``.
- :class:`WeightedChannels` — per-(channel, irrep) weighting of the
  spherical-harmonic basis (``MakeWeightedChannels``), optionally summed
  into the centre atoms by cuEquivariance's indexed output.
- :class:`Contracter` — the path-channel-coupled ``u,iu,ju,ku+ijk``
  cuEquivariance tensor product; weights ``(u, P)`` with paths enumerated
  ``(out, in1, in2)``, so an official ``Contracter.weights`` loads as is.

Tensor features are cuEquivariance ``ir_mul``, flattened ``(rows, D · u)``.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
    ``allegro/nn/_edgeembed.py``, ``allegro/nn/_strided/_channels.py``,
    ``allegro/nn/_strided/_contract.py``, ``_cueq_contracter.py``
    (nequip-allegro 0.8.3).
"""

from __future__ import annotations

import itertools
import math

import cuequivariance as cue
import torch
import torch.nn as nn

from molrep import cueq
from molrep.embedding.mlp import ScalarMLPFunction


class ProductTypeEmbedding(nn.Module):
    """``[E_c(t_s) ‖ E_n(t_t)] ⊙ W b`` — type embedding gated by a radial basis.

    The source (centre) and target (neighbour) types have separate tables of
    width ``dim / 2``; their concatenation is multiplied elementwise by a
    bias-free linear map of the radial basis (nequip init).

    Args:
        num_types: Number of atom types.
        num_basis: Radial basis size.
        dim: Output width (even).

    Raises:
        ValueError: If ``dim`` is odd.
    """

    def __init__(self, *, num_types: int, num_basis: int, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim must be even (split centre/neighbour), got {dim}")
        self.center_embed = nn.Embedding(num_types, dim // 2, dtype=dtype)
        self.neighbor_embed = nn.Embedding(num_types, dim // 2, dtype=dtype)
        self.basis_linear = ScalarMLPFunction(num_basis, dim, nonlinearity=None, dtype=dtype)

    def forward(
        self, types: torch.Tensor, edge_index: torch.Tensor, basis: torch.Tensor
    ) -> torch.Tensor:
        """Args: ``types`` ``(N,)``; ``edge_index`` ``(E, 2)``; ``basis`` ``(E, B)``.

        Returns:
            ``(E, dim)``.
        """
        embed = torch.cat(
            [
                self.center_embed(types[edge_index[:, 0]]),
                self.neighbor_embed(types[edge_index[:, 1]]),
            ],
            dim=-1,
        )
        return embed * self.basis_linear(basis)


class WeightedChannels(nn.Module):
    """Per-(channel, irrep) weighting of a one-channel irreps basis.

    ``out[e, d, c] = basis[e, d] · w[e, c, irrep(d)]``: every component ``d``
    of irrep ``r`` in ``channel`` ``c`` takes the weight ``w[c, r]``. Weights
    arrive channel-major, ``(E, u · R)`` read as ``(E, u, R)`` — the layout of
    Allegro's ``MakeWeightedChannels(weight_individual_irreps=True)``. The
    output is cuEquivariance ``ir_mul``, flattened ``(E, D · u)``.

    One cuEquivariance segmented polynomial, subscripts ``",u,u"``: operand 0
    is the basis as ``D`` scalar segments, operand 1 the weights as ``R``
    segments of ``(u,)`` (irrep-major, so the channel-major input is
    transposed per call), the output ``D`` segments of ``(u,)``; one path per
    component ``d`` joins ``(d, irrep(d), d)`` with coefficient 1. With
    ``index`` / ``num_rows`` the rows are summed into ``index`` by cue's
    indexed output (the Allegro environment sum), so the ``(E, D · u)``
    message is never built.

    Args:
        irrep_dims: Dimension ``2l + 1`` of each irrep of the basis, in order.
        channels: Output multiplicity ``u``.

    Reference:
        ``allegro/nn/_strided/_channels.py::MakeWeightedChannels``.
    """

    def __init__(self, *, irrep_dims: list[int], channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_irreps = len(irrep_dims)
        self.weight_numel = self.channels * self.num_irreps
        d = cue.SegmentedTensorProduct.from_subscripts(",u,u")
        for _ in range(self.num_irreps):
            d.add_segment(1, (self.channels,))
        for r, dim in enumerate(irrep_dims):
            for _ in range(dim):
                component = d.add_segment(0, ())
                d.add_segment(2, (self.channels,))
                d.add_path(component, r, component, c=1.0)
        self.op = cueq.SegmentedPolynomial(
            cue.SegmentedPolynomial.eval_last_operand(d), math_dtype=dtype
        )

    def forward(
        self,
        basis: torch.Tensor,
        weights: torch.Tensor,
        *,
        index: torch.Tensor | None = None,
        num_rows: int | None = None,
    ) -> torch.Tensor:
        """Weight the basis; with ``index`` sum the rows into ``num_rows`` buckets.

        Args:
            basis: ``(E, D)``.
            weights: ``(E, u · R)`` channel-major.
            index: Row of the output each pair adds to ``(E,)``; ``None``
                keeps one row per pair.
            num_rows: Output rows ``N`` when ``index`` is given.

        Returns:
            ``(E, D · u)`` ``ir_mul``, or ``(N, D · u)`` summed by ``index``.

        Raises:
            ValueError: If only one of ``index`` / ``num_rows`` is given.
        """
        if (index is None) != (num_rows is None):
            raise ValueError("index and num_rows come together")
        rows = weights.shape[0]
        w = weights.view(rows, self.channels, self.num_irreps).transpose(1, 2).reshape(rows, -1)
        if index is None:
            return self.op([basis, w])[0]
        return self.op(
            [basis, w],
            output_shapes={0: basis.new_empty((num_rows, 0))},
            output_indices={0: index},
        )[0]


class Contracter(nn.Module):
    """Path-channel-coupled ``u,iu,ju,ku+ijk`` tensor product.

    Args:
        irreps_in1: Irreps of the pair operand, one copy each (``"0e+1o+…"``).
        irreps_in2: Irreps of the gathered atom operand, one copy each.
        irreps_out: Irreps of the output, one copy each (order preserved).
        channels: Shared multiplicity ``u`` of every operand.
        method: cuEquivariance kernel method (``"uniform_1d"`` or
            ``"fused_tp"``); see :mod:`molrep.cueq`.

    Raises:
        ValueError: If no path connects the inputs to the outputs.
    """

    def __init__(
        self,
        *,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        channels: int,
        method: str = "uniform_1d",
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        u = int(channels)
        d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
        for _, ir in irreps_in1:
            d.add_segment(1, (ir.dim, u))
        for _, ir in irreps_in2:
            d.add_segment(2, (ir.dim, u))
        for _, ir in irreps_out:
            d.add_segment(3, (ir.dim, u))
        num_paths = 0
        for (i3, (_, ir3)), (i1, (_, ir1)), (i2, (_, ir2)) in itertools.product(
            enumerate(irreps_out), enumerate(irreps_in1), enumerate(irreps_in2)
        ):
            if ir3 in ir1 * ir2:
                for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                    d.add_path(None, i1, i2, i3, c=cg)
                    num_paths += 1
        if num_paths == 0:
            raise ValueError(f"no tensor-product path {irreps_in1} x {irreps_in2} -> {irreps_out}")
        self.num_paths = num_paths
        poly = cue.SegmentedPolynomial.eval_last_operand(d).flatten_coefficient_modes()
        self.tp = cueq.SegmentedPolynomial(poly, method=method, math_dtype=dtype)
        self.weights = nn.Parameter(torch.empty(u, num_paths, dtype=dtype))
        nn.init.uniform_(self.weights, -math.sqrt(3.0), math.sqrt(3.0))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Contract.

        Args:
            x1: Pair operand ``(E, D1 · u)`` ``ir_mul``.
            x2: Atom operand ``(N, D2 · u)`` ``ir_mul``.
            index: Row of ``x2`` for each pair ``(E,)`` (the source atom).

        Returns:
            ``(E, D_out · u)`` ``ir_mul``.
        """
        weights = self.weights.transpose(0, 1).reshape(1, -1)
        return self.tp([weights, x1, x2], input_indices={2: index})[0]
