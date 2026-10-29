"""Allegro layers: the type embedding, weighted channels and the Contracter.

The encoder's own blocks (generic pieces — the ``ScalarMLPFunction`` MLP,
``SincBesselRBF``, cutoffs, spherical harmonics, edge geometry — are in
:mod:`molrep.embedding`):

- :class:`ProductTypeEmbedding` — ``[E_c(t_s) ‖ E_n(t_t)] ⊙ W b``.
- :class:`WeightedChannels` — per-(channel, irrep) weighting of the
  spherical-harmonic basis (``MakeWeightedChannels``), optionally summed
  into the centre atoms (:class:`molrep.equivariance.WeightedChannels`).
- :class:`Contracter` — the path-channel-coupled ``u,iu,ju,ku+ijk`` tensor
  product (:class:`molrep.equivariance.Contracter`); weights ``(u, P)`` with
  paths enumerated ``(out, in1, in2)``, so an official ``Contracter.weights``
  loads as is.

Tensor features are ``ir_mul``, flattened ``(rows, D · u)``.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
    ``allegro/nn/_edgeembed.py``, ``allegro/nn/_strided/_channels.py``,
    ``allegro/nn/_strided/_contract.py``, ``_cueq_contracter.py``
    (nequip-allegro 0.8.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import molrep.eq as meq
from molrep import equivariance
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
    output is ``ir_mul``, flattened ``(E, D · u)``. With ``index`` /
    ``num_rows`` the rows are summed into ``index`` (the Allegro environment
    sum), so the ``(E, D · u)`` message is never built.

    The contraction is :class:`molrep.equivariance.WeightedChannels`, which
    reads the weights irrep-major; the transpose is a view.

    Args:
        irrep_dims: Dimension ``2l + 1`` of each irrep of the basis, in order.
        channels: Output multiplicity ``u``.
        dtype: Floating dtype of the coefficient table.

    Reference:
        ``allegro/nn/_strided/_channels.py::MakeWeightedChannels``.
    """

    def __init__(self, *, irrep_dims: list[int], channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_irreps = len(irrep_dims)
        self.weight_numel = self.channels * self.num_irreps
        self.op = equivariance.WeightedChannels(list(irrep_dims), self.channels, dtype=dtype)

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
        weight = weights.view(rows, self.channels, self.num_irreps).transpose(1, 2)
        if index is None:
            return self.op(basis, weight)
        return self.op(basis, weight, index, num_rows)


class Contracter(nn.Module):
    """Path-channel-coupled ``u,iu,ju,ku+ijk`` tensor product.

    The contraction is :class:`molrep.equivariance.Contracter`; this module
    owns the parameter in Allegro's layout, ``weights`` ``(channels, paths)``
    with paths enumerated ``(out, in1, in2)``, so an official
    ``Contracter.weights`` loads as is.

    Args:
        irreps_in1: Irreps of the pair operand, one copy each (``"0e+1o+…"``).
        irreps_in2: Irreps of the gathered atom operand, one copy each.
        irreps_out: Irreps of the output, one copy each (order preserved).
        channels: Shared multiplicity ``u`` of every operand.
        dtype: Floating dtype of the parameter and the coefficient table.

    Raises:
        ValueError: If no path connects the inputs to the outputs.
    """

    def __init__(
        self,
        *,
        irreps_in1: meq.Irreps,
        irreps_in2: meq.Irreps,
        irreps_out: meq.Irreps,
        channels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        u = int(channels)
        self.op = equivariance.Contracter(irreps_in1, irreps_in2, irreps_out, u, dtype=dtype)
        self.num_paths = self.op.weight_segments
        self.weights = nn.Parameter(torch.empty(u, self.num_paths, dtype=dtype))
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
        return self.op(x1, x2, self.weights.transpose(0, 1), index)
