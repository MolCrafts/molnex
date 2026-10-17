"""PiNet2 layers: the PiNN building blocks the PiNet2 body is made of.

Model-specific, so they live with the model (only generic pieces — the
``FFLayer`` MLP, cutoffs, radial bases, edge geometry — are in :mod:`molrep`):

- :func:`compute_d5`, :class:`PIXLayer`, :class:`ScaleLayer`,
  :class:`DotLayer` — Cartesian rank-3 / rank-5 channel operations
  (``(N, 3, C)`` vectors, ``(N, 5, C)`` traceless rank-2 tensors).
- :class:`PILayer`, :class:`IPLayer` — atom → pair and pair → atom (source).
- :class:`ResUpdate` — residual state update.
- :class:`InvarLayer`, :class:`EquivarLayer`, :class:`GCBlock`
  (:class:`GCOutput`) — one graph-convolution block.
- :class:`OutLayer` — per-block energy readout with residual accumulation.

Every Cartesian operation is linear in the components or a full contraction
over them, so rotation equivariance / invariance hold by construction.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems" J. Chem. Theory Comput. 21, 1382 (2025).
    https://doi.org/10.1021/acs.jctc.4c01570
    Shao et al. "PiNN: A Python Library for Building Atomic Neural Networks
    of Molecules and Materials" J. Chem. Inf. Model. 60, 1184 (2020).
    https://doi.org/10.1021/acs.jcim.9b00994
    ``pinn/networks/pinet.py``, ``pinn/networks/pinet2.py`` (Teoroo-CMC/PiNN).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum_compile_safe
from molrep.embedding.mlp import FFLayer


def compute_d5(d3: torch.Tensor) -> torch.Tensor:
    """Five-component symmetric-traceless rank-2 basis of unit vectors.

    Args:
        d3: Unit edge directions ``(E, 3)``.

    Returns:
        ``(E, 5)``: ``[⅔x²−⅓y²−⅓z², ⅔y²−⅓x²−⅓z², xy, xz, yz]``, PiNN's order.
    """
    x, y, z = d3[:, 0], d3[:, 1], d3[:, 2]
    x2, y2, z2 = x.square(), y.square(), z.square()
    return torch.stack(
        [
            (2.0 / 3.0) * x2 - (1.0 / 3.0) * y2 - (1.0 / 3.0) * z2,
            (2.0 / 3.0) * y2 - (1.0 / 3.0) * x2 - (1.0 / 3.0) * z2,
            x * y,
            x * z,
            y * z,
        ],
        dim=1,
    )


class PIXLayer(nn.Module):
    """Equivariant property → pair: ``W_i P_i + W_j P_j``, or ``P_j`` unweighted.

    Args:
        channels: Channel width ``C`` of the property.
        weighted: Mix both endpoints through bias-free channel linears; when
            ``False`` the pair takes the target property as is.
        dtype: Floating dtype of the linears (the owning module's).
    """

    def __init__(
        self, *, channels: int, weighted: bool = False, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.weighted = bool(weighted)
        self.channels = int(channels)
        if self.weighted:
            self.wi = nn.Linear(self.channels, self.channels, bias=False, dtype=dtype)
            self.wj = nn.Linear(self.channels, self.channels, bias=False, dtype=dtype)

    def forward(self, edge_index: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        """Pair channels.

        Args:
            edge_index: Source/target pairs ``(E, 2)``.
            px: Cartesian property ``(N, K, C)``.

        Returns:
            ``(E, K, C)``.
        """
        if self.weighted:
            return self.wi(px[edge_index[:, 0]]) + self.wj(px[edge_index[:, 1]])
        return px[edge_index[:, 1]]


class ScaleLayer(nn.Module):
    """Scale a Cartesian property by per-channel invariants: ``X_{·kc} s_{·c}``."""

    def forward(self, px: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """Args: ``px`` ``(R, K, C)``; ``p1`` ``(R, C)``. Returns ``(R, K, C)``."""
        return px * p1.unsqueeze(-2)


class DotLayer(nn.Module):
    """Contract the component axis: ``Σ_k (W_i X)_{kc} (W_j X)_{kc}``.

    Args:
        channels: Channel width ``C``.
        weighted: Mix each factor through a bias-free channel linear first.
        dtype: Floating dtype of the linears (the owning module's).
    """

    def __init__(
        self, *, channels: int, weighted: bool = False, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.weighted = bool(weighted)
        self.channels = int(channels)
        if self.weighted:
            self.wi = nn.Linear(self.channels, self.channels, bias=False, dtype=dtype)
            self.wj = nn.Linear(self.channels, self.channels, bias=False, dtype=dtype)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """Args: ``px`` ``(N, K, C)``. Returns the invariant ``(N, C)``."""
        if self.weighted:
            return (self.wi(px) * self.wj(px)).sum(1)
        return (px * px).sum(1)


class PILayer(nn.Module):
    """Property → interaction: ``I_ij = Σ_b W(P_i ‖ P_j)_{·b} · e_ijb``.

    ``FFLayer`` maps the concatenated endpoint properties to
    ``n_nodes[-1] · n_basis`` weights, reshaped to ``(E, out, n_basis)`` and
    contracted with the basis.

    Args:
        n_nodes: FF widths; the last is the interaction width.
        in_dim: Width of the concatenated endpoint properties (``2 · P``).
        n_basis: Radial basis size.
        activation: FF activation (:func:`molix.nn.activation_from_name`).
        use_bias: Bias on every FF linear.
        dtype: Floating dtype of the FF stack (the owning module's).

    Raises:
        ValueError: If ``n_nodes`` is empty.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        n_basis: int,
        activation: str | type[nn.Module] | None = None,
        use_bias: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not n_nodes:
            raise ValueError("PILayer requires at least one output width.")
        self.n_basis = int(n_basis)
        self.out_dim = int(n_nodes[-1])
        widths = [int(v) for v in n_nodes]
        widths[-1] *= self.n_basis
        self.ff_layer = FFLayer(
            widths, in_dim=int(in_dim), activation=activation, use_bias=use_bias, dtype=dtype
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        prop: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """Pair interactions.

        Args:
            edge_index: Source/target pairs ``(E, 2)``.
            prop: Atom properties ``(N, P)``.
            basis: Radial basis ``(E, n_basis)``.

        Returns:
            Interactions ``(E, n_nodes[-1])``.
        """
        inter = torch.cat([prop[edge_index[:, 0]], prop[edge_index[:, 1]]], dim=-1)
        weights = self.ff_layer(inter).reshape(-1, self.out_dim, self.n_basis)
        # Broadcast multiply + sum (not einsum): one fused reduction under
        # torch.compile instead of a memory-bound gemv.
        return (weights * basis.unsqueeze(1)).sum(-1)


class IPLayer(nn.Module):
    """Interaction → property: ``P_i = Σ_j I_ij`` onto the source atom."""

    def forward(
        self, edge_index: torch.Tensor, inter: torch.Tensor, num_atoms: int
    ) -> torch.Tensor:
        """Sum pair rows onto their source atom.

        Args:
            edge_index: Source/target pairs ``(E, 2)``.
            inter: Pair interactions ``(E, ...)``.
            num_atoms: Number of atoms ``N`` (a shape, not a tensor value).

        Returns:
            Atom properties ``(N, ...)``.
        """
        return scatter_sum_compile_safe(inter, edge_index[:, 0], num_atoms)


class ResUpdate(nn.Module):
    """``new + old`` (equal widths) or ``new + W old`` (bias-free projection).

    Args:
        in_dim: Channel width of the old state.
        out_dim: Channel width of the new state.
        dtype: Floating dtype of the projection (the owning module's).
    """

    def __init__(self, *, in_dim: int, out_dim: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.in_dim == self.out_dim:
            self.transform: nn.Module = nn.Identity()
        else:
            self.transform = nn.Linear(self.in_dim, self.out_dim, bias=False, dtype=dtype)

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Args: ``old`` ``(..., in_dim)``; ``new`` ``(..., out_dim)``. Returns ``new``'s shape."""
        return self.transform(old) + new


class InvarLayer(nn.Module):
    """Scalar invariant update ``PI → II → IP → PP``.

    ``II`` and ``PP`` are bias-free so a pair entering the cutoff, where the
    basis vanishes, contributes nothing.

    Args:
        p1_in_dim: Width of the incoming ``P1``.
        pp_nodes: PP widths.
        pi_nodes: PI widths.
        ii_nodes: II widths.
        n_basis: Radial basis size.
        activation: Activation of every FF stack.
        dtype: Floating dtype of every FF stack (the owning module's).

    Raises:
        ValueError: If ``pi_nodes`` is empty.
    """

    def __init__(
        self,
        *,
        p1_in_dim: int,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str | type[nn.Module] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not pi_nodes:
            raise ValueError("pi_nodes must not be empty.")
        self.pi_layer = PILayer(
            pi_nodes, in_dim=2 * int(p1_in_dim), n_basis=n_basis, activation=activation, dtype=dtype
        )
        self.ii_layer = FFLayer(
            ii_nodes, in_dim=int(pi_nodes[-1]), activation=activation, use_bias=False, dtype=dtype
        )
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(
            pp_nodes, in_dim=int(ii_nodes[-1]), activation=activation, use_bias=False, dtype=dtype
        )

    def forward(
        self, edge_index: torch.Tensor, p1: torch.Tensor, basis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: ``edge_index`` ``(E, 2)``; ``p1`` ``(N, P)``; ``basis`` ``(E, B)``.

        Returns:
            ``(p1_new (N, pp_nodes[-1]), i1 (E, ii_nodes[-1]))``.
        """
        i1 = self.ii_layer(self.pi_layer(edge_index, p1, basis))
        p1_new = self.pp_layer(self.ip_layer(edge_index, i1, p1.shape[0]))
        return p1_new, i1


class EquivarLayer(nn.Module):
    """Equivariant update for one Cartesian track (no activation, no bias).

    ``I_x = (PIX(P_x) + d) · i`` per pair, then ``IP`` and a bias-free ``PP``
    channel linear, then the invariant ``DotLayer`` of the result.

    Args:
        in_channels: Channel width of the incoming property (1 on the first
            block).
        out_channels: Channel width after the block (the scalar gate width).
        weighted: Weighted PIX / Dot variant.
        dtype: Floating dtype of every linear (the owning module's).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        weighted: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.pi_layer = PIXLayer(channels=int(in_channels), weighted=weighted, dtype=dtype)
        self.ip_layer = IPLayer()
        # After ``(PIX + d) · i`` the channel axis is the gate width, so PP is
        # out_channels → out_channels.
        self.pp_layer = FFLayer(
            [out_channels], in_dim=int(out_channels), activation=None, use_bias=False, dtype=dtype
        )
        self.dot_layer = DotLayer(channels=int(out_channels), weighted=weighted, dtype=dtype)

    def forward(
        self,
        edge_index: torch.Tensor,
        px: torch.Tensor,
        i1: torch.Tensor,
        diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``px`` ``(N, K, C_in)``; ``i1`` gate ``(E, C)``; ``diff`` ``(E, K)``.

        Returns:
            ``(px_new (N, K, C), ix (E, K, C), dotted (N, C))``.
        """
        # One broadcast multiply instead of two ScaleLayer + add (bit-identical).
        ix = (self.pi_layer(edge_index, px) + diff.unsqueeze(-1)) * i1.unsqueeze(-2)
        px_new = self.pp_layer(self.ip_layer(edge_index, ix, px.shape[0]))
        return px_new, ix, self.dot_layer(px_new)


class GCOutput(NamedTuple):
    """What one :class:`GCBlock` produces; tracks above the rank are ``None``.

    Attributes:
        p1: Scalar block output ``(N, C)``.
        p3: Vector block output ``(N, 3, C)``.
        p5: Rank-2 block output ``(N, 5, C)``.
        i1: Scalar pair interaction ``(E, C · n_props)``.
        i3: Vector pair interaction ``(E, 3, C)``.
        i5: Rank-2 pair interaction ``(E, 5, C)``.
    """

    p1: torch.Tensor
    p3: torch.Tensor | None
    p5: torch.Tensor | None
    i1: torch.Tensor
    i3: torch.Tensor | None
    i5: torch.Tensor | None


class GCBlock(nn.Module):
    """One PiNet2 graph-convolution block.

    The invariant layer emits ``n_props`` scalar chunks; chunk ``k`` gates
    equivariant track ``k``. The ``PP`` mixer reads ``[p1, dot(p3), dot(p5)]``
    and its chunks scale the next ``p3`` / ``p5``.

    Args:
        rank: ``1`` (scalar only), ``3`` (+ vectors) or ``5`` (+ rank-2).
        weighted: Weighted PIX / Dot variant.
        pp_nodes: PP widths; ``pp_nodes[-1]`` is the feature width ``C``.
        pi_nodes: PI widths.
        ii_nodes: II widths; ``ii_nodes[-1]`` must equal ``pp_nodes[-1]``
            (validated by the model spec).
        n_basis: Radial basis size.
        p1_in_dim: Width of the incoming ``P1``.
        p3_in_dim: Channel width of the incoming ``P3``.
        p5_in_dim: Channel width of the incoming ``P5``.
        activation: Activation of the invariant FF stacks.
        dtype: Floating dtype of every layer in the block (the owning module's).

    Raises:
        ValueError: If ``rank`` is not 1, 3 or 5.
    """

    def __init__(
        self,
        *,
        rank: int,
        weighted: bool,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        p1_in_dim: int,
        p3_in_dim: int = 1,
        p5_in_dim: int = 1,
        activation: str | type[nn.Module] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank not in (1, 3, 5):
            raise ValueError(f"rank must be 1, 3, or 5, got {rank}.")
        self.n_props = int(rank) // 2 + 1
        self.has_p3 = rank >= 3
        self.has_p5 = rank >= 5
        feature_dim = int(ii_nodes[-1])

        ii1_nodes = [int(v) for v in ii_nodes]
        ii1_nodes[-1] *= self.n_props
        self.invar_p1_layer = InvarLayer(
            p1_in_dim=int(p1_in_dim),
            pp_nodes=pp_nodes,
            pi_nodes=pi_nodes,
            ii_nodes=ii1_nodes,
            n_basis=n_basis,
            activation=activation,
            dtype=dtype,
        )
        self.equivar_p3_layer = (
            EquivarLayer(
                in_channels=p3_in_dim,
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                dtype=dtype,
            )
            if self.has_p3
            else None
        )
        self.equivar_p5_layer = (
            EquivarLayer(
                in_channels=p5_in_dim,
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                dtype=dtype,
            )
            if self.has_p5
            else None
        )
        pp1_nodes = [int(v) for v in pp_nodes]
        pp1_nodes[-1] = feature_dim * self.n_props
        self.pp_layer = FFLayer(
            pp1_nodes, in_dim=int(pp_nodes[-1]) * self.n_props, activation=activation, dtype=dtype
        )
        self.scale_layer = ScaleLayer()

    def forward(
        self,
        edge_index: torch.Tensor,
        basis: torch.Tensor,
        p1: torch.Tensor,
        p3: torch.Tensor | None = None,
        d3: torch.Tensor | None = None,
        p5: torch.Tensor | None = None,
        d5: torch.Tensor | None = None,
    ) -> GCOutput:
        """Run the block.

        Args:
            edge_index: Source/target pairs ``(E, 2)``.
            basis: Radial basis ``(E, B)``.
            p1: Scalar property ``(N, P)``.
            p3: Vector property ``(N, 3, C3)`` (rank ≥ 3).
            d3: Unit edge directions ``(E, 3)`` (rank ≥ 3).
            p5: Rank-2 property ``(N, 5, C5)`` (rank 5).
            d5: :func:`compute_d5` of ``d3`` (rank 5).

        Returns:
            :class:`GCOutput`.
        """
        p1_new, i1 = self.invar_p1_layer(edge_index, p1, basis)
        i1_chunks = torch.chunk(i1, self.n_props, dim=-1)
        mixed = [p1_new]
        p3_new = i3 = p5_new = i5 = None
        if self.equivar_p3_layer is not None:
            p3_new, i3, dotted = self.equivar_p3_layer(edge_index, p3, i1_chunks[1], d3)
            mixed.append(dotted)
        if self.equivar_p5_layer is not None:
            p5_new, i5, dotted = self.equivar_p5_layer(edge_index, p5, i1_chunks[2], d5)
            mixed.append(dotted)

        chunks = torch.chunk(self.pp_layer(torch.cat(mixed, dim=-1)), self.n_props, dim=-1)
        if p3_new is not None:
            p3_new = self.scale_layer(p3_new, chunks[1])
        if p5_new is not None:
            p5_new = self.scale_layer(p5_new, chunks[2])
        return GCOutput(p1=chunks[0], p3=p3_new, p5=p5_new, i1=i1, i3=i3, i5=i5)


class OutLayer(nn.Module):
    """Per-block output with residual accumulation (PiNN ``OutLayer``).

    ``out_k = Dense_biasless(FFLayer(p_k)) + out_{k-1}``. PiNN builds the
    ``FFLayer`` with no activation (Keras ``Dense`` default), so the default
    here is linear; the final projection carries no bias — the energy zero
    point belongs to the atomic reference energy.

    Args:
        n_nodes: FF widths.
        in_dim: Width of the block feature.
        out_units: Output channels.
        activation: FF activation; ``None`` (default) matches PiNN.

    Raises:
        ValueError: If ``n_nodes`` is empty.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        out_units: int = 1,
        activation: str | type[nn.Module] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if not n_nodes:
            raise ValueError("OutLayer requires at least one FF width.")
        self.ff_layer = FFLayer(
            n_nodes, in_dim=int(in_dim), activation=activation, use_bias=True, dtype=dtype
        )
        self.out_units = nn.Linear(int(n_nodes[-1]), int(out_units), bias=False, dtype=dtype)

    def forward(self, px: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        """Args: ``px`` ``(N, in_dim)``; ``prev_output`` ``(N, out_units)``. Returns its shape."""
        return self.out_units(self.ff_layer(px)) + prev_output
