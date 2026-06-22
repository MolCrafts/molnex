"""Residual non-linear equivariant interaction block (MACE OMOL variant).

Faithful port of MACE's ``RealAgnosticResidualNonLinearInteractionBlock`` on
the cuEquivariance backend. One edge→node message-passing layer with:

* node-attribute source/target embeddings concatenated onto the radial edge
  features that drive the tensor-product weights,
* a channel-wise tensor product ``node ⊗ Y_l(r̂)`` with neighbour scatter,
* a learnable density normalisation ``message / (ρ·β + α)``,
* a residual (``linear_res``) path and an equivariant gated nonlinearity,
* a separate ``skip_tp`` (Linear) carried out for the downstream product block.

Built entirely from ``cuequivariance`` primitives in the ``cue.ir_mul`` layout;
all sub-layer names mirror MACE so the official weights transfer by direct copy.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config

from .gate import GatedNonlinearity
from .radial import RadialMLP


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (dim 0).

    Implemented as a one-hot matmul instead of ``index_add_``. The natural
    ``out.index_add_(0, index, src)`` is correct in eager, but under
    ``torch.compile`` its scatter reduction runs in a *different order* than
    eager (CUDA: non-deterministic ``atomicAdd``; CPU: it additionally trips
    the torch-2.12 ``index.is_vec`` SIMD-codegen crash). When the scattered
    energy is a small difference of large terms (catastrophic cancellation, as
    in MACE ``inter_e``), that float64 ordering difference is amplified to ~4%
    between compiled and eager. This is NOT a wrong-codegen miscompile: forcing
    ``torch.use_deterministic_algorithms(True)`` (CUDA) or
    ``torch._inductor.config.cpp.simdlen = 0`` (CPU) makes ``index_add_``
    bit-exact again. The one-hot matmul is deterministic and has no scatter
    node, so it is bit-exact under compile on both backends with NO global
    flags, and stays functorch-traceable. Cost: a dense ``(E, dim_size)``
    one-hot = ``O(E * dim_size)`` — negligible for molecular graphs, but heavy
    for very large periodic systems.

    TODO: switch back to the cheaper ``index_add_`` once we either (a) move to
    a torch where the compiled scatter is deterministic by default, or (b) set
    the determinism / ``cpp.simdlen`` knobs on the compile path — needed to
    drop the O(E*dim_size) memory for large periodic systems.
    """
    onehot = (index.view(-1, 1) == torch.arange(dim_size, device=src.device).view(1, -1)).to(src.dtype)
    return (onehot.t() @ src.reshape(src.shape[0], -1)).reshape(dim_size, *src.shape[1:])


def _partition_gate(irreps_out: cue.Irreps) -> tuple[cue.Irreps, cue.Irreps, cue.Irreps]:
    """Split output irreps into (scalars, gates, gated) for the gated nonlinearity."""
    scalars, gated = [], []
    for mi in irreps_out:
        (scalars if mi.ir.l == 0 else gated).append(mi)
    gates = [(mi.mul, "0e") for mi in gated]
    return (
        cue.Irreps("O3", scalars),
        cue.Irreps("O3", " + ".join(f"{m}x{ir}" for m, ir in gates) if gates else ""),
        cue.Irreps("O3", gated),
    )


class ResidualInteraction(nn.Module):
    """MACE residual non-linear interaction layer (cuEquivariance).

    Args:
        node_attrs_irreps: One-hot atomic-number irreps, e.g. ``"83x0e"``.
        node_feats_irreps: Incoming node feature irreps.
        edge_attrs_irreps: Spherical-harmonics irreps, e.g. ``"1x0e+1x1o+1x2e+1x3o"``.
        edge_feats_irreps: Radial basis irreps, e.g. ``"8x0e"`` (scalar count = num_bessel).
        edge_irreps: ``linear_up`` output irreps (per-layer reduced channels).
        target_irreps: Tensor-product / output irreps (also the gated nonlin output).
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
        radial_mlp: Hidden sizes of the radial MLPs, e.g. ``[128, 128, 128]``.
    """

    def __init__(
        self,
        *,
        node_attrs_irreps: str,
        node_feats_irreps: str,
        edge_attrs_irreps: str,
        edge_feats_irreps: str,
        edge_irreps: str,
        target_irreps: str,
        hidden_irreps: str,
        radial_mlp: list[int],
    ) -> None:
        super().__init__()
        ftype = config.ftype

        node_feats = cue.Irreps("O3", node_feats_irreps)
        edge_attrs = cue.Irreps("O3", edge_attrs_irreps)
        edge_feats = cue.Irreps("O3", edge_feats_irreps)
        edge_ir = cue.Irreps("O3", edge_irreps)
        target = cue.Irreps("O3", target_irreps)
        hidden = cue.Irreps("O3", hidden_irreps)
        node_attrs = cue.Irreps("O3", node_attrs_irreps)

        n_scalar = sum(mi.mul for mi in node_feats if mi.ir.l == 0)
        node_scalar = cue.Irreps("O3", f"{n_scalar}x0e")

        def lin(i_in, i_out):
            return cuet.Linear(i_in, i_out, layout=cue.ir_mul, dtype=ftype)

        self.source_embedding = lin(node_attrs, node_scalar)
        self.target_embedding = lin(node_attrs, node_scalar)
        self.linear_up = lin(node_feats, edge_ir)

        self.conv_tp = cuet.ChannelWiseTensorProduct(
            edge_ir,
            edge_attrs,
            target,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            dtype=ftype,
            # Pure-torch path so functorch (ForceDerivation) / torch.compile can
            # trace the force: the fused kernel has no functorch setup_context.
            use_fallback=True,
        )
        irreps_mid = self.conv_tp.irreps_out

        input_dim = edge_feats.dim + 2 * node_scalar.dim
        self.conv_tp_weights = RadialMLP(
            [input_dim] + list(radial_mlp) + [self.conv_tp.weight_numel]
        )

        self.skip_tp = lin(node_feats, hidden)

        irreps_scalars, irreps_gates, irreps_gated = _partition_gate(target)
        self.equivariant_nonlin = GatedNonlinearity(
            irreps_scalars,
            [F.silu],
            irreps_gates,
            [torch.sigmoid] * len(list(irreps_gated)),
            irreps_gated,
            layout="ir_mul",
        )
        irreps_nonlin = self.equivariant_nonlin.irreps_in

        self.linear_res = lin(edge_ir, irreps_nonlin)
        self.linear_1 = lin(irreps_mid, irreps_nonlin)
        self.linear_2 = lin(target, target)

        self.density_fn = RadialMLP([input_dim, 64, 1])
        self.alpha = nn.Parameter(torch.tensor(20.0, dtype=ftype))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=ftype))

        # reshape_irreps (ir_mul): per-irrep (N, mul*d) -> (N, d, mul), cat on dim -2
        self._reshape_dims = [(mi.mul, mi.ir.dim) for mi in target]

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        ix, out, batch = 0, [], tensor.shape[0]
        for mul, d in self._reshape_dims:
            field = tensor[:, ix : ix + mul * d].reshape(batch, d, mul)
            ix += mul * d
            out.append(field)
        return torch.cat(out, dim=-2)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one residual non-linear interaction layer.

        Args:
            node_attrs: One-hot atomic numbers ``(N, n_elements)``.
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_bessel)``.
            edge_index: ``(2, E)`` with row 0 = sender, row 1 = receiver.
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(reshaped_message (N, ir_dim, mul), skip (N, hidden_dim))``.
        """
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats)
        node_feats = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_feats)

        source = self.source_embedding(node_attrs)
        target = self.target_embedding(node_attrs)
        edge_feats = torch.cat([edge_feats, source[edge_index[0]], target[edge_index[1]]], dim=-1)
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = _scatter_sum(edge_density, edge_index[1], num_nodes)

        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = _scatter_sum(mji, edge_index[1], num_nodes)

        message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
        message = self.equivariant_nonlin(message)
        message = self.linear_2(message)
        return self._reshape(message), sc
