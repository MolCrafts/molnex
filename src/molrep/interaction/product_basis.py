"""Equivariant product-basis block (MACE body-order expansion).

Faithful port of MACE's ``EquivariantProductBasisBlock`` on the
cuEquivariance backend: a per-element (or element-agnostic) symmetric
contraction that raises the body order of the node features, followed by an
equivariant linear and an optional residual skip add.

Built from ``cuequivariance`` primitives in the ``cue.ir_mul`` layout; sub-layer
names (``symmetric_contractions``, ``linear``) mirror MACE so the official
weights transfer by direct copy.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022, Eq. (10)-(11).
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn

from molix import config


class EquivariantProductBasis(nn.Module):
    """Symmetric-contraction product basis + linear (MACE product block).

    Args:
        node_feats_irreps: Incoming node feature irreps (interaction output),
            e.g. ``"1024x0e+1024x1o+1024x2e+1024x3o"``.
        target_irreps: Output irreps, e.g. ``"1024x0e+1024x1o+1024x2e"``.
        correlation: Body-order correlation (symmetric-contraction degree).
        num_elements: Element table size; ``1`` for an element-agnostic product
            (OMOL), in which case all atoms share one weight set.
        use_sc: Whether to add the skip connection ``sc`` from the interaction.
    """

    def __init__(
        self,
        *,
        node_feats_irreps: str,
        target_irreps: str,
        correlation: int,
        num_elements: int = 1,
        use_sc: bool = True,
    ) -> None:
        super().__init__()
        ftype = config.ftype
        self.use_sc = use_sc
        self.num_elements = int(num_elements)

        irreps_in = cue.Irreps("O3", node_feats_irreps)
        irreps_out = cue.Irreps("O3", target_irreps)

        self.symmetric_contractions = cuet.SymmetricContraction(
            irreps_in,
            irreps_out,
            contraction_degree=correlation,
            num_elements=self.num_elements,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
            original_mace=True,
            dtype=ftype,
            math_dtype=ftype,
        )
        self.linear = cuet.Linear(irreps_out, irreps_out, layout=cue.ir_mul, dtype=ftype)

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """Raise body order and project.

        Args:
            node_feats: ``(N, ir_dim, mul)`` multiplet features from the
                interaction's reshape (``cue.ir_mul`` order).
            sc: Skip connection ``(N, target_irreps.dim)`` or ``None``.
            node_attrs: One-hot atomic numbers ``(N, n_elements)``; ignored when
                ``num_elements == 1`` (element-agnostic).

        Returns:
            ``(N, target_irreps.dim)`` node features.
        """
        if self.num_elements == 1:
            index = torch.zeros(node_feats.shape[0], dtype=torch.int32, device=node_feats.device)
        else:
            index = node_attrs.argmax(dim=-1).to(torch.int32)
        out = self.symmetric_contractions(node_feats.flatten(1), index)
        out = self.linear(out)
        if self.use_sc and sc is not None:
            out = out + sc
        return out
