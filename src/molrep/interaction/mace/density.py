"""Density-normalised equivariant interaction blocks (MACE-MP / MatPES variant).

Faithful port of MACE's ``RealAgnosticDensityInteractionBlock`` and
``RealAgnosticDensityResidualInteractionBlock`` on the cuEquivariance backend.
One edge→node message-passing layer with:

* a channel-wise tensor product ``node ⊗ Y_l(r̂)`` with neighbour scatter,
* a **learned density** normalisation ``message / (ρ + 1)`` where
  ``ρ_i = Σ_j tanh(MLP(edge_feats_ij)²)·u(r_ij)`` — this replaces the fixed
  ``avg_num_neighbors`` divisor and is what lets one model span the
  density range of the periodic table,
* an element-selecting ``skip_tp`` (fully-connected tensor product against the
  one-hot node attributes).

The two blocks differ only in where ``skip_tp`` acts: the first-layer variant
applies it to the outgoing message (there is no residual to carry), while the
residual variant applies it to the incoming node features and hands the result
to the downstream product block as the skip connection.

Built entirely from ``cuequivariance`` primitives in the ``cue.ir_mul`` layout;
all sub-layer names mirror MACE so the official weights transfer by direct copy.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molrep.embedding.mlp import MomentNormalizedMLP

#: cuEquivariance execution method for ``skip_tp``.
#:
#: ``skip_tp`` contracts the node features against a **one-hot** element vector
#: (``89x0e`` for a MatPES model), a shape cuEq's ``fused_tp`` kernel handles
#: badly: on GH200 it measures 33 ms against 1.7 ms for ``naive`` on a 193-atom
#: graph — a 20x penalty for bit-identical output (1.4e-15). ``naive`` is also
#: what MACE's own converted cueq model runs, so this default keeps us on the
#: reference's execution path as well as its math.
SKIP_TP_METHOD = "naive"


class _DensityInteractionBase(nn.Module):
    """Shared wiring of the two density-normalised interaction blocks.

    Subclasses own only ``skip_tp``'s irreps and the placement of the skip
    connection in :meth:`forward`.

    Args:
        node_attrs_irreps: One-hot atomic-number irreps, e.g. ``"89x0e"``.
        node_feats_irreps: Incoming node feature irreps.
        edge_attrs_irreps: Spherical-harmonics irreps, e.g. ``"1x0e+1x1o+1x2e+1x3o"``.
        edge_feats_irreps: Radial basis irreps, e.g. ``"10x0e"``.
        edge_irreps: ``linear_up`` output irreps.
        target_irreps: Tensor-product / block output irreps.
        radial_mlp: Hidden widths of the radial weight MLP, e.g. ``[64, 64, 64]``.
        use_fallback: Pure-torch cuEq path (default ``True``, functorch-safe).
            Set ``False`` for fused kernels when forces use autograd.
        skip_tp_method: cuEquivariance execution method for ``skip_tp``; see
            :data:`SKIP_TP_METHOD`.
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
        radial_mlp: list[int],
        use_fallback: bool = True,
        skip_tp_method: str = SKIP_TP_METHOD,
    ) -> None:
        super().__init__()
        ftype = config.ftype
        self.use_fallback = use_fallback
        self.skip_tp_method = skip_tp_method

        node_feats = cue.Irreps("O3", node_feats_irreps)
        edge_attrs = cue.Irreps("O3", edge_attrs_irreps)
        edge_feats = cue.Irreps("O3", edge_feats_irreps)
        edge_ir = cue.Irreps("O3", edge_irreps)
        target = cue.Irreps("O3", target_irreps)
        self.node_attrs_irreps = cue.Irreps("O3", node_attrs_irreps)
        self.irreps_out = target

        self.linear_up = cuet.Linear(node_feats, edge_ir, layout=cue.ir_mul, dtype=ftype)

        self.conv_tp = cuet.ChannelWiseTensorProduct(
            edge_ir,
            edge_attrs,
            target,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            dtype=ftype,
            use_fallback=use_fallback,
        )

        num_radial = edge_feats.dim
        self.conv_tp_weights = MomentNormalizedMLP(
            [num_radial] + list(radial_mlp) + [self.conv_tp.weight_numel]
        )
        self.density_fn = MomentNormalizedMLP([num_radial, 1])

        self.linear = cuet.Linear(self.conv_tp.irreps_out, target, layout=cue.ir_mul, dtype=ftype)

        # reshape_irreps (ir_mul): per-irrep (N, mul*d) -> (N, d, mul), cat on dim -2
        self._reshape_dims = [(mi.mul, mi.ir.dim) for mi in target]

    def _build_skip_tp(self, dtype: torch.dtype) -> cuet.FullyConnectedTensorProduct:
        """Construct ``skip_tp`` at ``dtype``; subclasses choose the irreps."""
        raise NotImplementedError

    def _apply(self, *args, **kwargs):
        """Rebuild ``skip_tp`` when the module's dtype changes.

        ``cuet.FullyConnectedTensorProduct`` compiles its contraction graph at
        construction and bakes the working precision into it, so
        ``nn.Module.double()`` converts the parameters but leaves the graph in
        float32 — the next forward then raises a bare dtype mismatch deep inside
        cuEquivariance. Rebuilding at the new dtype and re-adopting the (already
        converted) weight makes ``Model(...).double()`` behave like it does for
        every other module here, instead of only supporting construction under
        ``molix.config.set_precision("fp64")``.
        """
        module = super()._apply(*args, **kwargs)
        weight = module.skip_tp.weight
        if weight.dtype != module._skip_tp_dtype:
            rebuilt = module._build_skip_tp(weight.dtype).to(weight.device)
            with torch.no_grad():
                rebuilt.weight.copy_(weight)
            rebuilt.weight.requires_grad_(weight.requires_grad)
            module.skip_tp = rebuilt
            module._skip_tp_dtype = weight.dtype
        return module

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        ix, out, batch = 0, [], tensor.shape[0]
        for mul, d in self._reshape_dims:
            field = tensor[:, ix : ix + mul * d].reshape(batch, d, mul)
            ix += mul * d
            out.append(field)
        return torch.cat(out, dim=-2)

    def _message(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        """Density-normalised aggregated message ``(N, target.dim)``."""
        num_nodes = node_feats.shape[0]
        source, target = edge_index[:, 0], edge_index[:, 1]
        node_feats = self.linear_up(node_feats)

        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = _scatter_sum(edge_density, target, num_nodes)

        mji = self.conv_tp(node_feats[source], edge_attrs, tp_weights)
        message = _scatter_sum(mji, target, num_nodes)
        return self.linear(message) / (density + 1.0)


class DensityInteraction(_DensityInteractionBase):
    """First-layer density-normalised interaction (MACE ``RealAgnosticDensity``).

    ``skip_tp`` mixes the outgoing message with the one-hot element attributes;
    there is no residual to pass on, so the returned skip connection is ``None``.

    Args:
        See :class:`_DensityInteractionBase`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.skip_tp = self._build_skip_tp(config.ftype)
        self._skip_tp_dtype = config.ftype

    def _build_skip_tp(self, dtype: torch.dtype) -> cuet.FullyConnectedTensorProduct:
        """``message ⊗ one_hot(Z) → message``: element-selective output mixing."""
        return cuet.FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            layout=cue.ir_mul,
            dtype=dtype,
            method=self.skip_tp_method,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run one first-layer interaction.

        Args:
            node_attrs: One-hot atomic numbers ``(N, n_elements)``.
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_radial)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(reshaped_message (N, ir_dim, mul), None)``.
        """
        message = self._message(node_feats, edge_attrs, edge_feats, edge_index, cutoff)
        message = self.skip_tp(message, node_attrs)
        return self._reshape(message), None


class DensityResidualInteraction(_DensityInteractionBase):
    """Residual density-normalised interaction (MACE ``RealAgnosticDensityResidual``).

    ``skip_tp`` maps the *incoming* node features (mixed with the one-hot element
    attributes) onto ``hidden_irreps``; the result is the skip connection the
    downstream product block adds back.

    Args:
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
        Other arguments: see :class:`_DensityInteractionBase`.
    """

    def __init__(self, *, hidden_irreps: str, node_feats_irreps: str, **kwargs) -> None:
        super().__init__(node_feats_irreps=node_feats_irreps, **kwargs)
        self._skip_tp_in = cue.Irreps("O3", node_feats_irreps)
        self._skip_tp_out = cue.Irreps("O3", hidden_irreps)
        self.skip_tp = self._build_skip_tp(config.ftype)
        self._skip_tp_dtype = config.ftype

    def _build_skip_tp(self, dtype: torch.dtype) -> cuet.FullyConnectedTensorProduct:
        """``node_feats ⊗ one_hot(Z) → hidden``: the residual the product adds back."""
        return cuet.FullyConnectedTensorProduct(
            self._skip_tp_in,
            self.node_attrs_irreps,
            self._skip_tp_out,
            layout=cue.ir_mul,
            dtype=dtype,
            method=self.skip_tp_method,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one residual interaction.

        Args:
            node_attrs: One-hot atomic numbers ``(N, n_elements)``.
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_radial)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(reshaped_message (N, ir_dim, mul), skip (N, hidden_irreps.dim))``.
        """
        sc = self.skip_tp(node_feats, node_attrs)
        message = self._message(node_feats, edge_attrs, edge_feats, edge_index, cutoff)
        return self._reshape(message), sc
