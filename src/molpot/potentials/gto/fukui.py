"""Fukui total-charge / total-spin equilibration (PolarMACE).

``total_spin`` is the ASE multiplicity ``M = 2S+1``. Spin-channel
monopole targets are ``(Q ± (M-1))/2``. Fukui weights are L1-normalized
per graph so they partition the residual.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum_compile_safe


class FukuiEquilibration(nn.Module):
    """Add Fukui-weighted residuals onto spin-resolved monopoles.

    Args of :meth:`forward`:
        spin_density: ``(N, 2, (l_max+1)**2)`` — channel 0/1 are spin-up/down.
        fukui_sources: ``(N, 2)`` raw Fukui weights (e.g. ``NonLinearBiasReadout``).
        batch: Graph id per atom ``(N,)``.
        total_charge: Per-graph charge ``(B,)`` in e.
        total_spin: Per-graph multiplicity ``M=2S+1``, dimensionless.

    Returns:
        Updated ``spin_density``; only the monopole ``[..., 0]`` changes.
    """

    def forward(
        self,
        spin_density: torch.Tensor,
        fukui_sources: torch.Tensor,
        batch: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
    ) -> torch.Tensor:
        n_graphs = int(total_charge.shape[0])
        fukui_norm = scatter_sum_compile_safe(fukui_sources, batch, n_graphs)[batch]
        fukui_norm = torch.where(fukui_norm == 0, torch.ones_like(fukui_norm), fukui_norm)
        fukui = fukui_sources / fukui_norm
        unpaired = total_spin - 1.0
        target_up = (total_charge + unpaired)[batch] / 2.0
        target_down = (total_charge - unpaired)[batch] / 2.0
        pred = scatter_sum_compile_safe(spin_density[:, :, 0], batch, n_graphs)[batch]
        out = spin_density.clone()
        out[:, 0, 0] = out[:, 0, 0] + fukui[:, 0] * (target_up - pred[:, 0])
        out[:, 1, 0] = out[:, 1, 0] + fukui[:, 1] * (target_down - pred[:, 1])
        return out
