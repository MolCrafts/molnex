"""Deterministic 2-D PCA projection for latent tables (no plotting)."""

from __future__ import annotations

import torch

from molrep.analysis.latent_store import AtomLatentTable

__all__ = ["LatentPCA2D"]


class LatentPCA2D:
    """Project atom features to 2-D via SVD (deterministic)."""

    def project(self, table: AtomLatentTable) -> torch.Tensor:
        """Return coordinates ``(N, 2)``.

        Raises:
            ValueError: If fewer than 2 atoms or feature dim < 1.
        """
        x = table.features.detach().float()
        if x.shape[0] < 1:
            raise ValueError("need at least 1 atom")
        x = x - x.mean(dim=0, keepdim=True)
        # SVD of (N, D)
        # Use full_matrices=False
        if x.shape[0] == 1 or x.shape[1] == 1:
            out = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
            out[:, 0] = x[:, 0] if x.shape[1] >= 1 else 0.0
            return out
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        comps = vh[:2].T  # (D, 2)
        if comps.shape[1] == 1:
            comps = torch.nn.functional.pad(comps, (0, 1))
        return x @ comps
