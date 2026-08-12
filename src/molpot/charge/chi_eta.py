"""Per-atom electronegativity and hardness head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config

__all__ = ["ChiEtaHead"]


class ChiEtaHead(nn.Module):
    """Map atom features → electronegativity ``χ`` and hardness ``η > 0``.

    Single responsibility: parameter prediction only (no linear solve).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        min_eta: float = 1e-3,
    ) -> None:
        super().__init__()
        self.min_eta = float(min_eta)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=config.ftype),
        )

    def forward(self, atom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            atom_features: ``(N, D)``.

        Returns:
            ``{"chi": (N,), "eta": (N,)}`` with ``eta >= min_eta``.
        """
        raw = self.mlp(atom_features)
        chi = raw[:, 0]
        eta = F.softplus(raw[:, 1]) + self.min_eta
        return {"chi": chi, "eta": eta}
