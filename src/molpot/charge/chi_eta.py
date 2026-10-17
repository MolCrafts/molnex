"""Per-atom electronegativity and hardness head."""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config

__all__ = ["ChiEtaHead"]


class ChiEtaHead(nn.Module):
    """Map atom features → electronegativity ``χ`` and hardness ``η > 0``.

    Single responsibility: parameter prediction only (no linear solve).
    ``χ`` is tanh-bounded to ``[-max_chi, max_chi]`` and ``η`` is
    sigmoid-bounded to ``[min_eta, max_eta]`` so the EEM solve stays finite.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        min_eta: float = 1e-3,
        max_eta: float = 20.0,
        max_chi: float = 5.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if max_eta <= min_eta:
            raise ValueError(f"max_eta {max_eta} must exceed min_eta {min_eta}")
        if max_chi <= 0:
            raise ValueError(f"max_chi {max_chi} must be positive")
        self.min_eta = float(min_eta)
        self.max_eta = float(max_eta)
        self.max_chi = float(max_chi)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=dtype),
        )
        # Zeroed like every parameter head: at init ``chi = 0`` and ``eta`` sits
        # mid-range, so the solve hands every atom the same charge ``Q/N``
        # instead of a random partition.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, atom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            atom_features: ``(N, D)``.

        Returns:
            ``{"chi": (N,), "eta": (N,)}`` with ``|chi| <= max_chi`` and
            ``min_eta <= eta <= max_eta``.
        """
        raw = self.mlp(atom_features)
        chi = self.max_chi * torch.tanh(raw[:, 0])
        eta = self.min_eta + (self.max_eta - self.min_eta) * torch.sigmoid(raw[:, 1])
        return {"chi": chi, "eta": eta}
