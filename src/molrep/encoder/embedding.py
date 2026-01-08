"""Atom embedding module for padded tensor inputs."""

import torch
import torch.nn as nn
from tensordict import TensorDict


class AtomTypeEmbedding(nn.Module):
    def __init__(self, num_types: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(num_types, d_model, padding_idx=padding_idx)

    def forward(self, td):
        Z = td["atoms", "Z"]              # [B, L]
        td["atoms", "h"] = self.emb(Z)    # [B, L, d_model] (padded)
        return td

class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, r_min: float = 0.0, r_max: float = 10.0, gamma: float | None = None):
        super().__init__()
        centers = torch.linspace(r_min, r_max, num_rbf)
        self.register_buffer("centers", centers)
        if gamma is None:
            gamma = (num_rbf / (r_max - r_min + 1e-8)) ** 2
        self.gamma = gamma

    def forward(self, dist):  # [B,L,L] -> [B,L,L,K]
        return torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)