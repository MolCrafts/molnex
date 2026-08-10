"""Leave-one-out k-NN type purity on atom latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from molrep.analysis.latent_store import AtomLatentTable

__all__ = ["NearestNeighbourTypePurity", "TypePurityReport"]


@dataclass(frozen=True)
class TypePurityReport:
    mean_purity: float | None
    k: int
    n_scored: int
    n_labeled: int


class NearestNeighbourTypePurity:
    """Leave-one-out L2 k-NN type purity.

    For each labeled atom, find k nearest other labeled atoms (by L2 on
    features) and score the fraction whose type matches. Mean over scored
    atoms. Unlabeled ``y_i == -1`` are excluded.

    Args:
        k: Neighbourhood size (default 1).
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def score(self, table: AtomLatentTable) -> TypePurityReport:
        if table.ref_atom_type is None:
            return TypePurityReport(mean_purity=None, k=self.k, n_scored=0, n_labeled=0)
        y = table.ref_atom_type
        labeled = y >= 0
        n_labeled = int(labeled.sum().item())
        if n_labeled < 2:
            return TypePurityReport(mean_purity=None, k=self.k, n_scored=0, n_labeled=n_labeled)
        feats = table.features
        idx = torch.where(labeled)[0]
        purities: list[float] = []
        for i in idx.tolist():
            others = idx[idx != i]
            if others.numel() == 0:
                continue
            d = torch.norm(feats[others] - feats[i], dim=-1)
            kk = min(self.k, int(others.numel()))
            nn = others[torch.topk(d, k=kk, largest=False).indices]
            match = (y[nn] == y[i]).float().mean().item()
            purities.append(float(match))
        if not purities:
            return TypePurityReport(mean_purity=None, k=self.k, n_scored=0, n_labeled=n_labeled)
        return TypePurityReport(
            mean_purity=sum(purities) / len(purities),
            k=self.k,
            n_scored=len(purities),
            n_labeled=n_labeled,
        )
