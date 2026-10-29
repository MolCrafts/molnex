"""MACE's charge/spin conditioning: per-graph scalars fused into the node features.

MACE's ``GenericJointEmbedding`` (OMOL charge/spin conditioning): each extra
per-graph feature (total charge, total spin) is embedded, the embeddings are
concatenated and projected, and the result is added to the initial node
features before the first interaction. Sub-layer names (``embedders.<name>``,
``project.0``) stay as they are so an official checkpoint still copies onto them.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022;
    OMol charge/spin conditioning. https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

__all__ = ["JointFeatureEmbedding", "JointFeatureSpec"]


class JointFeatureSpec(BaseModel):
    """Specification for one extra scalar feature in :class:`JointFeatureEmbedding`.

    Attributes:
        name: Feature name; also the lookup key in the ``features`` dict and the
            ``embedders`` ``ModuleDict``.
        kind: ``"categorical"`` (embedding table) or ``"continuous"`` (MLP).
        emb_dim: Embedding dimension produced for this feature.
        num_classes: Vocabulary size (categorical only).
        in_dim: Input dimension (continuous only).
        per: ``"graph"`` features are per-molecule and broadcast to atoms via the
            ``batch`` index; ``"node"`` features are already per-atom.
        offset: Added to a categorical index before lookup (e.g. shift signed
            charges into ``[0, num_classes)``).
        use_bias: Whether the continuous MLP uses bias.
    """

    name: str
    kind: Literal["categorical", "continuous"]
    emb_dim: int = Field(..., gt=0)
    num_classes: int | None = Field(None, gt=0)
    in_dim: int | None = Field(None, gt=0)
    per: Literal["graph", "node"] = "graph"
    offset: int = 0
    use_bias: bool = True


class JointFeatureEmbedding(nn.Module):
    """Fuse extra scalar node/graph features (e.g. total charge & spin) into
    the node features, faithful to MACE's ``GenericJointEmbedding``.

    Each feature is embedded independently — categorical via an
    ``nn.Embedding`` table, continuous via ``Linear → SiLU → Linear`` — the
    embeddings are concatenated and passed through a bias-free
    ``Linear → SiLU`` projection. Per-graph features are broadcast to atoms via
    the ``batch`` index. The result is a per-atom scalar feature vector
    ``(N, out_dim)`` that is **added** to the node embedding.

    Submodule names (``embedders.<name>`` and ``project.0``) mirror MACE so a
    weight converter is a direct copy.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022;
        OMol charge/spin conditioning. https://arxiv.org/abs/2206.07697
    """

    def __init__(
        self,
        *,
        feature_specs: list[JointFeatureSpec],
        out_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the joint feature embedding.

        Args:
            feature_specs: One spec per extra feature (e.g. total_charge,
                total_spin).
            out_dim: Output dimension of the projection (must equal the node
                feature channel count it is added to).
            dtype: Floating dtype of every table and linear (the owning module's).
        """
        super().__init__()
        if not feature_specs:
            raise ValueError("feature_specs must be non-empty.")
        self.specs = feature_specs
        self.out_dim = int(out_dim)

        self.embedders = nn.ModuleDict()
        for s in feature_specs:
            if s.kind == "categorical":
                if s.num_classes is None:
                    raise ValueError(f"categorical feature {s.name!r} needs num_classes.")
                self.embedders[s.name] = nn.Embedding(s.num_classes, s.emb_dim, dtype=dtype)
            else:
                if s.in_dim is None:
                    raise ValueError(f"continuous feature {s.name!r} needs in_dim.")
                self.embedders[s.name] = nn.Sequential(
                    nn.Linear(s.in_dim, s.emb_dim, bias=s.use_bias, dtype=dtype),
                    nn.SiLU(),
                    nn.Linear(s.emb_dim, s.emb_dim, bias=s.use_bias, dtype=dtype),
                )

        total_dim = sum(s.emb_dim for s in feature_specs)
        self.project = nn.Sequential(
            nn.Linear(total_dim, self.out_dim, bias=False, dtype=dtype),
            nn.SiLU(),
        )

    def forward(self, batch: torch.Tensor, **features: torch.Tensor) -> torch.Tensor:
        """Embed and fuse the extra features.

        Args:
            batch: Per-atom graph index ``(N,)`` used to broadcast per-graph
                features to atoms.
            **features: One tensor per spec ``name``; per-graph features are
                shaped ``(B,)``/``(B, ...)``, per-node ones ``(N,)``/``(N, ...)``.

        Returns:
            Fused per-atom feature ``(N, out_dim)``.
        """
        embs = []
        for s in self.specs:
            feat = features[s.name]
            if s.per == "graph":
                feat = feat[batch].unsqueeze(-1)
            if s.kind == "categorical":
                feat = (feat + s.offset).long().squeeze(-1)
            elif feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            embs.append(self.embedders[s.name](feat))
        return self.project(torch.cat(embs, dim=-1))
