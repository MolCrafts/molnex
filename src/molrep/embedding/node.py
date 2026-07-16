"""Embedding modules for molrep encoders."""

from __future__ import annotations

from typing import Literal

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator


class DiscreteEmbeddingSpec(BaseModel):
    """Specification for a single discrete feature embedding.

    This spec defines how to embed discrete categorical features (like atomic
    numbers) into continuous vectors.

    Attributes:
        input_key: Key to read discrete features from. Can be a
            string like "Z" or nested tuple like ("atom", "Z").
        num_classes: Number of discrete categories (vocabulary size).
            Must be positive.
        emb_dim: Dimension of the output embedding vectors. Must be positive.
    """

    input_key: str | tuple[str]
    num_classes: int = Field(..., gt=0)
    emb_dim: int = Field(..., gt=0)


class ContinuousEmbeddingSpec(BaseModel):
    """Specification for a single continuous feature embedding.

    This spec defines how to embed continuous features (like positions or
    distances) into learned representations via an MLP.

    Attributes:
        input_key: Key to read continuous features from.
        in_dim: Input feature dimension. Must be positive.
        emb_dim: Output embedding dimension. Must be positive.
        use_bias: Whether to use bias in linear layers. Defaults to True.
    """

    input_key: str | tuple[str]
    in_dim: int = Field(..., gt=0)
    emb_dim: int = Field(..., gt=0)
    use_bias: bool = True


class JointEmbeddingSpec(BaseModel):
    """Specification for a joint embedding module.

    Defines how to combine multiple discrete and/or continuous feature
    embeddings via concatenation and projection.

    Attributes:
        specs: List of individual embedding specs to combine.
        out_dim: Final output dimension after projection. Must be positive.
        output_key: Key to write combined embeddings to.

    Note:
        The input_keys property is automatically derived from the specs.
        Duplicate keys in specs are not allowed.
    """

    specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec]
    out_dim: int = Field(..., gt=0)
    output_key: str | tuple[str]

    @property
    def input_keys(self) -> list[str | tuple[str]]:
        """Derive input keys from specs.

        Returns:
            List of input keys extracted from all specs in order.
        """
        return [spec.input_key for spec in self.specs]

    @model_validator(mode="after")
    def _validate_spec(self) -> "JointEmbeddingSpec":
        """Validate that spec keys are unique.

        Returns:
            Self after validation.

        Raises:
            ValueError: If specs contain duplicate keys.
        """
        spec_keys = [spec.input_key for spec in self.specs]
        if len(spec_keys) != len(set(spec_keys)):
            raise ValueError("specs contain duplicate keys")

        return self


class JointEmbedding(nn.Module):
    """Joint embedding module with concatenation fusion.

    Embeds multiple discrete and/or continuous features, concatenates them,
    and projects to a unified representation space.

    For discrete features: Uses nn.Embedding lookup.
    For continuous features: Uses 2-layer MLP with SiLU activation.

    The concatenated features are projected through a linear layer with
    SiLU activation to produce the final output.

    Attributes:
        config: JointEmbeddingSpec configuration.
        embedders: ModuleList of embedding layers in spec order.
        project: Projection network from concatenated to output dimension.
    """

    def __init__(
        self,
        *,
        embedding_specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec],
        out_dim: int,
    ):
        """Initialize joint embedding module.

        Args:
            embedding_specs: List of embedding specifications to combine.
            out_dim: Output dimension after projection.

        Raises:
            ValueError: If no embedding specs provided.
        """
        super().__init__()

        self.config = JointEmbeddingSpec(
            specs=embedding_specs,
            out_dim=out_dim,
            output_key="_unused",  # Placeholder since we are removing keys
        )

        # Store embedders in a list to match spec order
        self.embedders = nn.ModuleList()
        for spec in self.config.specs:
            if isinstance(spec, DiscreteEmbeddingSpec):
                self.embedders.append(nn.Embedding(spec.num_classes, spec.emb_dim))
            else:
                self.embedders.append(
                    nn.Sequential(
                        nn.Linear(spec.in_dim, spec.emb_dim, bias=spec.use_bias),
                    )
                )

        total_dim = sum(spec.emb_dim for spec in self.config.specs)
        if total_dim == 0:
            raise ValueError("No feature embeddings configured.")

        self.project = nn.Sequential(
            cuet.Linear(
                cue.Irreps("O3", f"{total_dim}x0e"),
                cue.Irreps("O3", f"{out_dim}x0e"),
                layout=cue.ir_mul,
            ),
        )

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        """Embed features and project to output dimension.

        Args:
            **features: Input tensors containing features specified by
                config.input_keys.

        Returns:
            Concatenated and projected embedding tensor.
        """
        embs = []
        for i, spec in enumerate(self.config.specs):
            # Input keys can be tuples (bonds, dist) or strings
            if isinstance(spec.input_key, tuple):
                # Flatten hierarchical keys if they arrive as flattened kwargs
                # or handle them if they arrive as nested dicts?
                # The user said "don't pass dictionaries", so likely we
                # should use a unique string identifier.
                key = "_".join(spec.input_key)
            else:
                key = spec.input_key

            feat = features[key]
            emb = self.embedders[i](feat)
            embs.append(emb)

        return self.project(torch.cat(embs, dim=-1))


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
    ) -> None:
        """Initialize the joint feature embedding.

        Args:
            feature_specs: One spec per extra feature (e.g. total_charge,
                total_spin).
            out_dim: Output dimension of the projection (must equal the node
                feature channel count it is added to).
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
                self.embedders[s.name] = nn.Embedding(s.num_classes, s.emb_dim)
            else:
                if s.in_dim is None:
                    raise ValueError(f"continuous feature {s.name!r} needs in_dim.")
                self.embedders[s.name] = nn.Sequential(
                    nn.Linear(s.in_dim, s.emb_dim, bias=s.use_bias),
                    nn.SiLU(),
                    nn.Linear(s.emb_dim, s.emb_dim, bias=s.use_bias),
                )

        total_dim = sum(s.emb_dim for s in feature_specs)
        self.project = nn.Sequential(
            nn.Linear(total_dim, self.out_dim, bias=False),
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
