"""MACE conditioning embedding and scalar readouts.

The maps that sit at the two ends of the MACE body: the extra-feature
embedding that fuses per-graph scalars (total charge, spin) into the node
features before the first interaction, and the scalar readouts that turn the
final node features into per-atom energies. A product chooses which readouts to
attach, and how many:

- :class:`JointFeatureEmbedding` — MACE's ``GenericJointEmbedding`` (OMOL
  charge/spin conditioning), configured by :class:`JointFeatureSpec`.
- :class:`LinearReadout` — one ``cuet.Linear`` onto ``1x0e``.
- :class:`NonLinearReadout` — bias-free ``Linear → c·SiLU → Linear``.
- :class:`NonLinearBiasReadout` — ``Linear → SiLU → biased linear → SiLU →
  biased linear``. ``out_mul`` is 1 for an energy and 2 when a caller wants
  two scalar channels (the polar Fukui source).

Sub-layer names (``embedders.<name>``, ``project.0``, ``linear_1``,
``linear_mid``, ``linear_2``) stay as they are so an official checkpoint still
copies onto them.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022;
    OMol charge/spin conditioning. https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import math
from typing import Literal

import cuequivariance as cue
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

from molrep import cueq
from molrep.embedding.mlp import normalize2mom


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


class _ScalarO3Linear(nn.Module):
    """Scalar-only equivariant linear with bias (upstream MACE): ``(x @ W/sqrt(in)) + b``.

    Initialisation follows upstream MACE's linear-layer convention: the weight is drawn
    from the **global** RNG as standard normal ``N(0, 1)`` and the path
    normalisation ``1/sqrt(in_mul)`` is applied in :meth:`forward`
    (``self._alpha``) rather than folded into the init, so the stored weight
    stays unit-variance. The bias is zero-initialised.

    A zero weight (the previous init) makes the layer emit a constant per-atom
    energy on an untrained model, which silently zeroes the forces and voids
    every downstream force / parity assertion. Loading a checkpoint overwrites
    the random init exactly, so weight-transfer paths are unaffected.
    """

    def __init__(self, in_mul: int, out_mul: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_mul, self.out_mul = in_mul, out_mul
        self.weight = nn.Parameter(torch.randn(in_mul * out_mul, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_mul, dtype=dtype))
        self._alpha = 1.0 / math.sqrt(in_mul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(self.in_mul, self.out_mul)
        return self._alpha * (x @ w) + self.bias


class NonLinearBiasReadout(nn.Module):
    """Gated non-linear scalar readout to per-atom energy (single head).

    Args:
        irreps_in: Input node feature irreps (last-layer product output).
        mlp_dim: Hidden scalar width (``MLP_irreps`` count), e.g. 16.
        out_mul: Output scalar multiplicity. Energy heads use 1; Polar's
            Fukui source map uses 2.
    """

    def __init__(
        self,
        *,
        irreps_in: str,
        mlp_dim: int = 16,
        out_mul: int = 1,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.linear_1 = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self._act_cst = normalize2mom(F.silu)
        self.linear_mid = _ScalarO3Linear(mlp_dim, mlp_dim, dtype=dtype)
        self.linear_2 = _ScalarO3Linear(mlp_dim, out_mul, dtype=dtype)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * self._act_cst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-atom scalar energy.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom energy ``(N, 1)``.
        """
        x = self._act(self.linear_1(x))
        x = self._act(self.linear_mid(x))
        return self.linear_2(x)


class LinearReadout(nn.Module):
    """Equivariant linear projection of node features to a per-atom scalar.

    MACE's ``LinearReadoutBlock``: one ``cuet.Linear`` onto ``1x0e``. Used for
    every interaction layer except the last, where the non-linear readout
    (:class:`NonLinearReadout`) takes over.

    Args:
        irreps_in: Input node feature irreps, e.g. ``"128x0e+128x1o"``.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, *, irreps_in: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project node features to a per-atom scalar.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom scalar ``(N, 1)``.
        """
        return self.linear(x)


class NonLinearReadout(nn.Module):
    """Bias-free gated non-linear scalar readout (MACE ``NonLinearReadoutBlock``).

    ``Linear → c·SiLU → Linear``, both equivariant ``cuet.Linear`` layers with no
    bias; ``c`` is the moment normalisation constant (``normalize2mom``). This is the readout on
    the **last** interaction layer of the MACE-MP / MatPES foundation models.

    Distinct from :class:`NonLinearBiasReadout`, which is the OMOL variant with
    biases and an extra middle layer — do not substitute one for the other, the
    weight layouts differ.

    Args:
        irreps_in: Input node feature irreps (last-layer product output).
        mlp_dim: Hidden scalar width (MACE's ``MLP_irreps``), e.g. 16.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, *, irreps_in: str, mlp_dim: int = 16, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear_1 = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self.linear_2 = cueq.Linear(
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self._act_cst = normalize2mom(F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-atom scalar energy.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom energy ``(N, 1)``.
        """
        return self.linear_2(F.silu(self.linear_1(x)) * self._act_cst)
