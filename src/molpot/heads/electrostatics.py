"""LES-side prediction heads: atomic hardness κ and polarizability α.

Two heads consumed by :class:`molpot.potentials.EwaldMultipoleEnergy`'s
non-self-consistent linear-response branch:

* :class:`HardnessHead` — l=0 scalar MLP with softplus activation,
  producing strictly-positive ``κ_i ∈ ℝ⁺`` per atom.
* :class:`PolarizabilityHead` — l=0 isotropic ``α_iso`` (default) or
  l=0 + l=2 anisotropic ``α (N, 3, 3)`` decomposed as
  ``α_iso · I + α_deviator`` where ``α_deviator`` is a symmetric
  traceless 3×3 tensor reconstructed from the encoder's ``2e`` block.

Both heads are intentionally tensor-in / tensor-out (not TensorDict-in)
so they can be unit-tested with synthetic features and slotted into a
composer pipeline at the call site rather than by reading internal
batch keys. This mirrors the molrep / molzoo encoder convention.
"""

from __future__ import annotations

from typing import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from molix import config

# ---------------------------------------------------------------------------
# Shared helper — mirrors PermMultipoleHead._equivariant_moment_readout, kept
# private to this module to avoid coupling to multipole.py's internals.
# ---------------------------------------------------------------------------


def _scalar_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """``[Linear → SiLU] × len(hidden) → Linear`` (no activation on final layer)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, dtype=config.ftype))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=config.ftype))
    return nn.Sequential(*layers)


def _find_irrep_offset(tensor_irreps: cue.Irreps, target_ir: cue.Irrep) -> int:
    offset = 0
    for mul, ir in tensor_irreps:
        if ir == target_ir:
            return offset
        offset += mul * ir.dim
    raise ValueError(f"tensor_irreps must contain a {target_ir} segment; got {tensor_irreps}.")


# 5 → 3×3 conversion of a real spherical-harmonic l=2 channel into a
# symmetric traceless Cartesian tensor. Conventions match cuequivariance's
# ``2e`` real-spherical ordering ``(m=-2, m=-1, m=0, m=+1, m=+2)`` =
# ``(xy, yz, (3z²-r²)/(2√3), xz, (x²-y²)/2)`` in the real basis with
# unit-norm components. Tracelessness is structural — the construction
# can never produce a non-zero trace by design (verified to <1e-12 in
# float64 in the unit tests).
def _l2_to_cartesian(t: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 5)`` real-spherical l=2 → ``(N, 3, 3)`` symmetric traceless.

    Args:
        t: ``(N, 5)`` real-spherical-harmonic ``2e`` components, ordered
            ``[m=-2, m=-1, m=0, m=+1, m=+2]`` (cuequivariance default).

    Returns:
        ``(N, 3, 3)`` symmetric traceless tensor. Tracelessness holds
        exactly in exact arithmetic (1/√3 cancels) and to machine
        precision in float64.
    """
    n = t.shape[0]
    out = torch.zeros(n, 3, 3, dtype=t.dtype, device=t.device)
    out[:, 0, 1] = out[:, 1, 0] = t[:, 0]  # m=-2 → xy
    out[:, 1, 2] = out[:, 2, 1] = t[:, 1]  # m=-1 → yz
    out[:, 0, 2] = out[:, 2, 0] = t[:, 3]  # m=+1 → xz
    inv_sqrt3 = 1.0 / (3.0**0.5)
    out[:, 0, 0] = -inv_sqrt3 * t[:, 2] + t[:, 4]  # 1/3·(3·xx-r²) = xx - tr/3 part
    out[:, 1, 1] = -inv_sqrt3 * t[:, 2] - t[:, 4]
    out[:, 2, 2] = 2.0 * inv_sqrt3 * t[:, 2]
    return out


def _equivariant_l_readout(
    tensor_feats: torch.Tensor,
    scalar_feats: torch.Tensor,
    src: torch.Tensor,
    n_nodes: int,
    *,
    block_offset: int,
    block_size: int,
    out_dim: int,
    mul: int,
    scalar_proj: nn.Module,
    collapse: nn.Module,
    avg_num_neighbors: float | None,
) -> torch.Tensor:
    """PaiNN-style scalar-gated lℓ readout — see PermMultipoleHead's helper.

    Slices the ``lℓ`` block of the encoder's tensor track, gates per
    channel by an l=0 scalar projection of ``scalar_feats``, collapses
    ``u·lℓ → 1·lℓ`` via ``cuet.Linear``, then scatter-sums to source
    atoms.  Strictly SO(3)-equivariant under Wigner ``Dˡ(R)``.
    """
    e = tensor_feats.shape[0]
    v_flat = tensor_feats[:, block_offset : block_offset + block_size]
    v_l = v_flat.reshape(e, out_dim, mul).transpose(1, 2)  # (E, u, 2ℓ+1)
    gate = scalar_proj(scalar_feats)  # (E, u)
    gated = gate.unsqueeze(-1) * v_l  # (E, u, 2ℓ+1)
    gated_ir_mul = gated.transpose(1, 2).reshape(e, block_size)
    edge_out = collapse(gated_ir_mul)  # (E, 2ℓ+1)

    atom_out = torch.zeros(n_nodes, out_dim, dtype=edge_out.dtype, device=edge_out.device)
    atom_out.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_out), edge_out)
    if avg_num_neighbors is not None:
        atom_out = atom_out / (avg_num_neighbors**0.5)
    return atom_out


# ---------------------------------------------------------------------------
# Pydantic configs
# ---------------------------------------------------------------------------


class HardnessHeadSpec(BaseModel):
    """Configuration snapshot for :class:`HardnessHead`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_dim: int = Field(..., gt=0)
    hidden_dim: int = Field(default=128, gt=0)
    softplus_beta: float = Field(default=1.0, gt=0.0)


class PolarizabilityHeadSpec(BaseModel):
    """Configuration snapshot for :class:`PolarizabilityHead`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_dim: int = Field(..., gt=0)
    hidden_dim: int = Field(default=128, gt=0)
    anisotropic: bool = False
    softplus_beta: float = Field(default=1.0, gt=0.0)


# ---------------------------------------------------------------------------
# HardnessHead
# ---------------------------------------------------------------------------


class HardnessHead(nn.Module):
    """l=0 scalar readout of atomic hardness ``κ_i ∈ ℝ⁺``.

    Tensor-in / tensor-out: takes pooled atom features ``(N, F)`` and
    returns ``(N,)`` strictly-positive hardness values via softplus.
    ``κ`` is consumed by
    :class:`molpot.potentials.EwaldMultipoleEnergy` to produce the
    induced charge ``q_induced = -κ · Φ`` and its energy contribution
    ``U_iq = ½ Φ · q_induced``.

    Args:
        input_dim: Per-atom feature dim ``F``.
        hidden_dim: Hidden width of the scalar MLP. Default 128.
        softplus_beta: ``β`` of ``softplus(βx)/β``; large β ⇒ closer to
            ReLU. Default 1.0.
    """

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, softplus_beta: float = 1.0):
        super().__init__()
        self.config = HardnessHeadSpec(
            input_dim=input_dim, hidden_dim=hidden_dim, softplus_beta=softplus_beta
        )
        self.mlp = _scalar_mlp(input_dim, [hidden_dim], 1)
        self.softplus_beta = softplus_beta

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Compute κ from per-atom features.

        Args:
            atom_features: ``(N, F)`` per-atom feature tensor.

        Returns:
            ``(N,)`` strictly-positive κ.
        """
        raw = self.mlp(atom_features).squeeze(-1)
        return F.softplus(raw, beta=self.softplus_beta)


# ---------------------------------------------------------------------------
# PolarizabilityHead
# ---------------------------------------------------------------------------


class PolarizabilityHead(nn.Module):
    """Atomic polarizability ``α``: isotropic scalar or anisotropic 3×3.

    The isotropic path emits ``α_iso (N,)`` via softplus on a scalar MLP
    (mirrors :class:`HardnessHead`). The anisotropic path additionally
    consumes the encoder's ``2e`` tensor track to produce a deviator
    ``α_dev (N, 3, 3)`` (symmetric traceless), and returns the full
    ``α (N, 3, 3) = α_iso · I + α_dev``.

    The trace of ``α_dev`` is exactly zero by construction (the
    ``5 → 3×3`` map :func:`_l2_to_cartesian` is structurally traceless),
    so ``trace(α) = 3 · α_iso > 0`` matches the physical convention
    that the mean polarizability is positive.

    Args:
        input_dim: Per-atom (or per-edge for the l=2 path) feature dim.
        hidden_dim: Hidden width of the scalar MLP. Default 128.
        anisotropic: If True, also produce the l=2 deviator and return
            a 3×3 tensor. Requires ``tensor_irreps`` and the l=2 inputs
            at forward time. Default False.
        softplus_beta: ``β`` for the isotropic head's softplus.
        tensor_irreps: cuequivariance Irreps of the encoder's tensor
            track (only used when ``anisotropic=True``).
        avg_num_neighbors: Optional ⟨|N(i)|⟩ for the edge→atom pool
            normalisation in the l=2 readout.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        anisotropic: bool = False,
        softplus_beta: float = 1.0,
        tensor_irreps: cue.Irreps | None = None,
        avg_num_neighbors: float | None = None,
    ):
        super().__init__()
        self.config = PolarizabilityHeadSpec(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            anisotropic=anisotropic,
            softplus_beta=softplus_beta,
        )
        self.iso_mlp = _scalar_mlp(input_dim, [hidden_dim], 1)
        self.softplus_beta = softplus_beta
        self.anisotropic = anisotropic
        self.avg_num_neighbors = avg_num_neighbors

        if anisotropic:
            if tensor_irreps is None:
                raise ValueError(
                    "anisotropic=True requires tensor_irreps (the encoder's "
                    "tensor-track irreps with l_max ≥ 2)."
                )
            muls = tensor_irreps.muls
            if len(set(muls)) != 1:
                raise ValueError(
                    f"PolarizabilityHead requires uniform multiplicity in "
                    f"tensor_irreps; got muls={muls}."
                )
            self._u = int(muls[0])
            G = tensor_irreps.irrep_class
            try:
                self._dev_offset = _find_irrep_offset(tensor_irreps, G(2, +1))
            except ValueError as e:
                raise ValueError(
                    f"anisotropic=True needs a 2e segment in tensor_irreps; "
                    f"build the encoder with l_max>=2. ({e})"
                ) from None
            self._dev_size = 5 * self._u
            self.dev_proj = nn.Linear(input_dim, self._u, dtype=config.ftype)
            self.dev_collapse = cuet.Linear(
                irreps_in=cue.Irreps(G, [(self._u, "2e")]),
                irreps_out=cue.Irreps(G, [(1, "2e")]),
                layout=cue.ir_mul,
            )

    def forward(
        self,
        atom_features: torch.Tensor,
        *,
        tensor_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        scalar_edge_features: torch.Tensor | None = None,
        n_nodes: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute α.

        Isotropic path (``anisotropic=False``): returns ``(N,)`` α_iso.

        Anisotropic path (``anisotropic=True``): returns a dict with
        ``"alpha_iso"`` ``(N,)``, ``"alpha_dev"`` ``(N, 3, 3)``,
        ``"alpha"`` ``(N, 3, 3) = α_iso · I + α_dev``.

        Args:
            atom_features: ``(N, F)`` pooled atom features for the l=0 head.
            tensor_features: ``(E, irreps_dim)`` encoder tensor track,
                required when ``anisotropic=True``.
            edge_index: ``(E, 2)`` int tensor with source-target edge pairs,
                required when ``anisotropic=True``.
            scalar_edge_features: ``(E, F)`` per-edge scalar features for
                the l=2 readout's gate; required when ``anisotropic=True``.
            n_nodes: number of atoms ``N``; required when
                ``anisotropic=True`` (for the scatter destination size).
        """
        raw_iso = self.iso_mlp(atom_features).squeeze(-1)
        alpha_iso = F.softplus(raw_iso, beta=self.softplus_beta)

        if not self.anisotropic:
            return alpha_iso

        if (
            tensor_features is None
            or edge_index is None
            or scalar_edge_features is None
            or n_nodes is None
        ):
            raise ValueError(
                "anisotropic=True forward requires tensor_features, edge_index, "
                "scalar_edge_features, n_nodes."
            )

        src = edge_index[:, 0]
        dev_5 = _equivariant_l_readout(
            tensor_features,
            scalar_edge_features,
            src,
            n_nodes,
            block_offset=self._dev_offset,
            block_size=self._dev_size,
            out_dim=5,
            mul=self._u,
            scalar_proj=self.dev_proj,
            collapse=self.dev_collapse,
            avg_num_neighbors=self.avg_num_neighbors,
        )  # (N, 5)
        alpha_dev = _l2_to_cartesian(dev_5)  # (N, 3, 3) symmetric traceless

        eye3 = torch.eye(3, dtype=alpha_iso.dtype, device=alpha_iso.device)
        alpha = alpha_iso[:, None, None] * eye3[None, :, :] + alpha_dev

        return {"alpha_iso": alpha_iso, "alpha_dev": alpha_dev, "alpha": alpha}
