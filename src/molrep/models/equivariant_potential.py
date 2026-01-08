"""Equivariant potential network using cuequivariance_torch operators."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
import cuequivariance_torch as cue

from molpot.feats.cutoff import CosineCutoff
from molpot.feats.rbf import GaussianRBF


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter sum operation for aggregating messages."""
    if dim_size == 0:
        return src.new_zeros((0,) + src.shape[1:])
    if src.dim() == 1:
        src = src.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    if squeeze:
        out = out.squeeze(-1)
    return out


class AtomTypeEmbedding(nn.Module):
    """Map atomic numbers to scalar features."""

    def __init__(self, num_types: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_types, hidden_dim, padding_idx=padding_idx)

    def forward(self, td: TensorDict) -> TensorDict:
        td["atoms", "h"] = self.embedding(td["atoms", "Z"])
        return td


class RadialBasisEncoder(nn.Module):
    """Compute radial basis features from neighbor distances."""

    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)

    def forward(self, td: TensorDict) -> TensorDict:
        td["pairs", "rbf"] = self.rbf(td["pairs", "dist"])
        return td


class SphericalHarmonicsEncoder(nn.Module):
    """Compute spherical harmonics from neighbor displacement vectors using cuEquivariance."""

    def __init__(self, lmax: int):
        super().__init__()
        # cuEquivariance SphericalHarmonics takes ls (list of degrees), not lmax
        self.sh = cue.SphericalHarmonics(ls=list(range(lmax + 1)))

    def forward(self, td: TensorDict) -> TensorDict:
        td["pairs", "sh"] = self.sh(td["pairs", "diff"])
        return td


class EdgeFeatureAssembler(nn.Module):
    """Combine radial basis with cutoff to build scalar edge features."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = CosineCutoff(cutoff=cutoff)

    def forward(self, td: TensorDict) -> TensorDict:
        dist = td["pairs", "dist"]
        rbf = td["pairs", "rbf"]
        cutoff = self.cutoff(dist).unsqueeze(-1)
        td["pairs", "edge_feat"] = rbf * cutoff
        return td


class EquivariantInteractionBlock(nn.Module):
    """One equivariant message-passing step using cuequivariance_torch ops."""

    def __init__(
        self,
        hidden_dim: int,
        equivariant_dim: int,
        edge_feat_dim: int,
        tensor_product_op: nn.Module,
        activation: str = "silu",
    ):
        super().__init__()
        self.tensor_product = tensor_product_op
        self.eq_input = nn.Linear(hidden_dim, equivariant_dim)
        self.eq_gate = nn.Linear(hidden_dim, equivariant_dim)

        if activation == "silu":
            act = nn.SiLU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_feat_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, td: TensorDict) -> TensorDict:
        h = td["atoms", "h"]
        h_eq = td["atoms", "h_eq"]
        edge_feat = td["pairs", "edge_feat"]
        sh = td["pairs", "sh"]
        edge_i = td["pairs", "i"].long()
        edge_j = td["pairs", "j"].long()

        h_j = h.index_select(0, edge_j)
        scalar_msg = self.scalar_mlp(torch.cat([h_j, edge_feat], dim=-1))
        agg_scalar = _scatter_sum(scalar_msg, edge_i, h.size(0))
        h = h + self.scalar_update(agg_scalar)

        eq_in = self.eq_input(h_j)
        eq_msg = self.tensor_product(eq_in, sh)
        agg_eq = _scatter_sum(eq_msg, edge_i, h.size(0))
        gate = torch.sigmoid(self.eq_gate(h))
        h_eq = h_eq + agg_eq * gate

        td["atoms", "h"] = h
        td["atoms", "h_eq"] = h_eq
        return td


class EnergyReadout(nn.Module):
    """Predict molecular energy from scalar atom features."""

    def __init__(self, hidden_dim: int, energy_shift: float = 0.0):
        super().__init__()
        self.energy_proj = nn.Linear(hidden_dim, 1)
        self.energy_shift = energy_shift

    def forward(self, td: TensorDict) -> TensorDict:
        h = td["atoms", "h"]
        batch = td["graph", "batch"].long()
        atom_energy = self.energy_proj(h).squeeze(-1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        energy = _scatter_sum(atom_energy, batch, num_graphs)
        if self.energy_shift != 0.0:
            energy = energy + self.energy_shift
        td["target", "energy"] = energy
        return td


class EquivariantPotentialNet(TensorDictModuleBase):
    """Equivariant potential network using cuEquivariance operators.
    
    Uses cuequivariance_torch for spherical harmonics and tensor products.
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_dim: int,
        equivariant_dim: int,
        num_blocks: int,
        lmax: int,
        cutoff: float,
        num_rbf: int,
        activation: str = "silu",
        energy_shift: float = 0.0,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        
        self.in_keys = [
            ("atoms", "Z"),
            ("atoms", "xyz"),
            ("graph", "batch"),
            ("pairs", "i"),
            ("pairs", "j"),
            ("pairs", "dist"),
            ("pairs", "diff"),
        ]
        self.out_keys = [("target", "energy")]

        self.atom_embedding = AtomTypeEmbedding(
            num_types=num_atom_types,
            hidden_dim=hidden_dim,
        )
        self.radial_basis = RadialBasisEncoder(
            num_rbf=num_rbf,
            cutoff=cutoff,
        )
        self.spherical_harmonics = SphericalHarmonicsEncoder(lmax=lmax)
        self.edge_features = EdgeFeatureAssembler(cutoff=cutoff)

        # Create tensor product operators internally
        # Build irreps string for spherical harmonics based on lmax
        sh_irreps = " + ".join([f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)])
        
        self.blocks = nn.ModuleList([
            EquivariantInteractionBlock(
                hidden_dim=hidden_dim,
                equivariant_dim=equivariant_dim,
                edge_feat_dim=num_rbf,
                tensor_product_op=cue.TensorProduct(
                    irreps_in1=f"{equivariant_dim}x0e",
                    irreps_in2=sh_irreps,
                    irreps_out=f"{equivariant_dim}x0e",
                ),
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

        self.energy_readout = EnergyReadout(
            hidden_dim=hidden_dim,
            energy_shift=energy_shift,
        )
        self.equivariant_dim = equivariant_dim

    def forward(self, td: TensorDict) -> TensorDict:
        for key in self.in_keys:
            if key not in td:
                raise KeyError(f"Missing required TensorDict key: {key}")

        edge_i = td["pairs", "i"].long()
        edge_j = td["pairs", "j"].long()
        xyz = td["atoms", "xyz"]
        diff = xyz.index_select(0, edge_i) - xyz.index_select(0, edge_j)
        td["pairs", "diff"] = diff
        td["pairs", "dist"] = torch.linalg.norm(diff, dim=-1)

        td = self.atom_embedding(td)
        td = self.radial_basis(td)
        td = self.spherical_harmonics(td)
        td = self.edge_features(td)

        if ("atoms", "h_eq") not in td:
            h = td["atoms", "h"]
            td["atoms", "h_eq"] = h.new_zeros(h.size(0), self.equivariant_dim)

        for block in self.blocks:
            td = block(td)

        td = self.energy_readout(td)
        return td

    def __repr__(self) -> str:
        return (
            f"EquivariantPotentialNet(hidden_dim={self.blocks[0].eq_input.in_features}, "
            f"equivariant_dim={self.equivariant_dim}, "
            f"num_blocks={len(self.blocks)})"
        )
