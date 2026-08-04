"""PiNet property heads (dipole, polarizability) composed with a generic encoder.

These wrap ``molpot.heads``; they live next to the encoder only for import
stability. Planned migration: ``molpot.heads.pinet_*``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.heads import ChargeResponseHead, DipoleHead

from .geometry import edge_bond_diff


def pool_layer(features: torch.Tensor, reduction: str) -> torch.Tensor:
    """Pool the layer axis of ``(N, layers, ...)`` features."""
    if reduction == "mean":
        return features.mean(dim=1)
    if reduction == "sum":
        return features.sum(dim=1)
    if reduction == "last":
        return features[:, -1]
    raise ValueError(f"Unknown reduction {reduction!r}.")


class PiNetDipole(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.DipoleHead`."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        variant: str = "ac_ad",
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        vector_dipole: bool = True,
        charge_neutrality: bool = True,
        regularization: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = DipoleHead(
            node_scalar_dim=input_dim,
            node_vector_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            hidden_dim=hidden_dim,
            variant=variant,
            vector_dipole=vector_dipole,
            charge_neutrality=charge_neutrality,
            regularization=regularization,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = pool_layer(batch["atoms", "node_features"], self.layer_reduction)
        node_vectors = None
        if "p3_features" in batch["atoms"].keys():
            node_vectors = pool_layer(batch["atoms", "p3_features"], self.layer_reduction)
        edge_scalars = None
        edge_index = None
        edge_diff = None
        if self.head.uses_bc and "i1_features" in batch["edges"].keys():
            edge_scalars = pool_layer(batch["edges", "i1_features"], self.layer_reduction)
            edge_index = batch["edges", "edge_index"]
            pos = batch["atoms", "pos"]
            edge_diff = edge_bond_diff(batch["edges"], pos, edge_index)
        edge_vectors = None
        if self.head.uses_bc and "i3_features" in batch["edges"].keys():
            edge_vectors = pool_layer(batch["edges", "i3_features"], self.layer_reduction)
        oxidation = None
        if self.head.uses_os and "oxidation" in batch["atoms"].keys():
            oxidation = batch["atoms", "oxidation"]
        total_charge = None
        if self.head.uses_ac and self.head.charge_neutrality:
            try:
                total_charge = batch["graphs", "total_charge"]
            except KeyError:
                pass
        return self.head(
            pos=batch["atoms", "pos"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            node_scalars=node_scalars,
            node_vectors=node_vectors,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
            edge_index=edge_index,
            edge_diff=edge_diff,
            oxidation=oxidation,
            total_charge=total_charge,
        )


class PiNetPolarizability(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.ChargeResponseHead`."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        atom_types: list[int] | None = None,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = ChargeResponseHead(
            node_scalar_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            atom_types=atom_types,
            variant=variant,
            iso=iso,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            sigma=sigma,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = pool_layer(batch["atoms", "node_features"], self.layer_reduction)
        edge_scalars = pool_layer(batch["edges", "i1_features"], self.layer_reduction)
        edge_vectors = None
        if "i3_features" in batch["edges"].keys():
            edge_vectors = pool_layer(batch["edges", "i3_features"], self.layer_reduction)
        pos = batch["atoms", "pos"]
        edge_index = batch["edges", "edge_index"]
        edge_diff = edge_bond_diff(batch["edges"], pos, edge_index)
        return self.head(
            pos=pos,
            Z=batch["atoms", "Z"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            edge_index=edge_index,
            edge_diff=edge_diff,
            node_scalars=node_scalars,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
        )
