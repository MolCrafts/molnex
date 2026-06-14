"""Edge-centric energy readout for Allegro-style encoders.

Faithful port of ``mir-group/allegro::edgewise.py::EdgewiseReduce`` plus the
``edge_eng_mlp`` config defaults from ``mir-group/allegro::minimal.yaml``
(``edge_eng_mlp_nonlinearity=null``, ``edge_eng_mlp_latent_dimensions=[128]``,
``bias=False``). Pipeline:

    edge_features (E, F·(L+1))
        —— Linear(F·(L+1), H) → Linear(H, 1)            (purely linear, no activation)
        —— / √⟨|N|⟩                                       (dataset-wide aggregate norm)
        —— scatter_add by source                          (E_i)
        —— / √2                                           (double-counted derivatives)
        —— scatter_add by graph                           (E_b)

The readout MLP is intentionally fully linear: two ``Linear`` layers in series
with ``nonlinearity=None`` and ``bias=False``. Inserting an activation here
would deviate from the upstream config and is forbidden by §2 of
``src/molzoo/specs/allegro.md`` (row "Pair-energy readout MLP").

There is no per-source ``1/√|N(i)|`` fallback and no per-layer-MLP variant:
``avg_num_neighbors`` is a required dataset statistic, computed once in the
data pipeline. The encoder writes a flat DenseNet stack ``(E, F·(L+1))`` and
this head consumes it as a single feature vector — exactly what
``mir-group/allegro::edge_readout`` does.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
    https://arxiv.org/abs/2204.05249

    mir-group/allegro/nn/edgewise.py::EdgewiseReduce
    mir-group/allegro/configs/minimal.yaml (edge_eng_mlp_*)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from molrep.embedding.mlp import ScalarMLPFunction


class EdgeEnergyHead(nn.Module):
    """Map per-edge scalar features to total per-graph energy.

    Args:
        input_dim: Per-edge feature dim. For an Allegro encoder this is the
            full DenseNet stack width ``F · (L+1)`` (= ``encoder.output_dim``).
        hidden_dim: Readout hidden dim ``H``. Paper QM9 / minimal.yaml: ``128``.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for aggregate normalization.
            **Required** (no per-source fallback); compute once in the data
            pipeline (e.g. ``molix.data.tasks.AvgNumNeighborsStat``).
        out_key: TensorDict key under which to write the per-graph energy.

    Forward output:
        ``{out_key: (B,)}`` with the per-graph total energy tensor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        avg_num_neighbors: float,
        out_key: str = "energy",
    ):
        super().__init__()
        if avg_num_neighbors <= 0.0:
            raise ValueError(f"avg_num_neighbors must be positive; got {avg_num_neighbors}")
        # Two-layer purely-linear readout, matching mir-group/allegro::minimal.yaml
        # (``edge_eng_mlp_nonlinearity=null``, ``edge_eng_mlp_latent_dimensions=[128]``).
        # ``ScalarMLPFunction(nonlinearity=None)`` skips activation insertion and
        # uses ``gain=1`` initialisation, identical to upstream.
        self.mlp = ScalarMLPFunction(
            input_dim=input_dim,
            output_dim=1,
            hidden_layers_depth=1,
            hidden_layers_width=hidden_dim,
            nonlinearity=None,
            bias=False,
        )
        self.avg_num_neighbors = float(avg_num_neighbors)
        self._inv_sqrt_avg_n = 1.0 / math.sqrt(self.avg_num_neighbors)
        self._inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.out_key = out_key

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Aggregate edge features to per-graph energies.

        Args:
            batch: ``TensorDict`` with ``edges.edge_features`` of shape
                ``(E, F·(L+1))`` (the Allegro DenseNet stack), plus
                ``edges.edge_index``, ``atoms.Z``, ``atoms.batch``, and
                a ``graphs`` subdict that carries the per-graph batch size.

        Returns:
            ``{self.out_key: Tensor}`` with per-graph energy of shape ``(B,)``.
        """
        edge_features = batch["edges", "edge_features"]
        edge_e = self.mlp(edge_features).squeeze(-1) * self._inv_sqrt_avg_n  # (E,)

        edge_index = batch["edges", "edge_index"]
        n_nodes = int(batch["atoms", "Z"].shape[0])
        node_e = torch.zeros(n_nodes, dtype=edge_e.dtype, device=edge_e.device)
        node_e.scatter_add_(0, edge_index[:, 0], edge_e)  # (N,)
        # /√2 — mirrors mir-group/allegro::EdgewiseReduce. Per-source aggregate
        # is double-counted relative to a single-direction sum because dE/dr_i
        # picks up contributions from both dE/dr_ij and dE/dr_ji on every
        # undirected pair.
        node_e = node_e * self._inv_sqrt_2

        atom_batch = batch["atoms", "batch"]
        n_graphs = batch["graphs"].batch_size[0]
        energy = torch.zeros(n_graphs, dtype=node_e.dtype, device=node_e.device)
        energy.scatter_add_(0, atom_batch, node_e)  # (B,)

        return {self.out_key: energy}
