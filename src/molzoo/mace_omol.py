"""MACE-OMOL foundation model, assembled from molrep/molpot building blocks.

A faithful, native reimplementation of the official ``MACE-omol-0`` model
(``ScaleShiftMACE`` with charge/spin conditioning and non-linear residual
interactions) on the cuEquivariance backend. Every sub-block lives in
``molrep`` / ``molpot``; this module only wires them into the energy/force
forward of MACE's ``ScaleShiftMACE``.

Use :func:`load_omol_state_dict` to import weights from an official MACE-OMOL
checkpoint that has been converted to cueq layout via
``mace.cli.convert_e3nn_cueq``.

Reference:
    Batatia et al. "MACE" NeurIPS 2022; OMol25 foundation model.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix import config
from molpot.derivation.force import ForceDerivation
from molpot.heads.energy import AtomicReferenceEnergy
from molpot.heads.rescale import GlobalRescale
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.node import JointFeatureEmbedding, JointFeatureSpec
from molrep.embedding.radial import BesselRBF
from molrep.interaction.product_basis import EquivariantProductBasis
from molrep.interaction.residual import ResidualInteraction
from molrep.readout.scalar import NonLinearBiasReadout


def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size, *src.shape[1:]))
    out.index_add_(0, index, src)
    return out


class MACEOMol(nn.Module):
    """Native MACE-OMOL energy/force model (cuEquivariance)."""

    def __init__(
        self,
        *,
        atomic_numbers: list[int],
        atomic_energies: torch.Tensor,
        r_max: float = 6.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        l_max: int = 3,
        num_features: int = 1024,
        num_interactions: int = 3,
        correlation: int = 2,
        mlp_dim: int = 16,
        charge_classes: int = 201,
        charge_offset: int = 100,
        spin_classes: int = 101,
        spin_offset: int = 0,
        scale: float = 1.0,
        shift: float = 0.0,
        group=None,
    ) -> None:
        super().__init__()
        ftype = config.ftype
        self._group = group
        n_el = len(atomic_numbers)
        self.register_buffer(
            "z_table", torch.tensor(atomic_numbers, dtype=torch.long), persistent=True
        )
        self.num_interactions = num_interactions

        sh = "+".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1))
        feat0 = f"{num_features}x0e"
        target = "+".join(
            f"{num_features}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1)
        )

        self.node_embedding = cuet.Linear(
            cue.Irreps("O3", f"{n_el}x0e"),
            cue.Irreps("O3", feat0),
            layout=cue.ir_mul,
            dtype=ftype,
        )
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)
        self.bessel = BesselRBF(
            r_cut=r_max, num_radial=num_bessel, normalize=False, eps=0.0, trainable=True
        )
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max, exponent=num_polynomial_cutoff)

        self.joint_embedding = JointFeatureEmbedding(
            feature_specs=[
                JointFeatureSpec(
                    name="total_spin",
                    kind="categorical",
                    emb_dim=num_features,
                    num_classes=spin_classes,
                    per="graph",
                    offset=spin_offset,
                ),
                JointFeatureSpec(
                    name="total_charge",
                    kind="categorical",
                    emb_dim=num_features,
                    num_classes=charge_classes,
                    per="graph",
                    offset=charge_offset,
                ),
            ],
            out_dim=num_features,
        )
        self.embedding_readout = cuet.Linear(
            cue.Irreps("O3", feat0),
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=ftype,
        )
        self.atomic_energies = AtomicReferenceEnergy(
            atomic_energies=atomic_energies,
            atomic_numbers=atomic_numbers,
        )

        # per-layer irreps (mirror official ScaleShiftMACE hidden_irreps schedule)
        hidden_full = target.replace(f"+{num_features}x{l_max}{'e' if l_max % 2 == 0 else 'o'}", "")
        edge0 = feat0
        edge_mid = "+".join(
            f"128x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max)
        )  # 128x0e+128x1o+128x2e
        node_in = [feat0, hidden_full, hidden_full]
        edge_irr = [edge0, edge_mid, edge_mid]
        hidden_sched = [hidden_full, hidden_full, feat0]

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        for i in range(num_interactions):
            self.interactions.append(
                ResidualInteraction(
                    node_attrs_irreps=f"{n_el}x0e",
                    node_feats_irreps=node_in[i],
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{num_bessel}x0e",
                    edge_irreps=edge_irr[i],
                    target_irreps=target,
                    hidden_irreps=hidden_sched[i],
                    radial_mlp=[128, 128, 128],
                    group=group,
                )
            )
            self.products.append(
                EquivariantProductBasis(
                    node_feats_irreps=target,
                    target_irreps=hidden_sched[i],
                    correlation=correlation,
                    num_elements=1,
                    use_sc=True,
                    group=group,
                )
            )
        self.readout = NonLinearBiasReadout(irreps_in=feat0, mlp_dim=mlp_dim)
        self.scale_shift = GlobalRescale(scale=scale, shift=shift)
        self.force_derivation = ForceDerivation()

    def energy_forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        shifts: torch.Tensor | None = None,
        compute_forces: bool = True,
    ) -> dict:
        """Return total energy and forces.

        Args:
            positions: ``(N, 3)``.
            Z: atomic numbers ``(N,)``.
            edge_index: ``(2, E)`` sender/receiver.
            batch: graph index per atom ``(N,)``.
            total_charge / total_spin: per-graph ``(B,)``.
            shifts: optional PBC shift vectors ``(E, 3)``.
            compute_forces: whether to compute ``-dE/dx``.
        """
        positions = positions.clone().requires_grad_(compute_forces)
        total_energy = self._compute_energy(
            positions, Z, edge_index, batch, total_charge, total_spin, shifts
        )
        out = {"energy": total_energy}
        if compute_forces:
            grad = torch.autograd.grad(total_energy.sum(), positions, create_graph=self.training)[0]
            out["forces"] = -grad
        return out

    def _compute_energy(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        shifts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph total energy ``(B,)`` as a pure function of ``positions``.

        Shared core of :meth:`energy_forces` and :meth:`forward`. Recomputes all
        position-derived geometry internally so it can be differentiated with
        either ``torch.autograd.grad`` or ``torch.func.grad`` (ForceDerivation).

        Args:
            positions: ``(N, 3)``.
            Z: atomic numbers ``(N,)``.
            edge_index: ``(2, E)`` sender/receiver.
            batch: graph index per atom ``(N,)``.
            total_charge / total_spin: per-graph ``(B,)``.
            shifts: optional PBC shift vectors ``(E, 3)``.
        """
        num_nodes = positions.shape[0]
        num_graphs = int(batch.max().item()) + 1

        sender, receiver = edge_index[0], edge_index[1]
        vectors = positions[receiver] - positions[sender]
        if shifts is not None:
            vectors = vectors + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        # one-hot node attrs over the element table
        z_index = torch.searchsorted(self.z_table, Z)
        node_attrs = torch.zeros(
            num_nodes, self.z_table.numel(), dtype=positions.dtype, device=positions.device
        )
        node_attrs[torch.arange(num_nodes), z_index] = 1.0

        node_e0 = self.atomic_energies(Z)
        e0 = _scatter_sum(node_e0, batch, num_graphs)

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.bessel(lengths.squeeze(-1))
        cutoff = self.cutoff_fn(lengths.squeeze(-1)).unsqueeze(-1)

        node_feats = node_feats + self.joint_embedding(
            batch, total_spin=total_spin, total_charge=total_charge
        )
        emb_node_e = self.embedding_readout(node_feats).squeeze(-1)
        e0 = e0 + _scatter_sum(emb_node_e, batch, num_graphs)

        feats_last = None
        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_attrs, node_feats, edge_attrs, edge_feats, edge_index, cutoff
            )
            node_feats = self.products[i](node_feats, sc, node_attrs)
            feats_last = node_feats

        node_es = self.readout(feats_last).squeeze(-1)
        node_inter = self.scale_shift(node_es)
        inter_e = _scatter_sum(node_inter, batch, num_graphs)
        return e0 + inter_e

    def forward(self, td: TensorDict) -> TensorDict:
        """Run MACE-OMOL on a post-collate batch, writing energy and forces.

        Reads ``atoms.{Z,pos,batch}``, ``edges.edge_index`` (``(E, 2)`` with
        ``[:,0]`` source / ``[:,1]`` target per the molnex edge convention), and
        per-graph ``graphs.{total_charge,total_spin}`` (defaulting to a neutral
        singlet when absent). Forces are obtained through
        :class:`molpot.derivation.ForceDerivation` (``F = -∂E/∂pos`` via
        ``torch.func.grad``). Writes ``graphs.energy`` ``(B,)`` and
        ``atoms.forces`` ``(N, 3)`` back into ``td`` and returns it.

        Args:
            td: post-collate ``TensorDict`` with ``atoms`` / ``edges`` (and
                optionally ``graphs``) sub-dicts.

        Returns:
            The same ``td`` with ``graphs.energy`` and ``atoms.forces`` added.
        """
        Z = td["atoms", "Z"]
        positions = td["atoms", "pos"]
        batch = td["atoms", "batch"]
        # molnex edge_index is (E, 2) [source, target]; the core wants (2, E).
        edge_index = td["edges", "edge_index"].t().contiguous()
        num_graphs = int(batch.max().item()) + 1

        nested = td.keys(include_nested=True)
        if ("graphs", "total_charge") in nested:
            total_charge = td["graphs", "total_charge"]
        else:
            total_charge = torch.zeros(num_graphs, dtype=torch.long, device=Z.device)
        if ("graphs", "total_spin") in nested:
            total_spin = td["graphs", "total_spin"]
        else:
            total_spin = torch.zeros(num_graphs, dtype=torch.long, device=Z.device)

        energy = self._compute_energy(positions, Z, edge_index, batch, total_charge, total_spin)

        def energy_fn(pos: torch.Tensor) -> torch.Tensor:
            return self._compute_energy(pos, Z, edge_index, batch, total_charge, total_spin).sum()

        forces = self.force_derivation(energy_fn, positions)

        if "graphs" not in td.keys():
            td["graphs"] = TensorDict({}, batch_size=[num_graphs])
        td["graphs", "energy"] = energy
        td["atoms", "forces"] = forces
        return td


def load_omol_state_dict(model: MACEOMol, cueq_state: dict) -> tuple[list, list]:
    """Load a cueq-converted MACE-OMOL ``state_dict`` into :class:`MACEOMol`.

    Maps the official cueq key names onto this model's modules (direct copy);
    returns ``(missing_learnable, unexpected)`` for inspection.
    """
    remap = {}
    for k, v in cueq_state.items():
        nk = k
        if k.startswith("node_embedding.linear."):
            nk = "node_embedding." + k.split("node_embedding.linear.")[1]
        elif k.startswith("embedding_readout.linear."):
            nk = "embedding_readout." + k.split("embedding_readout.linear.")[1]
        elif k.startswith("readouts.0."):
            nk = "readout." + k.split("readouts.0.")[1]
        elif k.startswith("atomic_energies_fn."):
            continue  # handled at construction (Z-indexed)
        remap[nk] = v
    missing, unexpected = model.load_state_dict(remap, strict=False)
    learnable_leaf = (".weight", ".bias")
    learnable_scalar = ("alpha", "beta", "scale", "shift")
    miss_learn = [
        m
        for m in missing
        if (m.endswith(learnable_leaf) or m.split(".")[-1] in learnable_scalar)
        and not m.startswith("atomic_energies")
        and "scale_shift" not in m
    ]
    return miss_learn, unexpected
