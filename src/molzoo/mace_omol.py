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
from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molpot.derivation.force import autograd_forces_from_energy
from molpot.heads.energy import AtomicReferenceEnergy
from molpot.heads.rescale import GlobalRescale
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.node import JointFeatureEmbedding, JointFeatureSpec
from molrep.embedding.radial import BesselRBF
from molrep.interaction.product_basis import EquivariantProductBasis
from molrep.interaction.residual import ResidualInteraction
from molrep.readout.scalar import NonLinearBiasReadout
from molzoo.mace.checkpoint import OMOL_REMAP


class MACEOMol(nn.Module):
    """Native MACE-OMOL energy/force model (cuEquivariance).

    Args:
        atomic_numbers: Element table (z-table) in checkpoint order.
        atomic_energies: Per-element reference energies ``E0``, same order.
        r_max: Radial cutoff in Angstrom.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p``.
        l_max: Maximum spherical-harmonics order.
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        mlp_dim: Hidden width of the final non-linear readout.
        edge_channels: Per-``l`` channel count of the mid-layer edge irreps and
            the radial-MLP hidden width (128 in the shipped OMOL checkpoints —
            a deliberate bottleneck below ``num_features``).
        charge_classes: Embedding rows for the total-charge conditioning.
        charge_offset: Index offset applied to total charge (charge −100 → row 0).
        spin_classes: Embedding rows for the total-spin conditioning.
        spin_offset: Index offset applied to total spin.
        scale: ``atomic_inter_scale``.
        shift: ``atomic_inter_shift``.
    """

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
        edge_channels: int = 128,
        charge_classes: int = 201,
        charge_offset: int = 100,
        spin_classes: int = 101,
        spin_offset: int = 0,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        if num_interactions < 1:
            raise ValueError(f"num_interactions must be at least 1, got {num_interactions}")
        ftype = config.ftype
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

        # Per-layer irreps (mirror official ScaleShiftMACE hidden_irreps
        # schedule), built generically: the first layer reads/emits full-width
        # scalars, mid layers carry the hidden state through the edge_channels
        # bottleneck, and the last layer collapses back to scalars.
        hidden_full = target.replace(f"+{num_features}x{l_max}{'e' if l_max % 2 == 0 else 'o'}", "")
        edge_mid = "+".join(
            f"{edge_channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max)
        )  # e.g. 128x0e+128x1o+128x2e
        node_in = [feat0] + [hidden_full] * (num_interactions - 1)
        edge_irr = [feat0] + [edge_mid] * (num_interactions - 1)
        hidden_sched = [hidden_full] * (num_interactions - 1) + [feat0]

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
                    radial_mlp=[edge_channels] * 3,
                    # autograd force path → fused cuEq kernels OK
                    use_fallback=False,
                )
            )
            self.products.append(
                EquivariantProductBasis(
                    node_feats_irreps=target,
                    target_irreps=hidden_sched[i],
                    correlation=correlation,
                    num_elements=1,
                    use_sc=True,
                    use_fallback=False,
                )
            )
        self.readout = NonLinearBiasReadout(irreps_in=feat0, mlp_dim=mlp_dim)
        self.scale_shift = GlobalRescale(scale=scale, shift=shift)

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
    ) -> dict[str, torch.Tensor]:
        """Return total energy and forces.

        Args:
            positions: ``(N, 3)``.
            Z: atomic numbers ``(N,)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            batch: graph index per atom ``(N,)``.
            total_charge / total_spin: per-graph ``(B,)``.
            shifts: optional PBC shift vectors ``(E, 3)``.
            compute_forces: whether to compute ``-dE/dx``.
        """
        if not compute_forces:
            return {
                "energy": self._compute_energy(
                    positions.detach(), Z, edge_index, batch, total_charge, total_spin, shifts
                )
            }

        # One forward, then differentiate the graph it built — MACE's own
        # `get_outputs` shape. A closure that re-runs the energy would cost two
        # full forwards per call.
        pos = positions.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self._compute_energy(
                pos, Z, edge_index, batch, total_charge, total_spin, shifts
            )
        return {"energy": energy.detach(), "forces": autograd_forces_from_energy(energy, pos)}

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
        ``torch.autograd.grad`` (:func:`~molpot.derivation.force.autograd_forces_from_energy`).

        Args:
            positions: ``(N, 3)``.
            Z: atomic numbers ``(N,)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            batch: graph index per atom ``(N,)``.
            total_charge / total_spin: per-graph ``(B,)``.
            shifts: optional PBC shift vectors ``(E, 3)``.
        """
        # Derive the graph count from the per-graph ``total_charge`` length (a
        # static shape) rather than ``int(batch.max().item())``: the ``.item()``
        # forces a host sync that breaks the dynamo graph, blocking
        # ``torch.compile(fullgraph=True)`` of the functorch force path.
        num_graphs = total_charge.shape[0]

        source, target = edge_index[:, 0], edge_index[:, 1]
        vectors = positions[target] - positions[source]
        if shifts is not None:
            vectors = vectors + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        # one-hot node attrs over the element table
        z_index = torch.searchsorted(self.z_table, Z.reshape(-1)).to(dtype=torch.long)
        node_attrs = torch.nn.functional.one_hot(z_index, self.z_table.numel()).to(positions.dtype)

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
        :func:`molpot.derivation.force.autograd_forces_from_energy` (``F = -∂E/∂pos``
        via ``torch.autograd.grad`` on the energy graph already built by this
        call — one forward, one backward, as in MACE's ``get_outputs``). Writes
        ``graphs.energy`` ``(B,)`` and ``atoms.forces`` ``(N, 3)`` back into
        ``td`` and returns it.

        Args:
            td: post-collate ``TensorDict`` with ``atoms`` / ``edges`` (and
                optionally ``graphs``) sub-dicts.

        Returns:
            The same ``td`` with ``graphs.energy`` and ``atoms.forces`` added.
        """
        Z = td["atoms", "Z"]
        positions = td["atoms", "pos"]
        batch = td["atoms", "batch"]
        edge_index = td["edges", "edge_index"]  # (E, 2), the core's own convention

        nested = td.keys(include_nested=True)
        if ("graphs", "total_charge") in nested:
            total_charge = td["graphs", "total_charge"]
            num_graphs = int(total_charge.shape[0])
        else:
            # Only the defaulting path pays the host sync; a conditioned batch
            # derives the graph count from a static shape.
            num_graphs = int(batch.max().item()) + 1
            total_charge = torch.zeros(num_graphs, dtype=torch.long, device=Z.device)
        if ("graphs", "total_spin") in nested:
            total_spin = td["graphs", "total_spin"]
        else:
            # OMOL convention: 1 = closed-shell singlet (spin_offset=0 → index 1,
            # a trained row). Spin 0 hits an untrained embedding row → garbage.
            total_spin = torch.ones(num_graphs, dtype=torch.long, device=Z.device)

        needs_leaf = not (positions.requires_grad and positions.is_leaf)
        pos = positions.detach().requires_grad_(True) if needs_leaf else positions
        with torch.enable_grad():
            energy = self._compute_energy(pos, Z, edge_index, batch, total_charge, total_spin)
        forces = autograd_forces_from_energy(energy, pos)

        if "graphs" not in td.keys():
            td["graphs"] = TensorDict({}, batch_size=[num_graphs])
        td["graphs", "energy"] = energy.detach() if needs_leaf else energy
        td["atoms", "forces"] = forces
        return td


def load_omol_state_dict(model: MACEOMol, cueq_state: dict) -> tuple[list[str], list[str]]:
    """Load a cueq-converted MACE-OMOL ``state_dict`` into :class:`MACEOMol`.

    A thin wrapper over :data:`molzoo.mace.checkpoint.OMOL_REMAP`, the
    ``on_unexpected="return"`` preset of
    :class:`~molzoo.mace.checkpoint.CheckpointRemap`. Strict on the silent
    failure modes: a learnable parameter the checkpoint does not fill, or a
    shape mismatch, raises — a silently dropped tensor is exactly how a model
    runs, looks sane, and is quietly wrong. That is the ``bessel.freqs``
    incident: a ``.weight`` / ``.bias`` name-suffix test for "is this tensor
    learnable?" excused an ``nn.Parameter`` ending in neither, so the
    checkpoint's fitted Bessel frequencies were dropped and the analytic ones
    silently stayed; :mod:`molzoo.mace.checkpoint` records it in full.
    Unexpected checkpoint keys are still *returned* rather than raised, because
    the OMOL checkpoint family carries auxiliary heads this port deliberately
    does not model.

    Args:
        model: Target model, constructed with the checkpoint's hyper-parameters.
        cueq_state: ``state_dict`` of the cueq-converted official model.

    Returns:
        ``(missing_non_learnable, unexpected)`` — buffers the checkpoint did
        not provide and checkpoint keys with no home, for inspection. Both
        lists are now sorted (they used to come back in checkpoint order); the
        content is unchanged.

    Raises:
        RuntimeError: If a learnable parameter is left unfilled or a shape
            disagrees.
    """
    return OMOL_REMAP.load(model, cueq_state)
