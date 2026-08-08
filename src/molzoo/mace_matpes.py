"""MACE-MatPES foundation model, assembled from molrep/molpot building blocks.

A faithful, native reimplementation of the official ``MACE-matpes-*-0``
foundation models (``ScaleShiftMACE`` with density-normalised interactions, an
Agnesi radial transform and a ZBL pair-repulsion term) on the cuEquivariance
backend. Every sub-block lives in ``molrep`` / ``molpot``; this module only
wires them into the energy/force forward of MACE's ``ScaleShiftMACE``.

Architecture (as shipped in ``MACE-matpes-r2scan-omat-ft.model``)::

    E0[Z] (frozen per-element reference)
      + scale · ( ZBL(r) + Σ_layers readout_l(h_l) )

    h_0 = Linear(one_hot(Z))
    layer 0: DensityInteraction        → EquivariantProductBasis(use_sc=False)
    layer 1: DensityResidualInteraction → EquivariantProductBasis(use_sc=True)

Use :func:`load_matpes_state_dict` to import weights from an official
checkpoint that has been converted to cueq layout via
``mace.cli.convert_e3nn_cueq`` (run out of tree — molnex never imports
``mace-torch``).

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
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
from molpot.potentials.repulsion import ZBLRepulsion
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.radial import AgnesiTransform, BesselRBF
from molrep.interaction.density import DensityInteraction, DensityResidualInteraction
from molrep.interaction.product_basis import EquivariantProductBasis
from molrep.readout.scalar import LinearReadout, NonLinearReadout


def _sh_irreps(l_max: int) -> str:
    """Spherical-harmonics irreps up to ``l_max`` (parity ``(-1)^l``)."""
    return "+".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1))


class MACEMatpes(nn.Module):
    """Native MACE-MatPES energy/force model (cuEquivariance).

    Args:
        atomic_numbers: Element table (z-table) in checkpoint order.
        atomic_energies: Per-element reference energies ``E0``, same order.
        r_max: Radial cutoff in Angstrom.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p`` (also the ZBL
            envelope exponent, matching MACE).
        l_max: Maximum spherical-harmonics order.
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        max_hidden_l: Highest ``l`` carried in the node state between layers
            (1 for the shipped MatPES models, i.e. ``128x0e+128x1o``).
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        mlp_dim: Hidden width of the final non-linear readout.
        radial_mlp: Hidden widths of the radial weight MLP.
        scale: ``atomic_inter_scale``.
        shift: ``atomic_inter_shift``.
        use_fallback: Pure-torch cuEq path (``True``) or fused kernels
            (``False``, the default). Forces always take the autograd backend
            here — as they do upstream — so the functorch-compatibility reason
            for the fallback can never apply; the fused kernels are measured
            ~36x faster per MD step (SymmetricContraction alone 83x on GH200).
            Set ``True`` only where the fused ops are unavailable (CPU, or no
            ``cuequivariance-ops-torch`` wheel).
    """

    def __init__(
        self,
        *,
        atomic_numbers: list[int],
        atomic_energies: torch.Tensor,
        r_max: float = 6.0,
        num_bessel: int = 10,
        num_polynomial_cutoff: int = 5,
        l_max: int = 3,
        num_features: int = 128,
        max_hidden_l: int = 1,
        num_interactions: int = 2,
        correlation: int = 3,
        mlp_dim: int = 16,
        radial_mlp: list[int] | None = None,
        scale: float = 1.0,
        shift: float = 0.0,
        use_fallback: bool = False,
    ) -> None:
        super().__init__()
        if num_interactions < 2:
            raise ValueError(f"num_interactions must be at least 2, got {num_interactions}")
        ftype = config.ftype
        radial_mlp = list(radial_mlp) if radial_mlp is not None else [64, 64, 64]
        n_el = len(atomic_numbers)
        self.register_buffer(
            "z_table", torch.tensor(atomic_numbers, dtype=torch.long), persistent=True
        )
        self.z_table: torch.Tensor
        self.num_interactions = num_interactions

        sh = _sh_irreps(l_max)
        node_attrs_irreps = f"{n_el}x0e"
        feat0 = f"{num_features}x0e"
        # hidden state between layers, e.g. "128x0e+128x1o"
        hidden = "+".join(
            f"{num_features}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_hidden_l + 1)
        )
        # transient message irreps: node scalars ⊗ Y_l, e.g. "128x0e+...+128x3o"
        target = "+".join(
            f"{num_features}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1)
        )

        # ---- embeddings ----
        self.node_embedding = cuet.Linear(
            cue.Irreps("O3", node_attrs_irreps),
            cue.Irreps("O3", feat0),
            layout=cue.ir_mul,
            dtype=ftype,
        )
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)
        # normalize=False + eps=0 + trainable reproduces MACE's BesselBasis exactly.
        self.bessel = BesselRBF(
            r_cut=r_max, num_radial=num_bessel, normalize=False, eps=0.0, trainable=True
        )
        self.distance_transform = AgnesiTransform()
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max, exponent=num_polynomial_cutoff)
        self.pair_repulsion = ZBLRepulsion(exponent=num_polynomial_cutoff)
        self.atomic_energies = AtomicReferenceEnergy(
            atomic_energies=atomic_energies,
            atomic_numbers=atomic_numbers,
        )

        # ---- interaction / product / readout stack ----
        # The last layer keeps scalars only; every earlier layer keeps `hidden`.
        hidden_sched = [hidden] * (num_interactions - 1) + [feat0]

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.readouts = nn.ModuleList()
        for i in range(num_interactions):
            node_in = feat0 if i == 0 else hidden
            if i == 0:
                interaction: nn.Module = DensityInteraction(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in,
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{num_bessel}x0e",
                    edge_irreps=node_in,
                    target_irreps=target,
                    radial_mlp=radial_mlp,
                    use_fallback=use_fallback,
                )
            else:
                interaction = DensityResidualInteraction(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in,
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{num_bessel}x0e",
                    edge_irreps=node_in,
                    target_irreps=target,
                    hidden_irreps=hidden_sched[i],
                    radial_mlp=radial_mlp,
                    use_fallback=use_fallback,
                )
            self.interactions.append(interaction)
            self.products.append(
                EquivariantProductBasis(
                    node_feats_irreps=target,
                    target_irreps=hidden_sched[i],
                    correlation=correlation,
                    num_elements=n_el,
                    # The first layer's interaction has no residual to carry.
                    use_sc=i > 0,
                    use_fallback=use_fallback,
                )
            )
            is_last = i == num_interactions - 1
            self.readouts.append(
                NonLinearReadout(irreps_in=hidden_sched[i], mlp_dim=mlp_dim)
                if is_last
                else LinearReadout(irreps_in=hidden_sched[i])
            )

        self.scale_shift = GlobalRescale(scale=scale, shift=shift)
        self._elements_validated = False

    def validate_elements(self, Z: torch.Tensor) -> None:
        """Raise if any atomic number in ``Z`` is outside the element table.

        An element outside the table would be snapped onto a neighbouring row
        by ``searchsorted`` and silently produce a wrong energy. The check
        costs a host sync, so it is **not** on the per-step path: ``forward``
        runs it once per model instance (Z is constant over a trajectory, and
        a wrongly wired model/dataset pair fails on the first batch); callers
        binding a new system to an existing instance should invoke it
        themselves.
        """
        unknown = torch.unique(Z[~torch.isin(Z, self.z_table)])
        if unknown.numel():
            raise ValueError(
                f"atomic numbers {unknown.tolist()} are outside this model's "
                f"{self.z_table.numel()}-element table"
            )

    def _node_attrs(self, Z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """One-hot node attributes over the element table ``(N, n_elements)``."""
        z_index = torch.searchsorted(self.z_table, Z.reshape(-1)).to(dtype=torch.long)
        return torch.nn.functional.one_hot(z_index, self.z_table.numel()).to(dtype)

    def _compute_energy(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        shifts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph total energy ``(B,)`` as a pure function of ``positions``.

        Shared core of :meth:`energy_forces` and :meth:`forward`. Recomputes all
        position-derived geometry internally so it can be differentiated with
        ``torch.autograd.grad``.

        Args:
            positions: ``(N, 3)``.
            Z: Atomic numbers ``(N,)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            batch: Graph index per atom ``(N,)``.
            num_graphs: Number of graphs ``B`` (a static shape — deriving it from
                ``batch.max().item()`` would force a host sync and break dynamo).
            shifts: Optional PBC shift vectors ``(E, 3)`` added to the edge
                vectors (``unit_shifts @ cell``).

        Returns:
            Total energy per graph ``(B,)``.
        """
        source, target = edge_index[:, 0], edge_index[:, 1]
        vectors = positions[target] - positions[source]
        if shifts is not None:
            vectors = vectors + shifts
        lengths = torch.linalg.norm(vectors, dim=-1)

        node_attrs = self._node_attrs(Z, positions.dtype)

        # ---- reference energy (position independent) ----
        e0 = _scatter_sum(self.atomic_energies(Z), batch, num_graphs)

        # ---- edge features: cutoff on raw r, Bessel on transformed r ----
        cutoff = self.cutoff_fn(lengths).unsqueeze(-1)
        transformed = self.distance_transform(lengths, Z[source], Z[target])
        edge_feats = self.bessel(transformed) * cutoff
        edge_attrs = self.spherical_harmonics(vectors)

        # ---- interaction energy ----
        node_es = self.pair_repulsion(lengths, Z, edge_index)
        node_feats = self.node_embedding(node_attrs)
        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_attrs, node_feats, edge_attrs, edge_feats, edge_index
            )
            node_feats = self.products[i](node_feats, sc, node_attrs)
            node_es = node_es + self.readouts[i](node_feats).squeeze(-1)

        inter_e = _scatter_sum(self.scale_shift(node_es), batch, num_graphs)
        return e0 + inter_e

    def energy_forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
        shifts: torch.Tensor | None = None,
        compute_forces: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return total energy and forces from raw tensors.

        Args:
            positions: ``(N, 3)``.
            Z: Atomic numbers ``(N,)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            batch: Graph index per atom ``(N,)``.
            num_graphs: Number of graphs; inferred from ``batch`` when omitted
                (which costs a host sync — pass it on compiled paths).
            shifts: Optional PBC shift vectors ``(E, 3)``.
            compute_forces: Whether to compute ``-dE/dx``.

        Returns:
            ``{"energy": (B,), "forces": (N, 3)}`` (forces only when requested).
        """
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        if not compute_forces:
            energy = self._compute_energy(
                positions.detach(), Z, edge_index, batch, num_graphs, shifts
            )
            return {"energy": energy}

        # One forward, then differentiate the graph it built — MACE's own
        # `get_outputs` shape. Evaluating the energy and then re-running it
        # inside a closure would cost two full forwards per MD step.
        pos = positions.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self._compute_energy(pos, Z, edge_index, batch, num_graphs, shifts)
        return {"energy": energy.detach(), "forces": autograd_forces_from_energy(energy, pos)}

    def forward(self, td: TensorDict) -> TensorDict:
        """Run MACE-MatPES on a post-collate batch, writing energy and forces.

        Reads ``atoms.{Z,pos,batch}`` and ``edges.edge_index`` (``(E, 2)`` with
        ``[:,0]`` source / ``[:,1]`` target per the molnex edge convention), plus
        optional ``edges.shifts`` ``(E, 3)`` for periodic systems. The energy is
        evaluated once and differentiated in place via
        :func:`molpot.derivation.force.autograd_forces_from_energy` — MACE's own
        ``get_outputs`` shape, one forward plus one backward per call.

        Args:
            td: post-collate ``TensorDict`` with ``atoms`` / ``edges`` sub-dicts.

        Returns:
            The same ``td`` with ``graphs.energy`` ``(B,)`` and ``atoms.forces``
            ``(N, 3)`` added.
        """
        Z = td["atoms", "Z"]
        positions = td["atoms", "pos"]
        batch = td["atoms", "batch"]
        edge_index = td["edges", "edge_index"]  # (E, 2), the core's own convention
        # Prefer the graphs namespace's static batch size over a
        # ``batch.max().item()`` host sync (which would also break dynamo).
        if "graphs" in td.keys() and td["graphs"].batch_size:
            num_graphs = int(td["graphs"].batch_size[0])
        else:
            num_graphs = int(batch.max().item()) + 1

        # Once per instance: Z is constant over a trajectory and a wrongly
        # wired model/dataset pair fails on the first batch. Per-call this
        # check cost 0.34 ms and a dynamo graph break (data-dependent shape).
        if not self._elements_validated:
            self.validate_elements(Z)
            self._elements_validated = True
        nested = td.keys(include_nested=True)
        shifts = td["edges", "shifts"] if ("edges", "shifts") in nested else None

        needs_leaf = not (positions.requires_grad and positions.is_leaf)
        pos = positions.detach().requires_grad_(True) if needs_leaf else positions
        with torch.enable_grad():
            energy = self._compute_energy(pos, Z, edge_index, batch, num_graphs, shifts)
        forces = autograd_forces_from_energy(energy, pos)

        if "graphs" not in td.keys():
            td["graphs"] = TensorDict({}, batch_size=[num_graphs])
        td["graphs", "energy"] = energy.detach() if needs_leaf else energy
        td["atoms", "forces"] = forces
        return td


# Official cueq key (or key prefix) → molnex name. ``None`` drops the entry:
# either it is rebuilt from the constructor arguments (``r_max``, the cutoff
# scalars, the Z-indexed ``E0`` table) or it is a non-persistent constant.
# Longest prefix wins, so an exact key overrides its enclosing prefix.
# Anything unlisted maps through unchanged (``interactions.*``, ``products.*``,
# ``readouts.*``, ``scale_shift.*``).
_KEY_REMAP: dict[str, str | None] = {
    "node_embedding.linear.": "node_embedding.",
    "radial_embedding.bessel_fn.bessel_weights": "bessel.freqs",
    "radial_embedding.bessel_fn.": None,
    "radial_embedding.distance_transform.": "distance_transform.",
    "radial_embedding.cutoff_fn.": None,
    "pair_repulsion_fn.p": None,  # exponent is a plain int on ZBLRepulsion
    "pair_repulsion_fn.": "pair_repulsion.",
    "atomic_energies_fn.": None,
    "atomic_numbers": "z_table",
    "r_max": None,
    "num_interactions": None,
}
_KEY_REMAP_ORDER = sorted(_KEY_REMAP, key=len, reverse=True)


def load_matpes_state_dict(model: MACEMatpes, cueq_state: dict) -> None:
    """Load a cueq-converted MACE-MatPES ``state_dict`` into :class:`MACEMatpes`.

    Strict by design: every learnable tensor of ``model`` must be filled by the
    checkpoint and every checkpoint weight must land somewhere. A silently
    dropped tensor is the failure mode that produces a model which runs, looks
    sane, and is quietly wrong — so it raises instead.

    Args:
        model: Target model, constructed with the checkpoint's hyper-parameters.
        cueq_state: ``state_dict`` of the cueq-converted official model
            (``mace.cli.convert_e3nn_cueq``, run out of tree).

    Raises:
        RuntimeError: If a checkpoint key has no home, a shape disagrees, or a
            model parameter is left unfilled.
    """
    remap: dict[str, torch.Tensor] = {}
    for key, value in cueq_state.items():
        # cueq stores symbolic graph constants and irrep masks alongside the
        # weights; both are rebuilt by the module and carry nothing learned.
        if ".graph.c" in key or key.endswith("output_mask"):
            continue
        new_key: str | None = key
        for prefix in _KEY_REMAP_ORDER:
            if key == prefix or key.startswith(prefix):
                replacement = _KEY_REMAP[prefix]
                new_key = (
                    None
                    if replacement is None
                    else replacement + (key[len(prefix) :] if key != prefix else "")
                )
                break
        if new_key is not None:
            remap[new_key] = value

    own = model.state_dict()
    unexpected = sorted(set(remap) - set(own))
    if unexpected:
        raise RuntimeError(f"checkpoint keys with no home in MACEMatpes: {unexpected}")

    mismatched = []
    for key, value in remap.items():
        want = own[key].shape
        if value.shape == want:
            continue
        # MACE stores some frozen scalars as (1,) where molnex holds a 0-d
        # buffer; identical content, different rank.
        if value.numel() == own[key].numel():
            remap[key] = value.reshape(want)
        else:
            mismatched.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(want)}")
    if mismatched:
        raise RuntimeError(
            "shape mismatch — model built with the wrong config? " + ", ".join(sorted(mismatched))
        )

    unfilled = sorted(name for name, _ in model.named_parameters() if name not in remap)
    if unfilled:
        raise RuntimeError(f"parameters not covered by the checkpoint: {unfilled}")

    model.load_state_dict(remap, strict=False)
