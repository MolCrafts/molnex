"""Configuration-driven backbone shared by the MACE foundation models.

:class:`MACEEncoder` registers the module graph of one MACE variant — which
one is decided entirely by the five ``Literal`` switches on
:class:`~molzoo.mace.spec.MACESpec`. Submodule names are byte-identical to the
flat ``MACEMatpes`` / ``MACEOMol`` models, so an official checkpoint keeps
loading without a key rewrite and the energy/force variant classes can simply
inherit this backbone (upstream's ``ScaleShiftMACE(MACE)`` shape).

The class exposes primitives, not a pipeline: ``validate_elements`` /
``node_attrs`` / ``initial_node_features`` / ``angular_features`` /
``radial_features`` / ``conditioning`` / ``layer_features``. Composing them
into an energy — and differentiating it — is the caller's job.

Hot-path discipline: every branch is frozen into a plain attribute (or the
presence of an optional submodule) by ``__init__``. The configuration object
survives only as ``self._spec``, for provenance; reading a pydantic attribute
inside a layer loop is a dynamo graph break (cf. ``src/molzoo/mace_matpes.py:118``
at 0e05959, before deletion).

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

from molix import config
from molpot.heads.energy import AtomicReferenceEnergy
from molpot.heads.rescale import GlobalRescale
from molpot.potentials.repulsion import ZBLRepulsion
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.node import JointFeatureEmbedding, JointFeatureSpec
from molrep.embedding.radial import AgnesiTransform, BesselRBF
from molrep.interaction.mace.density import DensityInteraction, DensityResidualInteraction
from molrep.interaction.product_basis import EquivariantProductBasis
from molrep.interaction.residual import ResidualInteraction
from molrep.readout.mace import LinearReadout, NonLinearBiasReadout, NonLinearReadout
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec


def _irreps(l_max: int, mul: int) -> str:
    """``mul``-fold irreps up to ``l_max`` with natural parity ``(-1)^l``.

    Args:
        l_max: Highest angular-momentum order (inclusive).
        mul: Channel multiplicity of every order.

    Returns:
        A cuEquivariance irreps string, e.g. ``"128x0e+128x1o+128x2e"``.
    """
    return "+".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1))


class MACEEncoder(nn.Module):
    """MACE foundation-model backbone, built from a validated configuration.

    Args:
        spec: A validated :class:`~molzoo.mace.spec.MACESpec` — in practice a
            :class:`~molzoo.mace.spec.MACEMatpesSpec` or
            :class:`~molzoo.mace.spec.MACEOMolSpec`, which carry the
            variant-only fields the corresponding stack needs.

    Raises:
        TypeError: If the ``interaction`` switch asks for a stack whose
            variant-only fields the given spec does not carry.
    """

    def __init__(self, spec: MACESpec) -> None:
        super().__init__()
        ftype = config.ftype
        #: Provenance only (05-checkpoint reads it); never touched on a hot path.
        self._spec = spec

        n_el = len(spec.atomic_numbers)
        l_max = spec.l_max
        num_features = spec.num_features
        num_interactions = spec.num_interactions

        # ---- frozen switches (plain attributes: no pydantic on the hot path) ----
        is_residual = spec.interaction == "residual"
        self.num_interactions = num_interactions
        # MACE-MP/MatPES multiplies the polynomial envelope into the radial
        # basis; OMOL keeps it separate and hands it to the interaction, which
        # applies it inside the message. Same envelope, different fold point.
        self.fold_cutoff_into_radial = not is_residual
        self.pass_cutoff_to_interaction = is_residual

        node_attrs_irreps = f"{n_el}x0e"
        feat0 = f"{num_features}x0e"
        sh = _irreps(l_max, 1)
        # Transient message irreps: node scalars ⊗ Y_l, e.g. "128x0e+…+128x3o".
        target = _irreps(l_max, num_features)

        # ---- per-layer irreps schedule (the one place the variants differ) ----
        if not is_residual:
            if not isinstance(spec, MACEMatpesSpec):
                raise TypeError(
                    'interaction="density" needs a MACEMatpesSpec '
                    "(max_hidden_l / radial_mlp are not on the base spec)"
                )
            hidden = _irreps(spec.max_hidden_l, num_features)
            edge_irr = [feat0] + [hidden] * (num_interactions - 1)
            radial_mlp = list(spec.radial_mlp)
        else:
            if not isinstance(spec, MACEOMolSpec):
                raise TypeError(
                    'interaction="residual" needs a MACEOMolSpec '
                    "(edge_channels is not on the base spec)"
                )
            # OMOL drops the top order from the node state and squeezes the
            # mid-layer edge irreps through the edge_channels bottleneck.
            hidden = _irreps(l_max - 1, num_features)
            edge_irr = [feat0] + [_irreps(l_max - 1, spec.edge_channels)] * (num_interactions - 1)
            radial_mlp = [spec.edge_channels] * 3
        node_in = [feat0] + [hidden] * (num_interactions - 1)
        # The last layer keeps scalars only; every earlier layer keeps `hidden`.
        hidden_sched = [hidden] * (num_interactions - 1) + [feat0]

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
            r_cut=spec.r_max,
            num_radial=spec.num_bessel,
            normalize=False,
            eps=0.0,
            trainable=True,
        )
        self.distance_transform: AgnesiTransform | None = (
            AgnesiTransform() if spec.distance_transform == "agnesi" else None
        )
        self.cutoff_fn = PolynomialCutoff(r_cut=spec.r_max, exponent=spec.num_polynomial_cutoff)
        self.pair_repulsion: ZBLRepulsion | None = (
            ZBLRepulsion(exponent=spec.num_polynomial_cutoff)
            if spec.pair_repulsion == "zbl"
            else None
        )
        self.joint_embedding: JointFeatureEmbedding | None = None
        self.embedding_readout: cuet.Linear | None = None
        if spec.conditioning == "charge_spin":
            if not isinstance(spec, MACEOMolSpec):
                raise TypeError(
                    'conditioning="charge_spin" needs a MACEOMolSpec '
                    "(charge/spin class counts are not on the base spec)"
                )
            self.joint_embedding = JointFeatureEmbedding(
                feature_specs=[
                    JointFeatureSpec(
                        name="total_spin",
                        kind="categorical",
                        emb_dim=num_features,
                        num_classes=spec.spin_classes,
                        per="graph",
                        offset=spec.spin_offset,
                    ),
                    JointFeatureSpec(
                        name="total_charge",
                        kind="categorical",
                        emb_dim=num_features,
                        num_classes=spec.charge_classes,
                        per="graph",
                        offset=spec.charge_offset,
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
            atomic_energies=spec.atomic_energies,
            atomic_numbers=spec.atomic_numbers,
        )
        self.register_buffer(
            "z_table", torch.tensor(spec.atomic_numbers, dtype=torch.long), persistent=True
        )
        self.z_table: torch.Tensor

        # ---- interaction / product stack ----
        # The density family carries per-element symmetric-contraction weights;
        # the OMOL residual family shares one set across elements
        # (``num_elements=1``, mace_omol.py:179).
        product_elements = 1 if is_residual else n_el
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        for i in range(num_interactions):
            interaction: nn.Module
            if is_residual:
                interaction = ResidualInteraction(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in[i],
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{spec.num_bessel}x0e",
                    edge_irreps=edge_irr[i],
                    target_irreps=target,
                    hidden_irreps=hidden_sched[i],
                    radial_mlp=radial_mlp,
                    use_fallback=spec.use_fallback,
                )
            elif i == 0:
                # The first layer has no residual to carry, hence no skip input.
                interaction = DensityInteraction(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in[i],
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{spec.num_bessel}x0e",
                    edge_irreps=edge_irr[i],
                    target_irreps=target,
                    radial_mlp=radial_mlp,
                    use_fallback=spec.use_fallback,
                )
            else:
                interaction = DensityResidualInteraction(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in[i],
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{spec.num_bessel}x0e",
                    edge_irreps=edge_irr[i],
                    target_irreps=target,
                    hidden_irreps=hidden_sched[i],
                    radial_mlp=radial_mlp,
                    use_fallback=spec.use_fallback,
                )
            self.interactions.append(interaction)
            self.products.append(
                EquivariantProductBasis(
                    node_feats_irreps=target,
                    target_irreps=hidden_sched[i],
                    correlation=spec.correlation,
                    num_elements=product_elements,
                    use_sc=is_residual or i > 0,
                    use_fallback=spec.use_fallback,
                )
            )

        # ---- readout ----
        self.readouts: nn.ModuleList | None = None
        self.readout: NonLinearBiasReadout | None = None
        if spec.readout == "per_layer":
            self.readouts = nn.ModuleList(
                [
                    NonLinearReadout(irreps_in=hidden_sched[i], mlp_dim=spec.mlp_dim)
                    if i == num_interactions - 1
                    else LinearReadout(irreps_in=hidden_sched[i])
                    for i in range(num_interactions)
                ]
            )
        else:
            self.readout = NonLinearBiasReadout(irreps_in=feat0, mlp_dim=spec.mlp_dim)

        self.scale_shift = GlobalRescale(scale=spec.scale, shift=spec.shift)

    def validate_elements(self, Z: torch.Tensor) -> None:
        """Raise if any atomic number in ``Z`` is outside the element table.

        An element outside the table would be snapped onto a neighbouring row
        by ``searchsorted`` and silently produce a wrong energy. The check
        costs a host sync, so it is **not** meant for the per-step path: run it
        once per (model, system) pair — ``Z`` is constant over a trajectory,
        and a wrongly wired model/dataset pair fails on the first batch.

        Args:
            Z: Atomic numbers ``(N,)``.

        Raises:
            ValueError: Listing the atomic numbers outside the table.
        """
        unknown = torch.unique(Z[~torch.isin(Z, self.z_table)])
        if unknown.numel():
            raise ValueError(
                f"atomic numbers {unknown.tolist()} are outside this model's "
                f"{self.z_table.numel()}-element table"
            )

    def node_attrs(self, Z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """One-hot node attributes over the element table.

        Args:
            Z: Atomic numbers ``(N,)``; must be inside the table (see
                :meth:`validate_elements`).
            dtype: Floating dtype of the returned one-hot.

        Returns:
            One-hot element attributes ``(N, n_elements)``.
        """
        z_index = torch.searchsorted(self.z_table, Z.reshape(-1)).to(dtype=torch.long)
        return torch.nn.functional.one_hot(z_index, self.z_table.numel()).to(dtype)

    def initial_node_features(self, node_attrs: torch.Tensor) -> torch.Tensor:
        """Project the one-hot element attributes into the scalar node state.

        Args:
            node_attrs: One-hot element attributes ``(N, n_elements)``.

        Returns:
            Initial node features ``(N, num_features)``.
        """
        return self.node_embedding(node_attrs)

    def angular_features(self, vectors: torch.Tensor) -> torch.Tensor:
        """Real spherical harmonics ``Y_l(r̂)`` of the edge displacements.

        Args:
            vectors: Edge displacements ``(E, 3)`` in Å, e.g. from
                :func:`molzoo.mace.geometry.edge_vectors`.

        Returns:
            Edge attributes ``(E, (l_max + 1)²)``.
        """
        return self.spherical_harmonics(vectors)

    def radial_features(
        self, lengths: torch.Tensor, Z: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Radial basis and cutoff envelope of the edges.

        The envelope is always evaluated on the *raw* distance; whether it is
        folded into the returned basis (MACE-MP/MatPES) or left for the
        interaction to apply (OMOL) follows the ``interaction`` switch.

        Args:
            lengths: Edge lengths ``(E,)`` in Å, e.g. from
                :func:`molzoo.mace.geometry.edge_lengths`.
            Z: Atomic numbers ``(N,)`` — needed by the Agnesi transform, which
                is element-pair dependent.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target.

        Returns:
            ``(edge_feats (E, num_bessel), cutoff (E, 1))``.
        """
        cutoff = self.cutoff_fn(lengths).unsqueeze(-1)
        distances = lengths
        if self.distance_transform is not None:
            distances = self.distance_transform(lengths, Z[edge_index[:, 0]], Z[edge_index[:, 1]])
        edge_feats = self.bessel(distances)
        if self.fold_cutoff_into_radial:
            edge_feats = edge_feats * cutoff
        return edge_feats, cutoff

    def conditioning(
        self,
        batch: torch.Tensor,
        *,
        total_spin: torch.Tensor,
        total_charge: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom charge / spin embedding, broadcast from per-graph scalars.

        Args:
            batch: Graph index per atom ``(N,)``.
            total_spin: Per-graph total spin ``(B,)``.
            total_charge: Per-graph total charge ``(B,)`` in units of ``e``.

        Returns:
            Conditioning features ``(N, num_features)`` to add to the initial
            node features.

        Raises:
            ValueError: If this variant has no charge/spin conditioning.
        """
        if self.joint_embedding is None:
            raise ValueError(
                "this MACE variant carries no charge/spin conditioning "
                '(conditioning="none"); build it from a MACEOMolSpec instead'
            )
        return self.joint_embedding(batch, total_spin=total_spin, total_charge=total_charge)

    def layer_features(
        self,
        *,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the interaction/product stack, returning the per-layer node state.

        A ``list`` rather than a stacked tensor: the layer widths differ (every
        layer but the last carries ``hidden`` irreps, the last carries scalars).

        Args:
            node_feats: Initial node features ``(N, num_features)``.
            node_attrs: One-hot element attributes ``(N, n_elements)``.
            edge_attrs: Spherical harmonics ``(E, (l_max + 1)²)``.
            edge_feats: Radial basis features ``(E, num_bessel)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target.
            cutoff: Per-edge cutoff envelope ``(E, 1)``; ignored by variants
                that already folded it into ``edge_feats``.

        Returns:
            One node-feature tensor per interaction layer, each ``(N, …)``.
        """
        envelope = cutoff if self.pass_cutoff_to_interaction else None
        per_layer: list[torch.Tensor] = []
        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                node_attrs, node_feats, edge_attrs, edge_feats, edge_index, envelope
            )
            node_feats = self.products[i](node_feats, sc, node_attrs)
            per_layer.append(node_feats)
        return per_layer
