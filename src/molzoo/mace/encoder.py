"""Feature body shared by the MACE products.

:class:`MACEEncoder` registers embeddings, the interaction stack and the
product basis. It does not register an energy
readout, a reference energy, a scale-shift, a pair repulsion or an embedding
readout — those are heads, and the product that subclasses this body
registers them on itself. Submodule names stay unprefixed, so an official
checkpoint still loads onto the product (upstream's ``ScaleShiftMACE(MACE)``
shape, without a nested ``body.`` prefix).

The irreps schedule is :meth:`~molzoo.mace.spec.MACESpec.body_schedule`. The
class exposes primitives, not a pipeline: ``element_index`` / ``node_attrs`` /
``initial_node_features`` / ``angular_features`` / ``radial_features`` /
``conditioning`` / ``layer_features``. The element index is built once per
call and handed down the stack; nothing below recovers it again. Composing them
into an energy — and differentiating it — is the product's job
(:class:`molzoo.mace.potential.MacePotential`); :meth:`MACEEncoder.forward`
composes them once for the batch schema and writes ``atoms.node_features``,
the contract every molzoo encoder shares.

Element-table validation is the :class:`~molzoo.encoder.ElementTable` gate:
the first eager call of :meth:`forward` (or of a product's ``energy``) spends
it through ``_validate_once``; a traced call never validates, which is why one
eager forward precedes any compile.

Hot-path discipline: every branch is frozen into a plain attribute (or the
presence of an optional submodule) by ``__init__``. The configuration object
survives only as ``self._spec``, for provenance; reading a pydantic attribute
inside a layer loop is a dynamo graph break.

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
import torch
import torch.nn as nn
from tensordict import TensorDict

from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molrep.embedding.radial import AgnesiTransform, BesselRBF
from molrep.equivariance import ChannelWiseTensorProduct
from molrep.interaction.product import irreps_from_l_max, sh_irreps_from_l_max
from molzoo.encoder import ElementTable
from molzoo.mace.coupling import MaceO3
from molzoo.mace.heads import JointFeatureEmbedding, JointFeatureSpec
from molzoo.mace.layers import (
    AgnosticInteraction,
    AgnosticResidualInteraction,
    DensityInteraction,
    DensityResidualInteraction,
    ElementIndex,
    EquivariantProductBasis,
    OneHotLinear,
    ResidualInteraction,
)
from molzoo.mace.spec import MACESpec


class MACEEncoder(ElementTable, nn.Module):
    """Feature body: embeddings, interactions and the product basis.

    Args:
        spec: A validated :class:`~molzoo.mace.spec.MACESpec`. The interaction
            kind and :meth:`~molzoo.mace.spec.MACESpec.body_schedule` select
            the stack. Energy-head fields on the same object are ignored.
        dtype: Floating dtype every parameter and buffer is built at.
    """

    in_keys = [("atoms", "Z"), ("atoms", "pos"), ("pairs", "pair_index")]
    out_keys = [("atoms", "node_features")]

    def __init__(self, spec: MACESpec, *, dtype: torch.dtype) -> None:
        super().__init__()
        #: The build dtype of every float parameter and buffer.
        self.dtype = dtype
        #: Provenance only. Never read on a hot path.
        self._spec = spec

        n_el = len(spec.atomic_numbers)
        l_max = spec.l_max
        num_features = spec.num_features
        num_interactions = spec.num_interactions

        # ---- frozen switches (plain attributes: no pydantic on the hot path) ----
        schedule = spec.body_schedule()
        is_residual = spec.interaction == "residual"
        residual_first = schedule.residual_first
        self.num_interactions = num_interactions
        self.num_features = num_features
        # A density stack multiplies the polynomial envelope into the radial
        # basis. A residual stack keeps it separate and hands it to the
        # interaction, which applies it inside the message.
        self.fold_cutoff_into_radial = not is_residual
        self.pass_cutoff_to_interaction = is_residual

        node_attrs_irreps = f"{n_el}x0e"
        feat0 = f"{num_features}x0e"
        self.scalar_irreps = feat0
        sh = sh_irreps_from_l_max(l_max)
        # Transient message irreps: node scalars ⊗ Y_l, e.g. "128x0e+…+128x3o".
        target = irreps_from_l_max(l_max, num_features)
        hidden = irreps_from_l_max(schedule.hidden_l, num_features)
        if is_residual:
            edge_hidden = irreps_from_l_max(schedule.hidden_l, schedule.edge_channels)
            edge_irr = [feat0] + [edge_hidden] * (num_interactions - 1)
        else:
            edge_irr = [feat0] + [hidden] * (num_interactions - 1)
        node_in = [feat0] + [hidden] * (num_interactions - 1)
        # Upstream MACE narrows only the *last of several* layers to scalars:
        # a single-interaction model keeps its hidden irreps.
        hidden_sched = (
            [hidden] * num_interactions
            if schedule.keep_last_layer_irreps or num_interactions == 1
            else [hidden] * (num_interactions - 1) + [feat0]
        )
        self.layer_irreps = tuple(hidden_sched)
        radial_mlp = list(schedule.radial_mlp)
        product_elements = 1 if schedule.shared_product else n_el

        # ---- embeddings ----
        self.node_embedding = OneHotLinear(
            cue.Irreps("O3", node_attrs_irreps), cue.Irreps("O3", feat0), dtype=dtype
        )
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max, dtype=dtype)
        # normalize=False + eps=0 + trainable reproduces MACE's BesselBasis exactly.
        self.bessel = BesselRBF(
            r_cut=spec.r_max,
            num_radial=spec.num_bessel,
            normalize=False,
            eps=0.0,
            trainable=True,
            dtype=dtype,
        )
        self.distance_transform: AgnesiTransform | None = (
            AgnesiTransform(dtype=dtype) if spec.distance_transform == "agnesi" else None
        )
        self.cutoff_fn = PolynomialCutoff(
            r_cut=spec.r_max, exponent=spec.num_polynomial_cutoff, dtype=dtype
        )
        self.joint_embedding: JointFeatureEmbedding | None = None
        if spec.conditioning == "charge_spin":
            # The spec validator guarantees the four sizes on a conditioned spec.
            self.joint_embedding = JointFeatureEmbedding(
                feature_specs=[
                    JointFeatureSpec(
                        name="total_spin",
                        kind="categorical",
                        emb_dim=num_features,
                        num_classes=spec.spin_classes,
                        per="graph",
                        offset=spec.spin_offset or 0,
                    ),
                    JointFeatureSpec(
                        name="total_charge",
                        kind="categorical",
                        emb_dim=num_features,
                        num_classes=spec.charge_classes,
                        per="graph",
                        offset=spec.charge_offset or 0,
                    ),
                ],
                out_dim=num_features,
                dtype=dtype,
            )
        self.register_buffer(
            "z_table", torch.tensor(spec.atomic_numbers, dtype=torch.long), persistent=True
        )

        # ---- interaction / product stack ----
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
                    dtype=dtype,
                )
            elif spec.interaction == "agnostic":
                # MACE-OFF / MACE-MP-0: normalised by the training set's
                # average neighbour count instead of a learned density.
                common = dict(
                    node_attrs_irreps=node_attrs_irreps,
                    node_feats_irreps=node_in[i],
                    edge_attrs_irreps=sh,
                    edge_feats_irreps=f"{spec.num_bessel}x0e",
                    edge_irreps=edge_irr[i],
                    target_irreps=target,
                    radial_mlp=radial_mlp,
                    avg_num_neighbors=spec.avg_num_neighbors,
                    dtype=dtype,
                )
                interaction = (
                    AgnosticInteraction(**common)
                    if i == 0 and not residual_first
                    else AgnosticResidualInteraction(hidden_irreps=hidden_sched[i], **common)
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
                    dtype=dtype,
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
                    dtype=dtype,
                )
            self.interactions.append(interaction)
            self.products.append(
                EquivariantProductBasis(
                    # Checkpoint weights are coefficients in MACE's coupling basis.
                    group=MaceO3,
                    node_feats_irreps=target,
                    target_irreps=hidden_sched[i],
                    correlation=spec.correlation,
                    num_elements=product_elements,
                    use_sc=is_residual or residual_first or i > 0,
                    dtype=dtype,
                )
            )

    def element_index(self, Z: torch.Tensor) -> ElementIndex:
        """The atoms' rows in the element table, in every form the stack reads.

        Built once per call; ``layer_features`` and the per-element maps below
        it take the result instead of recovering the element from a one-hot.

        Args:
            Z: Atomic numbers ``(N,)``; must be inside the table (see
                :meth:`~molzoo.encoder.ElementTable.validate_elements`).

        Returns:
            The :class:`~molzoo.mace.layers.ElementIndex` over the ``C``
            table rows.
        """
        index = torch.searchsorted(self.z_table, Z.reshape(-1))
        return ElementIndex(index=index, index32=index.to(torch.int32))

    def node_attrs(self, Z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """One-hot node attributes over the element table.

        Args:
            Z: Atomic numbers ``(N,)``; must be inside the table (see
                :meth:`~molzoo.encoder.ElementTable.validate_elements`).
            dtype: Floating dtype of the returned one-hot.

        Returns:
            One-hot element attributes ``(N, n_elements)``.
        """
        z_index = torch.searchsorted(self.z_table, Z.reshape(-1)).to(dtype=torch.long)
        return torch.nn.functional.one_hot(z_index, self.z_table.numel()).to(dtype)

    def initial_node_features(self, elements: ElementIndex) -> torch.Tensor:
        """The element embedding: the scalar node state before any interaction.

        Args:
            elements: The atoms' elements (:meth:`element_index`).

        Returns:
            Initial node features ``(N, num_features)``.
        """
        return self.node_embedding(elements.index)

    def angular_features(self, vectors: torch.Tensor) -> torch.Tensor:
        """Real spherical harmonics ``Y_l(r̂)`` of the edge displacements.

        Args:
            vectors: Edge displacements ``(E, 3)`` in Å, e.g. from
                :func:`molrep.embedding.geometry.edge_vectors`.

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
                :func:`molrep.embedding.geometry.edge_lengths`.
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
        node_feats: torch.Tensor,
        total_charge: torch.Tensor | None,
        total_spin: torch.Tensor | None,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Condition the initial node state on the per-graph total charge and spin.

        One semantics for the encoder and every product: a conditioned spec
        (``conditioning="charge_spin"``) adds the joint embedding, defaulting a
        missing column to the neutral closed-shell singlet — charge ``0``, spin
        ``1`` (the multiplicity ``2S+1``, not a count of unpaired electrons;
        spin ``0`` would index an untrained embedding row); an unconditioned
        spec rejects either column, since silently ignoring it would return a
        plausible, wrong energy. The ``None`` tests are constant-folded by
        dynamo, so the branch costs no graph break.

        Args:
            node_feats: Initial node features ``(N, num_features)``.
            total_charge: Per-graph total charge ``(B,)`` in units of ``e``,
                or ``None``.
            total_spin: Per-graph spin multiplicity ``(B,)``, or ``None``.
            batch: Graph index per atom ``(N,)``.
            num_graphs: Number of graphs ``B``.

        Returns:
            The node features ``(N, num_features)`` — conditioned, or
            unchanged on an unconditioned spec.

        Raises:
            ValueError: If this spec has no charge/spin conditioning and a
                column was supplied.
        """
        if self.joint_embedding is None:
            if total_charge is not None or total_spin is not None:
                raise ValueError(
                    "this MACE variant has no charge/spin conditioning "
                    '(conditioning="none"): total_charge / total_spin would be '
                    "ignored and the energy would be quietly wrong — build it "
                    "from a MACEOMolSpec instead"
                )
            return node_feats
        if total_charge is None:
            total_charge = torch.zeros(num_graphs, dtype=torch.long, device=node_feats.device)
        if total_spin is None:
            total_spin = torch.ones(num_graphs, dtype=torch.long, device=node_feats.device)
        return node_feats + self.joint_embedding(
            batch, total_spin=total_spin, total_charge=total_charge
        )

    def layer_features(
        self,
        *,
        node_feats: torch.Tensor,
        elements: ElementIndex,
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
            elements: The atoms' elements (:meth:`element_index`).
            edge_attrs: Spherical harmonics ``(E, (l_max + 1)²)``.
            edge_feats: Radial basis features ``(E, num_bessel)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target.
            cutoff: Per-edge cutoff envelope ``(E, 1)``; ignored by variants
                that already folded it into ``edge_feats``.

        Returns:
            One node-feature tensor per interaction layer, each ``(N, …)``.
        """
        envelope = cutoff if self.pass_cutoff_to_interaction else None
        # A native tensor product sums over the edge schedule (CSR by target
        # and by source): built once here, shared by every layer, forward and
        # adjoints. cuEq takes no schedule.
        schedule = None
        if any(isinstance(block.conv_tp, ChannelWiseTensorProduct) for block in self.interactions):
            n_nodes = node_feats.shape[0]
            schedule = ChannelWiseTensorProduct.schedule(
                edge_index[:, 0], edge_index[:, 1], n_nodes, n_nodes
            )
        per_layer: list[torch.Tensor] = []
        for i in range(self.num_interactions):
            node_feats, sc = self.interactions[i](
                elements, node_feats, edge_attrs, edge_feats, edge_index, envelope, schedule
            )
            node_feats = self.products[i](node_feats, sc, elements)
            per_layer.append(node_feats)
        return per_layer

    # -------------------------------------------------------------- encoder
    def forward(self, td: TensorDict) -> TensorDict:
        """Encode a batch, writing ``atoms.node_features`` in place.

        The layer widths differ — every layer but the last carries the hidden
        irreps, the last carries scalars — so the ``(N, num_interactions,
        num_features)`` contract stacks each layer's invariant ``0e`` block,
        which in cue's ``ir_mul`` layout is the first ``num_features`` columns
        of that layer's output. The full mixed-l layer outputs stay reachable
        through :meth:`layer_features`.

        Args:
            td: Post-collate batch with ``atoms.{Z,pos,batch}``,
                ``pairs.pair_index`` and the ``graphs`` namespace at
                ``batch_size=[B]``. ``pairs.pair_diff`` (the imaged
                displacement) is used when present;
                ``graphs.{total_charge,total_spin}`` follow
                :meth:`conditioning` (defaulted on a conditioned spec,
                rejected on an unconditioned one).

        Returns:
            The same ``td`` with ``atoms.node_features``
            ``(N, num_interactions, num_features)`` added.
        """
        Z = td["atoms", "Z"]
        self._validate_once(Z)
        # Contiguous for AOTInductor stride specialisation (LAMMPS pair_style).
        edge_index = td["pairs", "pair_index"].contiguous()
        keys = td.keys(include_nested=True)
        diff = td["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        vectors = edge_vectors(td["atoms", "pos"], edge_index, edge_diff=diff)
        lengths = edge_lengths(vectors)

        elements = self.element_index(Z)
        node_feats = self.conditioning(
            self.initial_node_features(elements),
            td["graphs", "total_charge"] if ("graphs", "total_charge") in keys else None,
            td["graphs", "total_spin"] if ("graphs", "total_spin") in keys else None,
            td["atoms", "batch"],
            td["graphs"].batch_size[0],
        )
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        per_layer = self.layer_features(
            node_feats=node_feats,
            elements=elements,
            edge_attrs=self.angular_features(vectors),
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        td["atoms", "node_features"] = torch.stack(
            [h[:, : self.num_features] for h in per_layer], dim=1
        )
        return td
