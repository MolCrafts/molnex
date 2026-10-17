"""Official MACE checkpoint key tables (cuEquivariance dialect → molnex names).

The checkpoints these tables read are already in *cuEquivariance* layout
(``cueq`` in key names). That conversion (MACE's official cuEquivariance
converter) is an offline step run out of tree; molnex never imports
``mace-torch``. The remap machinery itself is
:class:`molzoo.checkpoint.CheckpointRemap`; each product's ``load`` builds it
over its family's table, with :data:`MACE_SKIP` as the ``skip`` tuple.

Each table maps an official key (or key prefix) to a molnex name; ``None``
drops the entry — it is rebuilt from the spec (``r_max``, the cutoff scalars,
the Z-indexed ``E0`` table) or is a non-persistent constant. Longest prefix
wins, so an exact key overrides its enclosing prefix. Anything unlisted maps
through unchanged (``interactions.*``, ``products.*``, ``readouts.*``,
``scale_shift.*``).

The only genuine difference between the shipped families is what an unhoused
key means — MatPES and MACE-OFF raise (a homeless key means the table is
stale), OMol and Polar return the list (those families ship auxiliary heads
this port does not model) — which each product states as the
``on_unexpected`` it passes.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
"""

from __future__ import annotations

#: Substrings of checkpoint keys that carry nothing learned and that the
#: modules rebuild from their irreps: cuEquivariance's symbolic graph
#: constants, irrep masks and symmetric-contraction projections. A
#: checkpoint's projection is not merely redundant: it belongs to the
#: polynomial the converter built, which need not be the module's (a float32
#: coupling computation keeps near-zero paths the float64 one prunes), and
#: loading it beside a different polynomial returns a wrong ``l >= 2`` output.
MACE_SKIP: tuple[str, ...] = (".graph.c", "output_mask", ".symmetric_contractions.projection")

#: Entries every family shares. The Bessel frequencies are MACE's fitted
#: values (Å⁻¹): without that entry the key is merely unexpected and
#: ``bessel.freqs`` keeps its analytic ``nπ/r_max`` init — the official OMol
#: values sit ~2.2e-7 Å⁻¹ away, straight in the energy parity residual.
_common: dict[str, str | None] = {
    "node_embedding.linear.": "node_embedding.",
    "radial_embedding.bessel_fn.bessel_weights": "bessel.freqs",
    "radial_embedding.bessel_fn.": None,
    "radial_embedding.cutoff_fn.": None,
    "atomic_energies_fn.": None,
    "atomic_numbers": "z_table",
    "r_max": None,
    "num_interactions": None,
}

#: MACE-MatPES: the common table plus the Agnesi transform and the ZBL term.
matpes_keys: dict[str, str | None] = {
    **_common,
    "radial_embedding.distance_transform.": "distance_transform.",
    "pair_repulsion_fn.p": None,  # exponent is a plain int on ZBLRepulsion
    "pair_repulsion_fn.": "pair_repulsion.",
}

#: MACE-OFF: the common table; mace >= 0.3.17 stores the neighbour divisor as
#: a buffer per block, which the reader already took into the spec.
off_keys: dict[str, str | None] = {
    **_common,
    **{f"interactions.{i}.avg_num_neighbors": None for i in range(16)},
}

#: MACE-LES: MACE-OFF plus one latent-charge readout per layer.
les_keys: dict[str, str | None] = {**off_keys, "les_readouts.": "charge_readouts."}

#: MACE-OMol: the common table plus the embedding readout and the single
#: final readout molnex holds where the official checkpoint numbers a
#: one-entry ``readouts`` list.
omol_keys: dict[str, str | None] = {
    **_common,
    "embedding_readout.linear.": "embedding_readout.",
    "readouts.0.": "readout.",
}

#: MACE-POLAR: the common table, the k-space constants both GTO blocks
#: rebuild, and the biased linears molnex nests one level deeper.
polar_keys: dict[str, str | None] = {
    **_common,
    "kspace_cutoff": None,
    "field_feature_norms": None,
    "electric_potential_descriptor.": None,
    "coulomb_energy.": None,
    "external_field_contribution.": None,
    "local_electron_energy.linear_up_q.weight": "local_electron_energy.linear_up_q.linear.weight",
    "local_electron_energy.linear_up_v.weight": "local_electron_energy.linear_up_v.linear.weight",
    **{
        f"field_dependent_charges_maps.{i}.readout.{name}.weight": (
            f"field_dependent_charges_maps.{i}.readout.{name}.linear.weight"
        )
        for i in range(4)
        for name in ("linear_mid", "linear_2")
    },
}
