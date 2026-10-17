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

Two weights change **layout** at this boundary, not only name. cuEquivariance
keeps every product's weight as one flat row ``(1, numel)``; the per-element
maps of :mod:`molzoo.mace.layers` store theirs element-major, one row per
element, so a call gathers rows instead of transposing blocks.
:func:`element_linear_weight` and :func:`one_hot_weight` are those
conversions — pure functions of the flat tensor and the target module, which
the module types declare as their ``upstream_weight``
(:class:`molzoo.checkpoint.ConvertedWeight`) and the loader applies once,
by type. Nothing downstream ever sees the flat layout.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from molzoo.mace.layers import ElementLinear, OneHotLinear

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


def element_linear_weight(linear: ElementLinear, flat: torch.Tensor) -> torch.Tensor:
    """``skip_tp``'s weight in :class:`~molzoo.mace.layers.ElementLinear`'s layout.

    The fully connected product keeps one flat row, path by path, each path's
    block ``(u, C, w)``; the module keeps ``(C, Σ_p u_p·w_p)``, every path's
    ``(u, w)`` block of an element contiguous in that element's row, at the
    column the path records.

    Args:
        linear: The target module; its paths carry each block's offset in the
            flat weight and its column in the module's.
        flat: The product's weight ``(1, Σ_p u_p·C·w_p)``.

    Returns:
        A new tensor ``(C, Σ_p u_p·w_p)``.

    Raises:
        ValueError: If ``flat`` is not that one-row tensor — an equal numel in
            another shape is not the product's layout.
    """
    elements = linear.num_elements
    numel = sum(path.u * elements * path.w for path in linear.paths)
    if tuple(flat.shape) != (1, numel):
        raise ValueError(
            f"expected the product's flat weight (1, {numel}), got {tuple(flat.shape)}"
        )
    weight = flat.new_empty(elements, sum(path.u * path.w for path in linear.paths))
    for path in linear.paths:
        size = path.u * elements * path.w
        block = flat[0, path.weight_start : path.weight_start + size].view(path.u, elements, path.w)
        weight[:, path.column : path.column + path.u * path.w] = block.transpose(0, 1).reshape(
            elements, path.u * path.w
        )
    return weight


def one_hot_weight(lookup: OneHotLinear, flat: torch.Tensor) -> torch.Tensor:
    """The linear's weight in :class:`~molzoo.mace.layers.OneHotLinear`'s layout.

    Args:
        lookup: The target module.
        flat: ``cuet.Linear(Cx0e → Fx0e)``'s weight ``(1, C·F)``.

    Returns:
        ``(C, F)``: row ``e`` is the linear's output on ``one_hot(e)`` before
        the path coefficient.

    Raises:
        ValueError: If ``flat`` is not that one-row tensor.
    """
    shape = (1, lookup.num_elements * lookup.out_features)
    if tuple(flat.shape) != shape:
        raise ValueError(f"expected the linear's flat weight {shape}, got {tuple(flat.shape)}")
    return flat.reshape(lookup.num_elements, lookup.out_features)
