"""Official MACE checkpoint → molnex ``state_dict``: one key remap for both families.

Loading an official checkpoint is a **dialect translation**: the same numbers
under the names upstream's module tree gave them. This module owns that and
nothing else — no unit conversion (both sides use eV, eV/Å, Å), no
architecture inference, no saving direction.

The checkpoints it reads are already in *cuEquivariance* layout (``cueq`` in
key names). That conversion (``mace.cli.convert_e3nn_cueq``) is an offline step
run out of tree; molnex never imports ``mace-torch``.

:class:`CheckpointRemap` is three ideas:

1. skip the entries cuEquivariance rebuilds anyway — symbolic graph constants
   (``".graph.c" in key``) and irrep masks (``key.endswith("output_mask")``);
2. rename by **longest matching prefix**, with ``None`` meaning "drop" (the
   value is rebuilt from a constructor argument, or is not persistent);
3. normalize scalar ranks and cuEq Linear's optional singleton shared-weight
   axis — never an arbitrary equal-numel reshape, never a CG basis conversion.

The only genuine difference between the two shipped families is what an
unhoused key means, and that is the single knob ``on_unexpected``: MatPES
raises (a homeless key means the mapping is stale), OMol returns the list (that
family ships auxiliary heads this port does not model).

**Strictness is not on that knob.** Under both policies an ``nn.Parameter``
the checkpoint does not fill, or a genuine shape disagreement, raises: a
silently dropped tensor produces a model that runs, looks sane and is quietly
wrong. "Learnable" therefore means membership in ``model.named_parameters()``
and never a ``.weight`` / ``.bias`` name pattern — that shortcut once excused
``bessel.freqs``, kept molnex's analytic values over the checkpoint's, and put
a 2.2e-7 Å⁻¹ offset straight into the energy parity residual
(``src/molzoo/specs/mace_omol.md`` §7.1).

Both lists returned by :meth:`CheckpointRemap.load` are sorted, so a diff of
two loader reports cannot move on dict ordering alone.

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

from collections.abc import Mapping
from typing import Literal

import torch
from torch import nn

#: Official cueq key (or key prefix) → molnex name for the MACE-MatPES family.
#: ``None`` drops the entry: either it is rebuilt from the constructor
#: arguments (``r_max``, the cutoff scalars, the Z-indexed ``E0`` table) or it
#: is a non-persistent constant. Longest prefix wins, so an exact key overrides
#: its enclosing prefix. Anything unlisted maps through unchanged
#: (``interactions.*``, ``products.*``, ``readouts.*``, ``scale_shift.*``).
matpes_keys: dict[str, str | None] = {
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

#: Official cueq key (or key prefix) → molnex name for the MACE-OMOL family,
#: mirroring :data:`matpes_keys`. ``None`` drops the entry (rebuilt from
#: the constructor arguments or a non-persistent constant). Longest prefix
#: wins; anything unlisted maps through unchanged (``interactions.*``,
#: ``products.*``, ``joint_embedding.*``, ``scale_shift.*``).
omol_keys: dict[str, str | None] = {
    "node_embedding.linear.": "node_embedding.",
    "embedding_readout.linear.": "embedding_readout.",
    "readouts.0.": "readout.",
    # MACE's fitted Bessel frequencies (Å⁻¹). Without this entry the key is
    # simply unexpected and `bessel.freqs` keeps its analytic n*pi/r_max init —
    # the official OMOL values sit ~2.2e-7 Å⁻¹ away, which was landing straight
    # in the reported energy parity residual (module docstring, §"Strictness").
    "radial_embedding.bessel_fn.bessel_weights": "bessel.freqs",
    "radial_embedding.bessel_fn.": None,
    "radial_embedding.cutoff_fn.": None,
    "atomic_energies_fn.": None,  # handled at construction (Z-indexed)
    "atomic_numbers": "z_table",
    "r_max": None,
    "num_interactions": None,
}


class CheckpointRemap:
    """Translate official MACE checkpoint keys into molnex ``state_dict`` keys.

    Product classes hold a table as ``_keys`` and call this from
    :meth:`MaceMatpes.load` / :meth:`MaceOmol.load` / :meth:`MacePolar.load`.

    The prefix search order — longest first, so an exact key wins over the
    prefix that encloses it — is computed once here rather than kept as a
    second module-level constant beside the table, which is what the two flat
    loaders did.

    A complete, runnable use of this surface — a synthetic checkpoint written in
    the official dialect, read back, and checked against hard-coded energy /
    force goldens, including both ``on_unexpected`` policies — is
    ``regressions/mace-subpackage-restructure-05-checkpoint.py``.

    Args:
        table: Official cueq key (or key prefix) → molnex name. ``None`` drops
            the entry. Keys absent from the table pass through unchanged.
        on_unexpected: What :meth:`load` does with a renamed key that has no
            home in the target model. ``"raise"`` (the default — the stricter
            behaviour is the default) refuses before touching the model;
            ``"return"`` hands the list back to the caller, which is what the
            OMol family needs because its checkpoints carry auxiliary heads
            this port deliberately does not model.

    Raises:
        ValueError: If ``on_unexpected`` is neither ``"raise"`` nor
            ``"return"``. A typo must not degrade into the lax branch.
    """

    def __init__(
        self,
        table: Mapping[str, str | None],
        *,
        on_unexpected: Literal["raise", "return"] = "raise",
    ) -> None:
        if on_unexpected not in ("raise", "return"):
            raise ValueError(f'on_unexpected must be "raise" or "return", got {on_unexpected!r}')
        self._table: dict[str, str | None] = dict(table)
        #: Prefixes longest-first: an exact key beats the prefix enclosing it.
        self._prefixes: tuple[str, ...] = tuple(sorted(self._table, key=len, reverse=True))
        self._on_unexpected = on_unexpected

    def rename(self, official_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Translate checkpoint keys, dropping what the model rebuilds itself.

        Pure: ``official_state`` is not modified, no model is consulted, and
        the tensors are passed through by reference — nothing is copied, cast
        or converted, so every value keeps the units it had upstream (the two
        sides agree: energies in eV, isolated-atom references in eV/atom,
        forces in eV/Å, lengths in Å, Bessel frequencies in Å⁻¹).

        Dropped entries are the cuEquivariance symbolic graph constants
        (``".graph.c" in key``) and irrep masks (``output_mask``), which the
        modules rebuild and which carry nothing learned, plus every key whose
        table entry is ``None``.

        Args:
            official_state: ``state_dict`` of the cueq-converted official model
                (converted by ``mace.cli.convert_e3nn_cueq``, run out of tree —
                molnex never imports ``mace-torch``).

        Returns:
            A new dict of molnex key → the same tensor objects.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in official_state.items():
            if ".graph.c" in key or key.endswith("output_mask"):
                continue
            new_key: str | None = key
            for prefix in self._prefixes:
                if key.startswith(prefix):
                    replacement = self._table[prefix]
                    new_key = None if replacement is None else replacement + key[len(prefix) :]
                    break
            if new_key is not None:
                if new_key in renamed:
                    raise RuntimeError(f"checkpoint key collision at {new_key!r} (from {key!r})")
                renamed[new_key] = value
        return renamed

    def load(
        self, model: nn.Module, official_state: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load an official checkpoint into ``model``, strictly.

        Preflight names, shapes and parameter coverage before writing tensors.
        Only scalar rank differences and cuEq Linear's singleton shared-weight
        axis may be reshaped; equal numel alone does not establish a layout.

        "Learnable" is defined as membership in ``model.named_parameters()``.
        Never a ``.weight`` / ``.bias`` name heuristic — that shortcut once
        excused the ``nn.Parameter`` ``bessel.freqs`` from this very check; the
        module docstring tells that story.

        Args:
            model: Target model, already constructed with the checkpoint's
                hyper-parameters.
            official_state: ``state_dict`` of the cueq-converted official model.

        Returns:
            ``(missing_buffers, unexpected)``: the non-learnable entries the
            checkpoint did not provide (typically the graph constants cueq
            rebuilds) and the renamed keys with no home in ``model``. Both
            lists are **sorted** — a deterministic normalisation, so two runs
            differ only when the checkpoint does. Under ``on_unexpected =
            "raise"`` the second list is always empty.

        Raises:
            RuntimeError: If a renamed key has no home in ``model`` and the
                policy is ``"raise"``; if a checkpoint tensor disagrees in
                shape/layout with the model's (the model was built from the wrong
                config); or — under **either** policy — if any
                ``nn.Parameter`` of ``model`` is left unfilled, which would
                silently keep whatever the constructor gave it (random for a
                linear weight, the analytic ``nπ/r_max`` for ``bessel.freqs``).
        """
        renamed = self.rename(official_state)
        # ``_extra_state`` is a module's own metadata (``get_extra_state``),
        # not a tensor: ``AtomicReferenceEnergy`` carries its ``scheme`` and
        # ``units`` there. An upstream checkpoint has no such key, and the
        # model's own must be left as constructed — so it takes no part in the
        # name, shape or coverage preflight below, all of which speak tensors.
        renamed = {k: v for k, v in renamed.items() if not _is_extra_state(k)}
        own = {k: v for k, v in model.state_dict().items() if not _is_extra_state(k)}

        unexpected = sorted(set(renamed) - set(own))
        if unexpected and self._on_unexpected == "raise":
            raise RuntimeError(
                f"checkpoint keys with no home in {type(model).__name__}: {unexpected}"
            )

        mismatched: list[str] = []
        import cuequivariance_torch as cuet

        modules = dict(model.named_modules())
        for key, value in renamed.items():
            if key not in own:
                continue  # reported through ``unexpected``
            want = own[key].shape
            if value.shape == want:
                continue
            # MACE stores some frozen scalars as (1,) where molnex holds a 0-d
            # buffer; identical content, different rank.
            owner, _, leaf = key.rpartition(".")
            shared_linear_axis = (
                leaf == "weight"
                and isinstance(modules.get(owner), cuet.Linear)
                and (
                    (value.ndim == 1 and tuple(want) == (1, value.numel()))
                    or (len(want) == 1 and tuple(value.shape) == (1, own[key].numel()))
                )
            )
            if value.numel() == own[key].numel() == 1 or shared_linear_axis:
                renamed[key] = value.reshape(want)
            else:
                mismatched.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(want)}")
        if mismatched:
            raise RuntimeError(
                "shape mismatch — model built with the wrong config? "
                + ", ".join(sorted(mismatched))
            )

        parameters = {name for name, _ in model.named_parameters()}
        unfilled = sorted(parameters - set(renamed))
        if unfilled:
            raise RuntimeError(f"parameters not covered by the checkpoint: {unfilled}")
        missing, _ = model.load_state_dict(renamed, strict=False)
        return sorted(set(missing) - parameters - {k for k in missing if _is_extra_state(k)}), (
            unexpected
        )


def _is_extra_state(key: str) -> bool:
    """Whether *key* is a module's ``get_extra_state`` slot rather than a tensor.

    ``torch.nn.Module`` appends this fixed suffix for any module that defines
    :meth:`~torch.nn.Module.get_extra_state`; the value is whatever that method
    returns (a ``dict`` for :class:`~molpot.heads.energy.AtomicReferenceEnergy`),
    so every tensor-shaped check has to step around it.
    """
    return key.endswith("_extra_state")


polar_keys: dict[str, str | None] = {
    "node_embedding.linear.": "node_embedding.",
    "radial_embedding.bessel_fn.bessel_weights": "bessel.freqs",
    "radial_embedding.bessel_fn.": None,
    "radial_embedding.cutoff_fn.": None,
    "atomic_energies_fn.": None,
    "atomic_numbers": "z_table",
    "r_max": None,
    "num_interactions": None,
    "kspace_cutoff": None,
    "field_feature_norms": None,
    "electric_potential_descriptor.": None,
    "coulomb_energy.": None,
    "external_field_contribution.": None,
    "local_electron_energy.linear_up_q.weight": "local_electron_energy.linear_up_q.linear.weight",
    "local_electron_energy.linear_up_v.weight": "local_electron_energy.linear_up_v.linear.weight",
}
for _i in range(4):
    _p = f"field_dependent_charges_maps.{_i}.readout"
    polar_keys[f"{_p}.linear_mid.weight"] = f"{_p}.linear_mid.linear.weight"
    polar_keys[f"{_p}.linear_2.weight"] = f"{_p}.linear_2.linear.weight"
