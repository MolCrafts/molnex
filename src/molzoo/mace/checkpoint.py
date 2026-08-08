"""Official MACE checkpoint → molnex ``state_dict``: one key remap for both families.

MACE is a message-passing neural network for the potential energy of a set of
atoms (arXiv:2206.07697; the energy expression molnex evaluates is written out
in :mod:`molzoo.mace.potential`). An official MACE checkpoint is a *fitted
potential energy surface*: a ``state_dict`` — PyTorch's flat ``name -> tensor``
dump of a model's weights — trained once, on a large dataset, and then used
unchanged. Loading one into molnex is a **dialect translation**: the same
numbers, under the names the upstream module tree happened to give them. This
module owns that translation and nothing else — no unit conversion (both sides
use eV for energies, eV/Å for forces, Å for ``r_max`` and positions, and eV/atom
for the isolated-atom reference energies), no architecture inference, no saving
direction.

The checkpoints it reads have already been converted, out of tree, into the
layout of *cuEquivariance* — NVIDIA's GPU library for the equivariant tensor
algebra MACE is built from, abbreviated ``cueq`` in key names and throughout
this module. That conversion (``mace.cli.convert_e3nn_cueq``) is an offline
step; molnex never imports ``mace-torch``.

:class:`CheckpointRemap` replaces the two hand-copied loaders
(``molzoo.mace_matpes.load_matpes_state_dict`` /
``molzoo.mace_omol.load_omol_state_dict``), which had drifted into two
byte-identical implementations of three ideas:

1. skip the entries cuEquivariance rebuilds anyway — symbolic graph constants
   (``".graph.c" in key``) and *irrep* masks (``key.endswith("output_mask")``).
   An irrep, short for irreducible representation, labels how a feature
   transforms when the molecule is rotated (a scalar stays put, a vector turns
   with it, and so on); the mask records which of those an operation emits, and
   cuEquivariance derives it from the operation's own definition;
2. rename by **longest matching prefix**, with ``None`` meaning "drop" (the
   value is rebuilt from a constructor argument, or is not persistent);
3. reshape a checkpoint tensor whose ``numel`` matches but whose rank does not
   — MACE stores some frozen scalars as ``(1,)`` where molnex holds a 0-d
   buffer.

The **only** genuine difference between the two shipped families — MatPES,
fitted on periodic materials, and OMol, fitted on molecules — is what an
unhoused checkpoint key means, and that is the single knob ``on_unexpected``:
MatPES raises (a key with no home means the mapping is stale), OMol returns the
list (that family ships auxiliary heads this port deliberately does not model).

Strictness is *not* on the knob
------------------------------

Under both policies, an ``nn.Parameter`` the checkpoint does not fill, or a
genuine shape disagreement, raises. Two incidents in this repo are why. Both
predate this module and their original line numbers no longer resolve (the same
step that created this file trimmed those two loaders to wrappers), so they are
quoted here rather than cited by anchor:

* The doctrine, stated by ``load_matpes_state_dict``
  (``src/molzoo/mace_matpes.py``, lines 405-410 before the trim; the sentence
  still stands in that function's docstring today): "a silently dropped tensor
  is the failure mode that produces a model which runs, looks sane, and is
  quietly wrong". Such a model does not crash; it reports plausible energies
  and forces that are simply not the fitted surface.
* The ``bessel.freqs`` incident in ``load_omol_state_dict``
  (``src/molzoo/mace_omol.py``, lines 437-439 before the trim; written up in
  ``src/molzoo/specs/mace_omol.md`` §7.1) is that doctrine defeated by a
  shortcut. MACE expands every interatomic distance ``r`` in a *Bessel* radial
  basis — the functions ``sin(ω_n r) / r``, whose frequencies ``ω_n`` start
  from the analytic values ``nπ/r_max`` (units Å⁻¹) and are then fitted like
  any other weight; molnex holds them in the ``nn.Parameter`` ``bessel.freqs``.
  The check for "was every learnable tensor filled?" used a ``.weight`` /
  ``.bias`` **name suffix** heuristic, which silently excused that parameter,
  since it ends in neither, so the fitted frequencies were dropped and the
  analytic ones stayed. The official values sit ~2.2e-7 Å⁻¹ away from them, and
  that difference went straight into the reported energy parity residual
  (7.0e-7 eV, flagged in §7.1 for re-measurement). Hence: *learnable* means
  exactly ``nn.Parameter``, i.e. membership in ``model.named_parameters()``,
  and never a name pattern.

Both lists returned by :meth:`CheckpointRemap.load` are **sorted**. The flat
OMol loader handed back torch's ``unexpected_keys`` in checkpoint-iteration
order; sorting is a deliberate determinism normalisation, so a diff of two
loader reports cannot move on dict ordering alone. The remap tables themselves
(:data:`MATPES_KEY_REMAP`, :data:`OMOL_KEY_REMAP`) are carried over key-for-key
from those two modules — their content is a chain invariant of the
``mace-subpackage-restructure`` spec chain and is not edited here.

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
MATPES_KEY_REMAP: dict[str, str | None] = {
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
#: mirroring :data:`MATPES_KEY_REMAP`. ``None`` drops the entry (rebuilt from
#: the constructor arguments or a non-persistent constant). Longest prefix
#: wins; anything unlisted maps through unchanged (``interactions.*``,
#: ``products.*``, ``joint_embedding.*``, ``scale_shift.*``).
OMOL_KEY_REMAP: dict[str, str | None] = {
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

    Construct one per checkpoint family (the two shipped ones are the module
    constants :data:`MATPES_REMAP` and :data:`OMOL_REMAP`), then either inspect
    the translation with :meth:`rename` or apply it with :meth:`load`::

        MATPES_REMAP.load(model, torch.load(path, weights_only=True))

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
                renamed[new_key] = value
        return renamed

    def load(
        self, model: nn.Module, official_state: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load an official checkpoint into ``model``, strictly.

        One named action: :meth:`rename`, then the unexpected-key policy, then
        the numel-conserving reshape, then ``load_state_dict(strict=False)``,
        then the strictness check. The policy branch and the shape check both
        run *before* any tensor reaches the model, so those two rejections
        never leave a half-loaded model behind. The parameter-coverage check is
        the one that cannot: it can only ask what ``load_state_dict`` failed to
        fill, so that failure does leave the model written to. It is fatal by
        design — construct a fresh model rather than retrying on this one.

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
                ``numel`` with the model's (the model was built from the wrong
                config); or — under **either** policy — if any
                ``nn.Parameter`` of ``model`` is left unfilled, which would
                silently keep whatever the constructor gave it (random for a
                linear weight, the analytic ``nπ/r_max`` for ``bessel.freqs``).
        """
        renamed = self.rename(official_state)
        own = model.state_dict()

        unexpected = sorted(set(renamed) - set(own))
        if unexpected and self._on_unexpected == "raise":
            raise RuntimeError(
                f"checkpoint keys with no home in {type(model).__name__}: {unexpected}"
            )

        mismatched: list[str] = []
        for key, value in renamed.items():
            if key not in own:
                continue  # reported through ``unexpected``
            want = own[key].shape
            if value.shape == want:
                continue
            # MACE stores some frozen scalars as (1,) where molnex holds a 0-d
            # buffer; identical content, different rank.
            if value.numel() == own[key].numel():
                renamed[key] = value.reshape(want)
            else:
                mismatched.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(want)}")
        if mismatched:
            raise RuntimeError(
                "shape mismatch — model built with the wrong config? "
                + ", ".join(sorted(mismatched))
            )

        missing, _ = model.load_state_dict(renamed, strict=False)
        parameters = {name for name, _ in model.named_parameters()}
        unfilled = sorted(name for name in missing if name in parameters)
        if unfilled:
            raise RuntimeError(f"parameters not covered by the checkpoint: {unfilled}")
        return sorted(set(missing) - parameters), unexpected


#: The MACE-MatPES loading policy: a checkpoint key with no home means the
#: mapping is stale, so refuse rather than load part of the surface.
MATPES_REMAP = CheckpointRemap(MATPES_KEY_REMAP)

#: The MACE-OMOL loading policy: unhoused keys are *returned*, because the OMol
#: checkpoints carry auxiliary heads this port deliberately does not model.
#: Everything else (unfilled parameters, shape disagreements) still raises.
OMOL_REMAP = CheckpointRemap(OMOL_KEY_REMAP, on_unexpected="return")
