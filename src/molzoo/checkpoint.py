"""Official checkpoint → molnex ``state_dict``: the pieces every model loader shares.

Loading an official checkpoint is a **dialect translation**: the same numbers
under the names upstream's module tree gave them. Each model package owns its
key table and its archive reader (``molzoo.mace.checkpoint`` /
``molzoo.mace.utils``, ``molzoo.allegro.checkpoint`` /
``molzoo.allegro.utils``); this module owns what they share:

* :class:`CheckpointRemap` — rename by **longest matching prefix** (``None``
  drops the entry), drop the keys whose name contains a ``skip`` substring
  (what the target rebuilds itself), land upstream's ``.weight`` in the layout
  the target module **declares by type, never by shape** — ``(in, out)``
  transposed on the ``.linear.weight`` of an ``upstream_linear_layout =
  "in_out"`` module, or converted by the function a product registers for the
  module's type (``conversions``; the official dialect lives in the model
  package's checkpoint module, never on the module) — allow only scalar-rank
  and cuEq-Linear singleton-axis reshapes, and refuse to leave any
  ``nn.Parameter`` unfilled.
* :class:`OfficialArchive` — the one reader pattern: ``read(path) ->
  (state, config)``, unpickling through :class:`StubUnpickler`, which rebuilds
  a pickled upstream module *tree* without importing upstream (every class
  outside torch / stdlib / numpy / cuEquivariance becomes an empty
  ``nn.Module`` subclass, so ``nn.Module.__setstate__`` restores parameters,
  buffers and plain attributes and ``state_dict()`` runs). The stubs are an
  import aid, not a sandbox: only read archives you trust.

**Strictness is not on the ``on_unexpected`` knob.** Under both policies an
``nn.Parameter`` the checkpoint does not fill, or a genuine shape
disagreement, raises: a silently dropped tensor produces a model that runs,
looks sane and is quietly wrong. "Learnable" therefore means membership in
``model.named_parameters()`` and never a ``.weight`` / ``.bias`` name pattern —
that shortcut once excused MACE's ``bessel.freqs`` and put a 2.2e-7 Å⁻¹ offset
into the energy parity residual (``src/molzoo/specs/mace_omol.md`` §7.1).
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import torch
from torch import nn


class ModuleStub(nn.Module):
    """Empty stand-in for an upstream class met while unpickling.

    Keeps constructor / reduce arguments in ``_pickle_args`` (irreps-like
    tuples are rebuilt that way) and accepts the container calls a pickled
    ``ModuleList``-like object replays (``__setitem__``, ``append``).
    """

    _pickle_args: tuple[Any, ...]

    def __new__(cls, *args: object, **kwargs: object) -> ModuleStub:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_pickle_args", args)
        return instance

    def __init__(self, *args: object, **kwargs: object) -> None:
        nn.Module.__init__(self)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            nn.Module.__setstate__(self, state)
        else:
            object.__setattr__(self, "_pickle_state", state)

    def __setitem__(self, *args: object) -> None:
        return None

    def append(self, *args: object) -> None:
        return None


class StubUnpickler(pickle.Unpickler):
    """``pickle.Unpickler`` that resolves torch / stdlib / numpy / molrep.eq
    globals and stubs every other class as a :class:`ModuleStub` subclass.

    Official checkpoints name classes under ``cuequivariance`` and
    ``cuequivariance_torch``. Those paths load from :mod:`molrep.eq` and
    :mod:`molrep.eq.torch`.

    One policy for every reader. Tensors stored outside the pickle (a zip's
    ``.data/`` records) come through ``persistent_load``, supplied by the
    reader; ``torch.load`` sets its own.

    Args:
        file: Binary file positioned at the pickle.
        persistent_load: Resolver for persistent ids, or ``None``.
    """

    _ROOTS: ClassVar[tuple[str, ...]] = (
        "torch",
        "collections",
        "builtins",
        "__builtin__",
        "_codecs",
        "copyreg",
        "numpy",
    )
    _FRONTEND: ClassVar[tuple[tuple[str, str], ...]] = (
        ("cuequivariance_torch", "molrep.eq.torch"),
        ("cuequivariance", "molrep.eq"),
    )
    _stubs: ClassVar[dict[tuple[str, str], type]] = {}

    def __init__(
        self, file: Any, *, persistent_load: Callable[[Any], Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(file, **kwargs)
        if persistent_load is not None:
            self.persistent_load = persistent_load

    def find_class(self, module: str, name: str) -> Any:
        for old, new in self._FRONTEND:
            if module == old or module.startswith(old + "."):
                module = new + module[len(old) :]
                break
        if (
            module.split(".")[0] in self._ROOTS
            or module == "molrep.eq"
            or module.startswith("molrep.eq.")
        ):
            return super().find_class(module, name)
        key = (module, name)
        if key not in StubUnpickler._stubs:
            StubUnpickler._stubs[key] = type(name, (ModuleStub,), {"__module__": module})
        return StubUnpickler._stubs[key]


class _StubPickleModule:
    """``pickle_module`` for ``torch.load``: :class:`StubUnpickler` behind the pickle API."""

    Unpickler = StubUnpickler

    @staticmethod
    def load(file: Any, **kwargs: Any) -> Any:
        return StubUnpickler(file, **kwargs).load()


class OfficialArchive:
    """Base of every official-archive reader: ``read(path) -> (state, config)``.

    A reader never imports the upstream package. It unpickles through
    :class:`StubUnpickler` (:meth:`unpickle` for a raw pickle,
    :meth:`torch_load` for a ``torch.save`` file), rebuilds the module tree,
    and recovers the hyper-parameters from the tree's plain attributes. Each
    model package subclasses it (``molzoo.mace.utils.MaceArchive``,
    ``molzoo.allegro.utils.NequipPackage``) and its product's ``load`` calls
    :meth:`read`, then :class:`CheckpointRemap`.
    """

    def read(self, path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Return ``(state, config)``: upstream-named tensors and model hyper-parameters."""
        raise NotImplementedError

    @staticmethod
    def unpickle(data: bytes, *, persistent_load: Callable[[Any], Any] | None = None) -> Any:
        """Unpickle raw ``data`` with the stub policy."""
        return StubUnpickler(io.BytesIO(data), persistent_load=persistent_load).load()

    @staticmethod
    def torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
        """``torch.load`` a file with the stub policy (state dicts and pickled modules alike)."""
        return torch.load(
            path, map_location=map_location, weights_only=False, pickle_module=_StubPickleModule
        )

    @staticmethod
    def tensor_state(state: Any) -> dict[str, torch.Tensor]:
        """Validate a ``state_dict``: a non-empty mapping of string keys to tensors.

        Raises:
            TypeError: Otherwise.
        """
        if not isinstance(state, Mapping) or not state:
            raise TypeError("archive state_dict must be a nonempty tensor mapping")
        if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
            raise TypeError("state_dict must contain only string keys and tensors")
        return dict(state)


class CheckpointRemap:
    """Translate official checkpoint keys into molnex ``state_dict`` keys.

    Each model package's ``load`` builds one over its key table
    (:mod:`molzoo.mace.checkpoint`, :mod:`molzoo.allegro.checkpoint`).

    The prefix search order — longest first, so an exact key wins over the
    prefix that encloses it — is computed once here rather than kept as a
    second module-level constant beside the table, which is what the two flat
    loaders did.

    Args:
        table: Official cueq key (or key prefix) → molnex name. ``None`` drops
            the entry. Keys absent from the table pass through unchanged.
        on_unexpected: What :meth:`load` does with a renamed key that has no
            home in the target model. ``"raise"`` (the default — the stricter
            behaviour is the default) refuses before touching the model;
            ``"return"`` hands the list back to the caller, for a family whose
            checkpoints carry auxiliary heads the port deliberately does not
            model.
        skip: Substrings of official keys to drop before the rename — entries
            that carry nothing learned and that the target rebuilds itself
            (``molzoo.mace.checkpoint.MACE_SKIP``). Checked before the
            unexpected-key policy, so a skipped key is never reported.
        conversions: Module type → ``(module, upstream weight) -> weight``,
            for a module whose forward reads its ``weight`` in a layout the
            checkpoint does not ship (``molzoo.mace.checkpoint.MACE_CONVERSIONS``).
            Applied to upstream's ``<p>.weight`` by the target's type — never by
            shape — before any shape comparison.

    Raises:
        ValueError: If ``on_unexpected`` is neither ``"raise"`` nor
            ``"return"``. A typo must not degrade into the lax branch.
    """

    def __init__(
        self,
        table: Mapping[str, str | None],
        *,
        on_unexpected: Literal["raise", "return"] = "raise",
        skip: tuple[str, ...] = (),
        conversions: Mapping[type[nn.Module], Callable[[nn.Module, torch.Tensor], torch.Tensor]] = (
            MappingProxyType({})
        ),
    ) -> None:
        if on_unexpected not in ("raise", "return"):
            raise ValueError(f'on_unexpected must be "raise" or "return", got {on_unexpected!r}')
        self._table: dict[str, str | None] = dict(table)
        #: Prefixes longest-first: an exact key beats the prefix enclosing it.
        self._prefixes: tuple[str, ...] = tuple(sorted(self._table, key=len, reverse=True))
        self._on_unexpected = on_unexpected
        self._skip: tuple[str, ...] = tuple(skip)
        self._conversions = dict(conversions)

    def rename(self, official_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Translate checkpoint keys, dropping what the model rebuilds itself.

        Pure: ``official_state`` is not modified, no model is consulted, and
        the tensors are passed through by reference — nothing is copied, cast
        or converted, so every value keeps the units it had upstream (the two
        sides agree: energies in eV, isolated-atom references in eV/atom,
        forces in eV/Å, lengths in Å, Bessel frequencies in Å⁻¹).

        Dropped entries are every key containing one of the ``skip``
        substrings given at construction and every key whose table entry is
        ``None``.

        Args:
            official_state: ``state_dict`` of the official model, in the
                dialect the table translates.

        Returns:
            A new dict of molnex key → the same tensor objects.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in official_state.items():
            if any(substring in key for substring in self._skip):
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
        The target module's type, not the tensor's shape, decides how
        upstream's ``<p>.weight`` lands: a module declaring
        ``upstream_linear_layout = "in_out"`` has the ``(in, out)`` matrix
        transposed onto its ``<p>.linear.weight``; a module of a type in
        ``conversions`` has that function applied. Both before any shape
        comparison. Otherwise only scalar rank differences and cuEq Linear's
        singleton shared-weight axis may be reshaped; equal numel alone does
        not establish a layout.

        "Learnable" is defined as membership in ``model.named_parameters()``.
        Never a ``.weight`` / ``.bias`` name heuristic — that shortcut once
        excused the ``nn.Parameter`` ``bessel.freqs`` from this very check; the
        module docstring tells that story.

        Args:
            model: Target model, already constructed with the checkpoint's
                hyper-parameters.
            official_state: ``state_dict`` of the official model.

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
        # The target type decides the layout of upstream's ``<p>.weight``, before
        # any shape comparison: a module declaring ``upstream_linear_layout =
        # "in_out"`` keeps its weight in a contained ``nn.Linear``
        # (``<p>.linear.weight``, ``(out, in)``) where upstream stores ``(in,
        # out)``; a module of a registered type reads a layout upstream never
        # ships, and ``conversions`` names the boundary function that produces
        # it. A square hidden layer, or an equal-numel flat weight, would
        # otherwise match by shape and load
        # unconverted — the silently wrong result the equal-numel rule below
        # exists to refuse.
        for path, module in model.named_modules():
            key = f"{path}.weight" if path else "weight"
            if key not in renamed:
                continue
            convert = next(
                (fn for kind, fn in self._conversions.items() if isinstance(module, kind)), None
            )
            if convert is not None:
                renamed[key] = convert(module, renamed[key])
            elif getattr(type(module), "upstream_linear_layout", None) == "in_out":
                renamed[f"{path}.linear.weight" if path else "linear.weight"] = renamed.pop(key).t()
        own = {k: v for k, v in model.state_dict().items() if not _is_extra_state(k)}

        unexpected = sorted(set(renamed) - set(own))
        if unexpected and self._on_unexpected == "raise":
            raise RuntimeError(
                f"checkpoint keys with no home in {type(model).__name__}: {unexpected}"
            )

        mismatched: list[str] = []
        from molrep.equivariance import Linear

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
                and isinstance(modules.get(owner), Linear)
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
