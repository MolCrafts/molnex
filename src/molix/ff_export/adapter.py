"""BackendAdapter protocol + self-registering peer types.

Mirrors :class:`molix.engine.EngineAdapter`: each backend is a named
peer class (``LammpsAdapter``, ``OpenMMAdapter``, a future ``GromacsAdapter``),
**not** a ``Translator(method="openmm")`` switch.

References:
    Placement note: ``.claude/notes/learnable-classical-ff.md`` — one peer class
    per backend, never a ``method=`` switch (the
    ``learnable-classical-ff-08-ff-export`` spec it came from was completed and
    removed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from molix.ff_export.force_spec import ForceSpec
from molpot.ir import NonbondedScaling, PotentialIR

__all__ = ["BackendAdapter"]


class BackendAdapter(ABC):
    """Strategy that translates :class:`~molpot.ir.PotentialIR` → :class:`ForceSpec`.

    Concrete subclasses set a class-level :attr:`name` (auto-registered) and
    implement :meth:`translate`. Resolve by name via :meth:`from_name`.

    Attributes:
        name: Backend key the subclass registers itself under (``"lammps"``,
            ``"openmm"``). Left empty on this abstract base, which is what
            keeps the base itself out of the registry.
    """

    name: str = ""
    _registry: dict[str, type[BackendAdapter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            BackendAdapter._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> BackendAdapter:
        """Instantiate the registered adapter for ``name`` (e.g. ``"openmm"``).

        Args:
            name: A key from :meth:`names`.

        Returns:
            A new adapter of that backend, built with its own defaults.

        Raises:
            ValueError: ``name`` is not registered; the message lists the keys
                that are.
        """
        try:
            return cls._registry[name]()
        except KeyError:
            valid = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(f"unknown adapter {name!r}; valid adapters: {valid}") from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """All registered adapter names."""
        return tuple(sorted(cls._registry))

    @staticmethod
    def as_floats(t: torch.Tensor) -> list[float]:
        """Flatten ``t`` to a Python float list (CPU, detached)."""
        return [float(x) for x in t.detach().cpu().reshape(-1).tolist()]

    @staticmethod
    def as_ints(t: torch.Tensor) -> list[int]:
        """Flatten ``t`` to a Python int list (CPU, detached)."""
        return [int(x) for x in t.detach().cpu().reshape(-1).tolist()]

    @staticmethod
    def scaling_dict(scaling: NonbondedScaling | None) -> dict[str, float] | None:
        """Flatten the 1–2 / 1–3 / 1–4 nonbonded scales for serialization.

        A pair of atoms is *1–2* when one covalent bond separates them, *1–3*
        when two do and *1–4* when three do; a force field damps or removes
        those nonbonded pairs because its bonded terms already describe them
        (see :class:`~molpot.ir.NonbondedScaling`). Every backend spells the
        six resulting factors differently, so the flat spelling lives here in
        the export layer rather than on the physics type: the values are read
        through :meth:`~molpot.ir.NonbondedScaling.triple`, which is the same
        per-class ``(s12, s13, s14)`` accessor
        :meth:`~molpot.ir.NonbondedScaling.as_dict` reports nested. The key
        set follows :meth:`~molpot.ir.NonbondedScaling.kinds` rather than a
        hard-coded ``q`` / ``lj`` pair, so an interaction class a later model
        declares — ``induced`` for a polarizable force field — is exported
        under ``scale_induced_<12|13|14>`` instead of being silently dropped.

        Args:
            scaling: The force field's per-interaction-class scales, or
                ``None`` when the IR carries none.

        Returns:
            One ``scale_<kind>_<12|13|14>`` key per declared class —
            ``{"scale_q_12", …, "scale_lj_14"}`` for the Class-I default —
            → dimensionless factor, where ``0`` removes the pair and ``1``
            keeps it in full. ``None`` when ``scaling`` is ``None``, which a
            backend must treat as "no exclusion policy declared", not as "no
            exclusions".
        """
        if scaling is None:
            return None
        return {
            f"scale_{kind}_{relation}": value
            for kind in scaling.kinds()
            for relation, value in zip((12, 13, 14), scaling.triple(kind), strict=True)
        }

    @abstractmethod
    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Translate ``ir`` into a backend force specification.

        Args:
            ir: Class-I parameter bags (+ optional scaling).
            meta: Optional attachments (type systems, symbolic FF, …).

        Returns:
            A serializable :class:`ForceSpec`.

        Raises:
            UnsupportedTermError: When an IR bag has no faithful mapping.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
