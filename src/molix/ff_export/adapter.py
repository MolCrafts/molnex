"""BackendAdapter protocol + self-registering peer types.

Mirrors :class:`molix.engine.EngineAdapter`: each backend is a named
peer class (``OpenMMAdapter``, future ``GromacsAdapter``), **not** a
``Translator(method="openmm")`` switch.

References:
    Spec: learnable-classical-ff-08-ff-export
    Placement: ``.claude/notes/learnable-classical-ff.md`` (no method=)
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
    """

    name: str = ""
    _registry: dict[str, type[BackendAdapter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            BackendAdapter._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> BackendAdapter:
        """Instantiate the registered adapter for ``name`` (e.g. ``"openmm"``)."""
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
        :meth:`~molpot.ir.NonbondedScaling.as_dict` reports nested. Only the
        ``q`` (Coulomb) and ``lj`` (Lennard-Jones) classes are flattened: an
        interaction class a later model declares — ``induced`` for a
        polarizable force field — needs its own key spelling added here and is
        otherwise absent from the exported spec.

        Args:
            scaling: The force field's per-interaction-class scales, or
                ``None`` when the IR carries none.

        Returns:
            ``{"scale_q_12", "scale_q_13", "scale_q_14", "scale_lj_12",
            "scale_lj_13", "scale_lj_14"}`` → dimensionless factor, where
            ``0`` removes the pair and ``1`` keeps it in full. ``None`` when
            ``scaling`` is ``None``, which a backend must treat as "no
            exclusion policy declared", not as "no exclusions".
        """
        if scaling is None:
            return None
        q12, q13, q14 = scaling.triple("q")
        lj12, lj13, lj14 = scaling.triple("lj")
        return {
            "scale_q_12": q12,
            "scale_q_13": q13,
            "scale_q_14": q14,
            "scale_lj_12": lj12,
            "scale_lj_13": lj13,
            "scale_lj_14": lj14,
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
