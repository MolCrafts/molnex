"""ForceFieldCompiler — Potential IR → backend force-spec.

Single primitive: :meth:`ForceFieldCompiler.compile`. Backend selection is by
peer :class:`BackendAdapter` type (or registered name), never
``compile(method=…)``.

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-08-ff-export
"""

from __future__ import annotations

from typing import Any

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.force_spec import ForceSpec
from molix.ff_export.openmm_adapter import OpenMMAdapter as _OpenMMAdapter  # noqa: F401
from molpot.ir import PotentialIR

__all__ = ["ForceFieldCompiler"]


class ForceFieldCompiler:
    """Compile Class-I :class:`~molpot.ir.PotentialIR` into a :class:`ForceSpec`.

    Args:
        adapter: A :class:`BackendAdapter` instance, or a registered name
            (default ``"openmm"``).

    Example:
        >>> from molix.ff_export import ForceFieldCompiler
        >>> from molpot.ir import BondBag, PotentialIR
        >>> import torch
        >>> bonds = BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.5]))
        >>> spec = ForceFieldCompiler("openmm").compile(PotentialIR(bonds=bonds))
        >>> spec.forces[0]["parameters"][0]["k"]
        41840.0
    """

    def __init__(self, adapter: BackendAdapter | str = "openmm") -> None:
        if isinstance(adapter, str):
            self.adapter: BackendAdapter = BackendAdapter.from_name(adapter)
        elif isinstance(adapter, BackendAdapter):
            self.adapter = adapter
        else:
            raise TypeError(f"adapter must be BackendAdapter or str name, got {type(adapter)!r}")

    def compile(
        self,
        ir: PotentialIR,
        *,
        type_systems: Any = None,
        symbolic: Any = None,
    ) -> ForceSpec:
        """Translate ``ir`` through the configured backend adapter.

        Args:
            ir: Class-I potential intermediate representation.
            type_systems: Optional type-system metadata (passed through to
                force-spec ``metadata``).
            symbolic: Optional symbolic force-field metadata (passed through).

        Returns:
            Backend force specification (OpenMM units when using
            :class:`~molix.ff_export.OpenMMAdapter`).

        Raises:
            UnsupportedTermError: When the adapter cannot map an IR bag.
        """
        meta: dict[str, Any] = {}
        if type_systems is not None:
            meta["type_systems"] = type_systems
        if symbolic is not None:
            meta["symbolic"] = symbolic
        return self.adapter.translate(ir, meta=meta or None)
