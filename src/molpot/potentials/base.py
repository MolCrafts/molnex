"""Base class for molpot PyTorch potentials.

All molpot potentials inherit from BasePotential, which provides:
- PyTorch nn.Module functionality
- PotentialProtocol compliance (calc_energy, calc_forces)
- Automatic force computation via :class:`~molpot.derivation.ForceDerivation`
  (autograd backend — universal for cuEq and pure-torch potentials)
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from molpot.derivation.force import ForceDerivation


class BasePotential(nn.Module, ABC):
    """Base class for all molpot PyTorch potentials.

    Implements PotentialProtocol for compatibility with molpy ForceField.
    All molpot potentials (classic and ML) inherit from this class.

    Attributes:
        name: Potential name for registration (e.g., "lj126_torch", "pinet")
        type: Potential type for categorization (e.g., "pair", "bond", "ml")
    """

    name: str = "base"
    type: str = "unknown"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Universal protocol path: autograd works for every potential,
        # including cuEquivariance-fused ones.
        self._force_derivation = ForceDerivation()

    def table(self, name: str, overrides: Mapping[str, Any]) -> torch.Tensor:
        """This call's parameter table for *name* — the caller's, else the registered one.

        A bonded term carries its tables as buffers, which is what a **fixed**
        force field is: an exported OPLS-AA field looks its ``k`` up once and
        never changes it. A **learned** field is the other case — its tables are
        predicted per batch, and they have to be computed *inside* the compiled
        region or the energy is not traceable: a table produced outside carries
        a ``grad_fn`` that ``torch.autograd.grad`` refuses to trace through
        ("autograd.grad with external grad_fn"), and putting it on the frame
        instead of on the module does not change that — measured 2026-09-11,
        ``.scratch/classical_train_compile/where_tables_enter.py``.

        So every bonded term accepts its tables per call, and the buffers stay
        as the default. One named lookup rather than the same two lines in each
        of the six terms.

        Args:
            name: Parameter name, e.g. ``"k"`` / ``"r0"``.
            overrides: The call's keyword arguments. A ``None`` entry (or an
                absent one) means "use the registered table".

        Returns:
            The table to price this call with.

        Raises:
            AttributeError: Neither the caller nor the module has it.
        """
        value = overrides.get(name)
        if value is not None:
            return value
        registered = getattr(self, name, None)
        if registered is None:
            raise AttributeError(
                f"{type(self).__name__} has no parameter {name!r}: it was neither "
                f"passed to this call nor registered at construction"
            )
        return registered

    def rows(
        self,
        name: str,
        overrides: Mapping[str, Any],
        types: torch.Tensor | None,
    ) -> torch.Tensor:
        """This call's table *per interaction*: gathered by *types*, or used as-is.

        ``types is None`` states that the table already carries one row per
        interaction — which is what a **learned** field produces, since its
        head predicts a row for every bond/angle/torsion it was handed. The
        gather would then be ``table[arange(n)]``: a copy that returns the
        table unchanged, and whose backward is an
        ``index_put_(accumulate=True)`` scattering with atomics.

        That backward was **77 % of all device time** on a production-shaped
        Class-I step (profile 2026-09-16, medium tier: ``_index_put_impl_``
        32.2 s and the three ``indexing_backward_kernel`` variants 33.5 s of
        84.9 s total). A fixed, typed field still passes real type ids and
        still gathers — this only removes the identity case.

        Args:
            name: Parameter name, e.g. ``"k"`` / ``"r0"``.
            overrides: The call's keyword arguments.
            types: Per-interaction type index, or ``None`` when the table is
                already per-interaction.

        Returns:
            The table with one row per interaction.
        """
        table = self.table(name, overrides)
        return table if types is None else table[types]

    def calc_energy(self, data=None, **kwargs: Any) -> float:
        """Calculate energy (PotentialProtocol method).

        This method provides compatibility with molpy's Potential interface.
        It calls forward() and converts the result to a Python float.
        """
        energy_tensor = self.forward(data, **kwargs)
        return float(energy_tensor.item())

    def calc_forces(
        self, data=None, *, as_numpy: bool = True, **kwargs: Any
    ) -> np.ndarray | torch.Tensor:
        """Calculate forces (PotentialProtocol method).

        Computes forces as ``F = -∂E/∂x`` via
        :class:`~molpot.derivation.ForceDerivation` (autograd backend). Positions
        are passed back through ``pos=`` so :meth:`forward` evaluates at the
        candidate geometry (``_get_positions`` resolves ``kwargs["pos"]`` /
        ``atoms.pos`` / ``pos`` first).

        Args:
            data: Optional molecule dict / TensorDict.
            as_numpy: If ``True`` (default, molpy protocol), return a host
                ``numpy`` array. Set ``False`` to keep a device ``Tensor`` for
                in-graph / MD hot paths.
            **kwargs: Forwarded to :meth:`forward`; may include ``pos=``.

        Returns:
            Forces ``(N, 3)`` as ``ndarray`` or ``Tensor``.
        """
        pos = self._get_positions(data, **kwargs).detach()

        def energy_fn(p: torch.Tensor) -> torch.Tensor:
            return self.forward(data, **{**kwargs, "pos": p}).sum()

        forces = self._force_derivation(energy_fn, pos)
        if as_numpy:
            return forces.detach().cpu().numpy()
        return forces

    @abstractmethod
    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses.

        Args:
            data: Optional dictionary with molecule fields
            **kwargs: Alternate way to pass explicit tensors such as positions
                or atom types.

        Returns:
            Energy as torch.Tensor (scalar)
        """
        pass

    def _get_positions(self, data=None, **kwargs: Any) -> torch.Tensor:
        """Extract positions from kwargs or data (``pos`` / ``atoms.pos``).

        Preferred keys (in order): ``kwargs["pos"]``, ``data["pos"]``,
        nested ``data["atoms"]["pos"]`` (TensorDict or dict). The legacy
        ``atoms.x`` / ``data["x"]`` aliases are rejected with a clear error.
        """
        pos = kwargs.get("pos")
        if pos is None and data is not None:
            if isinstance(data, dict) or hasattr(data, "get"):
                pos = data.get("pos") if hasattr(data, "get") else None
                if pos is None:
                    try:
                        atoms = data["atoms"]
                        if hasattr(atoms, "get"):
                            pos = atoms.get("pos")
                        elif isinstance(atoms, dict):
                            pos = atoms.get("pos")
                    except (KeyError, TypeError):
                        pos = None
                if pos is None:
                    # Explicitly reject legacy x aliases so callers fix the key.
                    has_x = False
                    try:
                        if isinstance(data, dict) and "x" in data:
                            has_x = True
                        elif "atoms" in data and (
                            (hasattr(data["atoms"], "keys") and "x" in data["atoms"].keys())
                            or (isinstance(data["atoms"], dict) and "x" in data["atoms"])
                        ):
                            has_x = True
                    except (KeyError, TypeError):
                        pass
                    if has_x:
                        raise ValueError(
                            "Found positions under legacy key 'x' / 'atoms.x'. "
                            "Use 'pos' or 'atoms.pos' per the molnex batch contract."
                        )

        if pos is None:
            raise ValueError(
                "Could not extract positions from data or kwargs "
                "(expected kwargs['pos'], data['pos'], or data['atoms']['pos'])."
            )

        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos).float()

        return pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.type}')"
