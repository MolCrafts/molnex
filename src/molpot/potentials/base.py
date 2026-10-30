"""Base class for the term kernels of molpot.

A :class:`BasePotential` is an ``nn.Module`` that prices one interaction term
from keyword-only typed tensors — positions, the term's index and type columns,
the per-interaction graph index with the graph count, and its parameter
tables — and returns the energy. It derives nothing: forces, virials and
Hessians come from :class:`molix.autograd.Derivative` over the container that
owns the term (:class:`molpot.forcefield.Potentials`), never from the term.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class BasePotential(nn.Module, ABC):
    """Base class for the term kernels under :mod:`molpot.potentials`.

    Attributes:
        name: Potential name for registration (e.g., "bond_harmonic_torch")
        category: What the potential prices (e.g., "bond", "long_range"). Not
            ``type``, which would shadow :meth:`torch.nn.Module.type`.
    """

    name: str = "base"
    category: str = "unknown"

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
            overrides: The call's table arguments. A ``None`` entry (or an
                absent one) means "use the registered table".

        Returns:
            The table to price this call with.

        Raises:
            AttributeError: Neither the caller nor the module has it.
        """
        value = overrides.get(name)
        if value is not None:
            return value
        try:
            return self.get_buffer(name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no parameter {name!r}: it was neither "
                f"passed to this call nor registered at construction"
            ) from None

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
            overrides: The call's table arguments.
            types: Per-interaction type index, or ``None`` when the table is
                already per-interaction.

        Returns:
            The table with one row per interaction.
        """
        table = self.table(name, overrides)
        return table if types is None else table[types]

    @staticmethod
    def per_graph(
        energy: torch.Tensor,
        term_batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        """Reduce per-interaction energies to the term's energy.

        Args:
            energy: One energy per interaction ``(M,)``.
            term_batch: Graph index per interaction ``(M,)``; ``None`` sums
                every interaction into one scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.

        Returns:
            ``(B,)`` when ``term_batch`` is given, else a scalar.
        """
        if term_batch is None:
            return energy.sum()
        return energy.new_zeros(num_graphs).index_add_(0, term_batch, energy)

    @abstractmethod
    def forward(
        self, **inputs: torch.Tensor | int | None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Price the term from keyword-only typed inputs.

        Each subclass declares its exact keyword-only parameters — there is no
        positional input and no batch dict. A bonded term takes ``pos``, its
        ``<term>_index`` / ``<term>_types`` columns, ``term_batch`` with
        ``num_graphs`` and its parameter tables, and returns a scalar or a
        ``(B,)`` energy; :class:`~molpot.potentials.elec.ewald_multipole.EwaldMultipoleEnergy`
        returns its energy beside the per-atom potential and field.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"
