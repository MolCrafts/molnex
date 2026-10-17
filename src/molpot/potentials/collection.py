"""``Potentials`` — the torch twin of molpy's ``Potentials``.

molpy's ``Potentials`` is the composite of the one ``Potential`` concept and
is itself a potential; this is the same concept with autograd: a
``ModuleDict`` of molpot potentials, keyed by style, whose energy is the sum of
its members'.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from molpot.potentials.registry import registered, resolve

__all__ = ["Potentials"]


class Potentials(nn.ModuleDict):
    """A sum of potentials, keyed by style (``"bond:harmonic"``, ``"pair:lj/cut"``, …).

    A potential depends on the force field, never the reverse: the one way to
    build this collection from a declaration is :meth:`from_forcefield`.

    Example::

        pots = Potentials.from_forcefield(ff)
        energy = pots(frame)                                  # carried tables
        energy = pots(frame, {"bond:harmonic": {"k": k, "r0": r0}, ...})
    """

    @classmethod
    def from_forcefield(cls, forcefield: Any) -> "Potentials":
        """The potentials of every style a molpy ``ForceField`` declares.

        Each style becomes its registered potential, built by that class's
        ``from_style``; styles that declare types carry their parameters, the
        others take theirs per call. ``atom`` styles price nothing and are
        skipped.

        Args:
            forcefield: A molpy ``ForceField`` — the force-field IR, owned by
                molrs; there is no second description of a force field here.

        Returns:
            The collection, keyed ``"category:name"`` in declaration order.

        Raises:
            ValueError: A declared style has no registered potential.
        """
        members: dict[str, nn.Module] = {}
        for style in forcefield.styles:
            if style.category == "atom":
                continue
            key = f"{style.category}:{style.name}"
            potential = resolve(key)
            if potential is None:
                raise ValueError(
                    f"no molpot potential implements {key!r}; registered: {registered()}"
                )
            members[key] = potential.from_style(style, forcefield=forcefield)
        return cls(members)

    def forward(
        self,
        frame: TensorDictBase,
        params: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Energy per graph ``(B,)``.

        Args:
            frame: The batch every member reads.
            params: Per style, that member's parameters; a member missing here
                uses its carried table.

        Raises:
            ValueError: The collection is empty, or ``params`` names a style
                it does not hold.
        """
        if not len(self):
            raise ValueError("an empty Potentials prices nothing")
        params = params or {}
        unknown = [key for key in params if key not in self]
        if unknown:
            raise ValueError(f"params for {unknown}, which this Potentials does not hold")
        total: torch.Tensor | None = None
        for key, potential in self.items():
            energy = potential(frame, **params.get(key, {}))
            total = energy if total is None else total + energy
        assert total is not None
        return total
