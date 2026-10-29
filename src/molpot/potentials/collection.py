"""``Potentials`` — the torch twin of molpy's ``Potentials``.

molpy's ``Potentials`` is the composite of the one ``Potential`` concept and
is itself a potential; this is the same concept with autograd: a
``ModuleDict`` of molpot potentials, keyed by style, whose energy is the sum of
its members'.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from molpy.ff.forcefield import Style
from tensordict import TensorDictBase

from molpot.potentials.registry import registered, resolve

__all__ = ["Potentials"]


class Potentials(nn.ModuleDict):
    """One force field's energy: the sum of its styles.

    Keyed by style (``"bond:harmonic"``, ``"pair:lj/cut"``, …). A potential
    depends on the force field, never the reverse: the one way to build this
    from a declaration is :meth:`from_forcefield`, and that call returns this
    one object.

    Example::

        pots = Potentials.from_forcefield(ff)
        energy = pots(frame)                                  # carried tables
        energy = pots(frame, {"bond:harmonic": {"k": k, "r0": r0}, ...})
        energy = pots(frame, params, styles=("bond:harmonic",))
    """

    @classmethod
    def from_forcefield(cls, forcefield: Any) -> "Potentials":
        """One ``Potentials`` for a molpy ``ForceField``.

        The return value is that one object. Each non-atom style is a member
        of it, built by that style's ``from_style``; a style that declares
        types carries their parameters, and the others take theirs per call.
        ``atom`` styles price nothing and are skipped.

        Args:
            forcefield: A molpy ``ForceField`` — the force-field IR, owned by
                molrs; there is no second description of a force field here.

        Returns:
            The one collection, keyed ``"category:name"`` in declaration order.

        Raises:
            ValueError: A declared style has no registered potential.
        """
        members: dict[str, nn.Module] = {}
        for style in forcefield.get_styles(Style):
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
        *,
        styles: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Energy per graph ``(B,)``.

        Args:
            frame: The batch every member reads.
            params: Per style, that member's parameters; a member missing here
                uses its carried table.
            styles: The members to price, or ``None`` for every member. A
                fixed tuple, so a traced call stays straight-line code.

        Raises:
            ValueError: The collection is empty, ``params`` names a style it
                does not hold, or ``styles`` is empty or names one it does
                not hold.
        """
        if not len(self):
            raise ValueError("an empty Potentials prices nothing")
        params = {} if params is None else params
        unknown = [key for key in params if key not in self]
        if unknown:
            raise ValueError(f"params for {unknown}, which this Potentials does not hold")
        if styles is None:
            chosen = tuple(self.keys())
        else:
            chosen = tuple(styles)
            missing = [key for key in chosen if key not in self]
            if not chosen or missing:
                raise ValueError(
                    f"styles names {missing or 'nothing'}; this Potentials holds "
                    f"{tuple(self.keys())}"
                )
        total: torch.Tensor | None = None
        for key in chosen:
            energy = self[key](frame, **params.get(key, {}))
            total = energy if total is None else total + energy
        assert total is not None
        return total
