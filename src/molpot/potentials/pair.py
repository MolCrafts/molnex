"""Pair styles, as molrs prices them, over the cutoff pair list.

References:
    LAMMPS ``pair_style lj/cut``, https://docs.lammps.org/pair_lj.html
    LAMMPS ``pair_style coul/cut``, https://docs.lammps.org/pair_coul.html
    molrs ``molrs-python/docs/guides/forcefield-conventions.md`` §Pair styles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tensordict import TensorDictBase

from molpot.potentials.potential import PairPotential
from molpot.potentials.registry import register
from molpot.potentials.vdw.lj126 import MixingRule

__all__ = ["LJCut", "PairCoulCut"]

#: molrs ``mixing`` spellings → the rule that implements them. molrs prices an
#: undeclared rule as ``arithmetic`` (Lorentz-Berthelot).
_MIXING = {
    "arithmetic": MixingRule.LORENTZ_BERTHELOT,
    "geometric": MixingRule.GEOMETRIC,
}


@register
class LJCut(PairPotential):
    """``pair lj/cut``: ``w · 4 εᵢⱼ [(σᵢⱼ/r)¹² − (σᵢⱼ/r)⁶]`` for ``r < cutoff``.

    ``w`` is the ``special_bonds`` LJ weight of the pair's relation; ``εᵢⱼ`` /
    ``σᵢⱼ`` are mixed from per-atom ``epsilon`` (energy) / ``sigma`` (Å) by the
    style's ``mixing``. Hard truncation, no shift.

    Args:
        cutoff: Radius, Å.
        special: ``(w12, w13, w14)`` LJ weights.
        mixing: ``"arithmetic"`` (Lorentz-Berthelot) or ``"geometric"``.
        table: Per atom type ``epsilon`` / ``sigma`` ``(n_types,)``, or ``None``.
    """

    style = "pair:lj/cut"
    params = ("epsilon", "sigma")

    def __init__(
        self,
        *,
        cutoff: float,
        special: Sequence[float],
        mixing: str = "arithmetic",
        table: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(cutoff=cutoff, special=special)
        if mixing not in _MIXING:
            raise ValueError(f"pair:lj/cut: mixing {mixing!r} is not one of {sorted(_MIXING)}")
        self.mixing = _MIXING[mixing]
        self.has_table = table is not None
        if table is not None:
            self.register_buffer("table_epsilon", torch.as_tensor(table["epsilon"]))
            self.register_buffer("table_sigma", torch.as_tensor(table["sigma"]))

    @classmethod
    def from_style(cls, style: Any, *, forcefield: Any) -> LJCut:
        """The potential of a molpy ``pair lj/cut`` style.

        Raises:
            ValueError: The style declares no cutoff, an ``n``/``m`` other than
                12/6 or a ``shift``, or a cross row (NBFIX), none of which this
                potential prices.
        """
        params = dict(style.params)
        if "cutoff" not in params:
            raise ValueError("pair:lj/cut: the style declares no cutoff")
        if float(params.get("n", 12)) != 12 or float(params.get("m", 6)) != 6:
            raise ValueError("pair:lj/cut: only n = 12, m = 6 is priced")
        if params.get("shift") not in (None, 0, 0.0, "no", False):
            raise ValueError("pair:lj/cut: a shifted potential is not priced")
        lj, _coul = forcefield.special_bonds
        return cls(
            cutoff=float(params["cutoff"]),
            special=[float(w) for w in lj],
            mixing=str(params.get("mixing", "arithmetic")),
            table=cls._table(style, forcefield),
        )

    @staticmethod
    def _table(style: Any, forcefield: Any) -> dict[str, torch.Tensor] | None:
        rows = list(style.get_types())
        if not rows:
            return None
        by_type: dict[str, tuple[float, float]] = {}
        for row in rows:
            itom, jtom = row.itom, row.jtom
            if jtom is not None and jtom.name != itom.name:
                raise ValueError(
                    f"pair:lj/cut: cross row {row.name!r} ({itom.name}-{jtom.name}) is not priced"
                )
            by_type[itom.name] = (float(row["epsilon"]), float(row["sigma"]))
        atom_types = [t.name for t in forcefield.get_types("atom")]
        missing = [name for name in atom_types if name not in by_type]
        if missing:
            raise ValueError(f"pair:lj/cut: atom types {missing} have no self row")
        eps, sig = zip(*(by_type[name] for name in atom_types), strict=True)
        return {
            "epsilon": torch.tensor(eps, dtype=torch.float64),
            "sigma": torch.tensor(sig, dtype=torch.float64),
        }

    def forward(self, frame: TensorDictBase, **params: torch.Tensor) -> torch.Tensor:
        """Energy per graph ``(B,)``; ``epsilon`` / ``sigma`` per atom, else the table."""
        pair_index, r, weight = self.geometry(frame)
        if params:
            epsilon, sigma = params["epsilon"], params["sigma"]
        elif self.has_table:
            type_id = frame["atoms", "type"].long()
            epsilon, sigma = self.table_epsilon[type_id], self.table_sigma[type_id]
        else:
            raise ValueError("pair:lj/cut: no epsilon/sigma were passed and no types declared")
        mixed = self.mixing.mix(
            {"epsilon": epsilon.to(r.dtype), "sigma": sigma.to(r.dtype)}, pair_index
        )
        sr6 = (mixed["sigma_ij"] / r).pow(6)
        per_pair = weight * 4.0 * mixed["epsilon_ij"] * (sr6 * sr6 - sr6)
        return self.reduce(frame, pair_index, per_pair)


@register
class PairCoulCut(PairPotential):
    """``pair coul/cut``: ``w · coulomb · qᵢqⱼ / (dielectric · (r + delta))`` for ``r < cutoff``.

    ``w`` is the ``special_bonds`` Coulomb weight of the pair's relation.
    ``coulomb`` and ``dielectric`` are the force field's — required, as molrs
    requires them: a default would be the kernel speaking for the force field.

    Args:
        cutoff: Radius, Å (``inf`` for none).
        special: ``(w12, w13, w14)`` Coulomb weights.
        coulomb: Coulomb constant, energy·Å/e².
        dielectric: Relative permittivity.
        delta: Distance buffer, Å.
    """

    style = "pair:coul/cut"
    params = ("charge",)

    def __init__(
        self,
        *,
        cutoff: float,
        special: Sequence[float],
        coulomb: float,
        dielectric: float,
        delta: float = 0.0,
    ) -> None:
        super().__init__(cutoff=cutoff, special=special)
        self.coulomb = float(coulomb)
        self.dielectric = float(dielectric)
        self.delta = float(delta)

    @classmethod
    def from_style(cls, style: Any, *, forcefield: Any) -> PairCoulCut:
        """The potential of a molpy ``pair coul/cut`` style.

        Raises:
            ValueError: The style lacks ``coulomb`` or ``dielectric``.
        """
        params = dict(style.params)
        missing = [name for name in ("coulomb", "dielectric") if name not in params]
        if missing:
            raise ValueError(f"pair:coul/cut: the style declares no {missing}")
        _lj, coul = forcefield.special_bonds
        return cls(
            cutoff=float(params.get("cutoff", float("inf"))),
            special=[float(w) for w in coul],
            coulomb=float(params["coulomb"]),
            dielectric=float(params["dielectric"]),
            delta=float(params.get("delta", 0.0)),
        )

    def forward(self, frame: TensorDictBase, **params: torch.Tensor) -> torch.Tensor:
        """Energy per graph ``(B,)``; ``charge`` per atom, else ``atoms.charge``."""
        pair_index, r, weight = self.geometry(frame)
        charge = params["charge"] if params else frame["atoms", "charge"]
        charge = charge.to(r.dtype)
        qq = charge[pair_index[:, 0]] * charge[pair_index[:, 1]]
        per_pair = weight * self.coulomb * qq / (self.dielectric * (r + self.delta))
        return self.reduce(frame, pair_index, per_pair)
