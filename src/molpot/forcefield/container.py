"""ForceField — classical MM parameter container (OOP, single responsibility).

Owns only: styles, type/param tables, ``to_potentials()``.
Does not: IO, training, energy evaluation, projection.
"""

from __future__ import annotations

from typing import Any

import torch

from molpot.forcefield.potentials import Potentials
from molpot.ir.scaling import NonbondedScaling

__all__ = ["ForceField"]


class ForceField:
    """Classical force-field data container (general Class-I or concrete form).

    General Fourier-style Class-I and OPLS/GAFF share this type — same status,
    different specificity of styles/params.

    Evaluation lives on the bound handle::

        pots = ff.to_potentials()
        E = pots.calc_energy(system)
    """

    def __init__(
        self,
        name: str = "forcefield",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
    ) -> None:
        self.name = name
        self.units = units
        self.nonbonded_scaling = (
            nonbonded_scaling if nonbonded_scaling is not None else NonbondedScaling()
        )
        self._styles: dict[str, bool] = {}
        # per-style options owned by the potential form (e.g. cutoff, smearing)
        self._style_options: dict[str, dict[str, Any]] = {}
        self._params: dict[str, dict[str, torch.Tensor]] = {}

    # --- alternate constructors (styles only; still the container) ---

    @classmethod
    def class_i(
        cls,
        *,
        name: str = "class-i",
        units: str = "real",
        dihedral: str = "fourier",
        with_pair: bool = False,
        with_coulomb: bool = False,
    ) -> ForceField:
        """Class-I template: harmonic bond/angle + multi-term dihedral.

        ``dihedral`` is the training form name (``fourier`` / ``periodic`` /
        ``opls``). Pair LJ / Coulomb off unless requested.
        """
        ff = cls(name=name, units=units)
        ff.def_atomstyle("full")
        ff.def_bondstyle("harmonic")
        ff.def_anglestyle("harmonic")
        ff.def_dihedralstyle(dihedral)
        if with_pair:
            ff.def_pairstyle("lj/cut")
        if with_coulomb:
            ff.def_pairstyle("coulomb")
        return ff

    @classmethod
    def opls(
        cls,
        *,
        name: str = "OPLS",
        units: str = "real",
        with_pair: bool = False,
        with_coulomb: bool = False,
    ) -> ForceField:
        """Concrete OPLS-shaped template (harmonic + OPLS dihedral).

        Styles only — no type table. Pair LJ / Coulomb optional.
        """
        return cls.class_i(
            name=name,
            units=units,
            dihedral="opls",
            with_pair=with_pair,
            with_coulomb=with_coulomb,
        )

    # --- style registration ---

    def def_atomstyle(self, name: str = "full") -> ForceField:
        self._styles[f"atom:{name}"] = True
        return self

    def def_bondstyle(self, name: str = "harmonic") -> ForceField:
        self._styles[f"bond:{name}"] = True
        return self

    def def_anglestyle(self, name: str = "harmonic") -> ForceField:
        self._styles[f"angle:{name}"] = True
        return self

    def def_dihedralstyle(self, name: str = "periodic") -> ForceField:
        self._styles[f"dihedral:{name}"] = True
        return self

    def def_improperstyle(self, name: str = "harmonic") -> ForceField:
        self._styles[f"improper:{name}"] = True
        return self

    def def_pairstyle(self, name: str = "lj/cut", **options: Any) -> ForceField:
        """Register a pair style; options (e.g. ``cutoff=``) go on the Potential.

        Examples::

            ff.def_pairstyle("lj/cut", cutoff=12.0)
            ff.def_pairstyle("coulomb")  # open; or with cell → periodic
            ff.def_pairstyle(
                "coulomb",
                calculator="pme",
                smearing=5.0,
                mesh_spacing=1.0,
                real_space_cutoff=12.0,
            )
        """
        key = f"pair:{name}"
        self._styles[key] = True
        if options:
            self._style_options[key] = dict(options)
        return self

    def style_options(self, key: str) -> dict[str, Any]:
        """Options registered for a style key (e.g. ``pair:lj/cut``)."""
        return dict(self._style_options.get(key, {}))

    # --- queries ---

    @property
    def style_names(self) -> list[str]:
        return list(self._styles.keys())

    def has_style(self, key: str) -> bool:
        return key in self._styles

    def bond_style(self) -> str | None:
        return self._first_style("bond")

    def angle_style(self) -> str | None:
        return self._first_style("angle")

    def dihedral_style(self) -> str | None:
        return self._first_style("dihedral")

    def improper_style(self) -> str | None:
        return self._first_style("improper")

    def pair_style(self) -> str | None:
        """First registered pair style (may be one of several)."""
        return self._first_style("pair")

    def pair_styles(self) -> list[str]:
        """All registered pair style names (e.g. ``['lj/cut', 'coulomb']``)."""
        prefix = "pair:"
        return [k.split(":", 1)[1] for k in self._styles if k.startswith(prefix)]

    def has_coulomb(self) -> bool:
        return self.has_style("pair:coulomb")

    def _first_style(self, category: str) -> str | None:
        prefix = f"{category}:"
        for key in self._styles:
            if key.startswith(prefix):
                return key.split(":", 1)[1]
        return None

    def params(self, category: str) -> dict[str, torch.Tensor] | None:
        return self._params.get(category)

    # --- copy-on-write param tables (container mutation API) ---

    def clone(self) -> ForceField:
        out = ForceField(
            name=self.name,
            units=self.units,
            nonbonded_scaling=self.nonbonded_scaling,
        )
        out._styles = dict(self._styles)
        out._style_options = {k: dict(v) for k, v in self._style_options.items()}
        out._params = {cat: dict(p) for cat, p in self._params.items()}
        return out

    def with_params(
        self,
        *,
        bonds: dict[str, torch.Tensor] | None = None,
        angles: dict[str, torch.Tensor] | None = None,
        propers: dict[str, torch.Tensor] | None = None,
        impropers: dict[str, torch.Tensor] | None = None,
        pairs: dict[str, torch.Tensor] | None = None,
        atoms: dict[str, torch.Tensor] | None = None,
    ) -> ForceField:
        """Copy-on-write: return a new container with the given param tables."""
        out = self.clone()
        if bonds is not None:
            out._params["bonds"] = dict(bonds)
        if angles is not None:
            out._params["angles"] = dict(angles)
        if propers is not None:
            out._params["propers"] = dict(propers)
        if impropers is not None:
            out._params["impropers"] = dict(impropers)
        if pairs is not None:
            out._params["pairs"] = dict(pairs)
        if atoms is not None:
            out._params["atoms"] = dict(atoms)
        return out

    def to_potentials(self, **kwargs: Any) -> Potentials:
        """Assemble evaluable :class:`Potentials` from this container's styles.

        Pair cutoffs / electrostatic calculator options come from
        :meth:`def_pairstyle` options on each Potential, not from kwargs here.
        """
        return Potentials(self, **kwargs)

    def __repr__(self) -> str:
        styles = ", ".join(self.style_names) or "(no styles)"
        filled = ", ".join(self._params.keys()) or "(no params)"
        return (
            f"ForceField(name={self.name!r}, units={self.units!r}, "
            f"styles=[{styles}], params=[{filled}])"
        )
