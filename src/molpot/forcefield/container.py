"""ForceField — molpy ForceField with torch param tables and a torch override.

This **is** a :class:`molpy.ForceField`. Styles and native types go through
molpy; learnable continuous tables live on the instance. ``to_potentials()``
overrides the molrs CPU kernels with :class:`Potentials` (torch).
IO is :mod:`molpy.io` — pass this object to the writers after :meth:`realize`.
"""

from __future__ import annotations

import math
from typing import Any, Self

import torch
from molpy import ForceField as MolpyForceField

from molpot.forcefield.potentials import Potentials
from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    PotentialIR,
    ProperTorsionBag,
)
from molpot.ir.scaling import NonbondedScaling

__all__ = ["ForceField", "ClassIForceField", "OPLSForceField"]


class ForceField(MolpyForceField):
    """Generic molpy ForceField plus torch param tables.

    Concrete forms inherit this type (``ClassIForceField``,
    ``OPLSForceField``) and register their styles in ``__init__``.
    Evaluation::

        pots = ff.to_potentials()   # overrides molpy native → torch Potentials
        E = pots.calc_energy(system)
    """

    def __new__(cls, name: str = "forcefield", units: str = "real", **_kwargs: Any):
        # molrs __new__ only accepts (name, units); Python-only kwargs go to __init__.
        return super().__new__(cls, name, units)

    def __init__(
        self,
        name: str = "forcefield",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(name, units)
        self.nonbonded_scaling = (
            nonbonded_scaling if nonbonded_scaling is not None else NonbondedScaling()
        )
        self._style_options: dict[str, dict[str, Any]] = {}
        self._params: dict[str, dict[str, torch.Tensor]] = {}

    def def_atomstyle(self, name: str = "full") -> Self:
        super().def_atomstyle(name)
        return self

    def def_bondstyle(self, name: str = "harmonic") -> Self:
        super().def_bondstyle(name)
        return self

    def def_anglestyle(self, name: str = "harmonic") -> Self:
        super().def_anglestyle(name)
        return self

    def def_dihedralstyle(self, name: str = "periodic") -> Self:
        super().def_dihedralstyle(name)
        return self

    def def_improperstyle(self, name: str = "harmonic") -> Self:
        super().def_improperstyle(name)
        return self

    def def_pairstyle(self, name: str = "lj/cut", **options: Any) -> Self:
        """Register a pair style; extra options (cutoff, smearing, …) stay on the Potential."""
        super().def_pairstyle(name)
        if options:
            self._style_options[f"pair:{name}"] = dict(options)
        return self

    def style_options(self, key: str) -> dict[str, Any]:
        """Options registered for a style key (e.g. ``pair:lj/cut``)."""
        return dict(self._style_options.get(key, {}))

    def has_style(self, key: str) -> bool:
        return key in self.style_names()

    def bond_style(self) -> str | None:
        return self._first_style("bond")

    def angle_style(self) -> str | None:
        return self._first_style("angle")

    def dihedral_style(self) -> str | None:
        return self._first_style("dihedral")

    def improper_style(self) -> str | None:
        return self._first_style("improper")

    def pair_style(self) -> str | None:
        return self._first_style("pair")

    def pair_styles(self) -> list[str]:
        prefix = "pair:"
        return [k.split(":", 1)[1] for k in self.style_names() if k.startswith(prefix)]

    def has_coulomb(self) -> bool:
        return self.has_style("pair:coulomb")

    def _first_style(self, category: str) -> str | None:
        prefix = f"{category}:"
        for key in self.style_names():
            if key.startswith(prefix):
                return key.split(":", 1)[1]
        return None

    def params(self, category: str) -> dict[str, torch.Tensor] | None:
        return self._params.get(category)

    def clone(self) -> Self:
        # molrs __new__ only accepts (name, units); set Python fields after.
        # Subclass __init__ may already register styles — skip those.
        out = type(self)(self.name, self.units)
        out.nonbonded_scaling = self.nonbonded_scaling
        for key in self.style_names():
            if out.has_style(key):
                continue
            cat, _, sname = key.partition(":")
            getattr(out, f"def_{cat}style")(sname)
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
    ) -> Self:
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

    def to_potentials(self, frame: Any = None, **kwargs: Any) -> Potentials:
        """Override molpy native kernels with torch :class:`Potentials`."""
        del frame
        return Potentials(self, **kwargs)

    def realize(
        self,
        system: Any,
        *,
        masses: torch.Tensor | list[float] | None = None,
    ) -> Self:
        """Fill native molpy type tables from torch params + topology.

        Identity types: one atom type per atom, one bonded type per
        interaction. After this, :mod:`molpy.io` can write ``self`` directly.
        """
        pos = self._system_tensor(system, "atoms", "pos")
        n_atoms = int(pos.shape[0])
        mass_list = self._mass_list(masses, n_atoms)
        charges = self._charge_list(system, n_atoms)
        astyle = self.get_style("atom", "full")
        if astyle is None:
            self.def_atomstyle("full")
            astyle = self.get_style("atom", "full")
        atom_types = []
        atoms = self.params("atoms")
        for i in range(n_atoms):
            kwargs: dict[str, Any] = {"mass": mass_list[i]}
            if charges is not None:
                kwargs["charge"] = charges[i]
            if atoms is not None and "epsilon" in atoms and "sigma" in atoms:
                kwargs["epsilon"] = float(atoms["epsilon"][i].item())
                kwargs["sigma"] = float(atoms["sigma"][i].item())
            atom_types.append(astyle.def_type(str(i + 1), **kwargs))
        self._realize_bonds(system, atom_types)
        self._realize_angles(system, atom_types)
        self._realize_propers(system, atom_types)
        self._realize_pairs(atom_types)
        return self

    def to_ir(self) -> PotentialIR:
        """Materialize Class-I :class:`~molpot.ir.PotentialIR` bags from params."""
        bonds_bag = None
        bp = self.params("bonds")
        if bp is not None and "k" in bp and "r0" in bp:
            bonds_bag = BondBag(k=bp["k"], r0=bp["r0"])

        angles_bag = None
        ap = self.params("angles")
        if ap is not None and "k" in ap and "theta0" in ap:
            angles_bag = AngleBag(k=ap["k"], theta0=ap["theta0"])

        propers_bag = None
        pp = self.params("propers")
        if pp is not None:
            propers_bag = self._propers_to_ir(pp)

        impropers_periodic = None
        impropers_harmonic = None
        ip = self.params("impropers")
        if ip is not None:
            if "chi0" in ip:
                impropers_harmonic = ImproperHarmonicBag(k=ip["k"], chi0=ip["chi0"])
            elif "periodicity" in ip:
                impropers_periodic = ImproperPeriodicBag(
                    k=ip["k"],
                    periodicity=ip["periodicity"],
                    phase=ip["phase"],
                    idivf=ip["idivf"],
                )

        lj_bag = None
        charge_bag = None
        atoms = self.params("atoms")
        if atoms is not None:
            if "epsilon" in atoms and "sigma" in atoms:
                lj_bag = LJBag(epsilon=atoms["epsilon"], sigma=atoms["sigma"])
            q = atoms.get("q")
            if q is not None:
                charge_bag = ChargeBag(q=q)

        return PotentialIR(
            bonds=bonds_bag,
            angles=angles_bag,
            propers=propers_bag,
            impropers_periodic=impropers_periodic,
            impropers_harmonic=impropers_harmonic,
            lj=lj_bag,
            charges=charge_bag,
            scaling=self.nonbonded_scaling,
            unit_system="class_i_canonical",
        )

    def _propers_to_ir(self, pp: dict[str, torch.Tensor]) -> ProperTorsionBag:
        if "k1" in pp:
            k1, k2, k3, k4 = pp["k1"], pp["k2"], pp["k3"], pp["k4"]
            n = k1.shape[0]
            device, dtype = k1.device, k1.dtype
            k = 0.5 * torch.stack([k1, k2, k3, k4], dim=-1)
            phase = torch.zeros(n, 4, device=device, dtype=dtype)
            phase[:, 1] = math.pi
            phase[:, 3] = math.pi
            periodicity = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
            idivf = pp.get("idivf", torch.ones(n, device=device, dtype=dtype))
            return ProperTorsionBag(k=k, periodicity=periodicity, phase=phase, idivf=idivf)

        pp = ProperTorsionBag._cosine_fields(pp)
        if "k" not in pp:
            raise ValueError("propers params need 'k', cartesian 'a'/'b', or OPLS 'k1..k4'")
        idivf = pp.get("idivf")
        if idivf is None:
            idivf = torch.ones(pp["k"].shape[0], device=pp["k"].device, dtype=pp["k"].dtype)
        return ProperTorsionBag(
            k=pp["k"],
            periodicity=pp["periodicity"],
            phase=pp["phase"],
            idivf=idivf,
        )

    def _realize_bonds(self, system: Any, atom_types: list[Any]) -> None:
        if self.bond_style() != "harmonic":
            return
        bp = self.params("bonds")
        bonds = self._arity(system, "bonds", 2)
        if bp is None or not bonds:
            return
        bstyle = self.get_style("bond", "harmonic")
        for n, (i, j) in enumerate(bonds):
            bstyle.def_type(
                atom_types[i],
                atom_types[j],
                k=float(bp["k"][n].item()),
                r0=float(bp["r0"][n].item()),
            )

    def _realize_angles(self, system: Any, atom_types: list[Any]) -> None:
        if self.angle_style() != "harmonic":
            return
        ap = self.params("angles")
        angles = self._arity(system, "angles", 3)
        if ap is None or not angles:
            return
        astyle = self.get_style("angle", "harmonic")
        for n, (i, j, k) in enumerate(angles):
            astyle.def_type(
                atom_types[i],
                atom_types[j],
                atom_types[k],
                k=float(ap["k"][n].item()),
                theta0=float(ap["theta0"][n].item()),
            )

    def _realize_propers(self, system: Any, atom_types: list[Any]) -> None:
        style = self.dihedral_style()
        if style is None:
            return
        pp = self.params("propers")
        propers = self._arity(system, "propers", 4)
        if pp is None or not propers:
            return
        dstyle = self.get_style("dihedral", style)
        for n, (i, j, k, l) in enumerate(propers):
            if "k1" in pp:
                dstyle.def_type(
                    atom_types[i],
                    atom_types[j],
                    atom_types[k],
                    atom_types[l],
                    c1=float(pp["k1"][n].item()),
                    c2=float(pp["k2"][n].item()),
                    c3=float(pp["k3"][n].item()),
                    c4=float(pp["k4"][n].item()),
                )
            else:
                dstyle.def_type(
                    atom_types[i],
                    atom_types[j],
                    atom_types[k],
                    atom_types[l],
                    k=float(pp["k"][n].reshape(-1)[0].item())
                    if pp["k"].ndim == 1
                    else float(pp["k"][n, 0].item()),
                )

    def _realize_pairs(self, atom_types: list[Any]) -> None:
        if "lj/cut" not in self.pair_styles():
            return
        atoms = self.params("atoms")
        if atoms is None or "epsilon" not in atoms:
            return
        pstyle = self.get_style("pair", "lj/cut")
        for i, at in enumerate(atom_types):
            pstyle.def_type(
                at,
                at,
                epsilon=float(atoms["epsilon"][i].item()),
                sigma=float(atoms["sigma"][i].item()),
            )

    @staticmethod
    def _system_tensor(system: Any, *keys: str) -> torch.Tensor:
        value = system[keys] if len(keys) > 1 else system[keys[0]]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"system{list(keys)} must be a tensor, got {type(value)!r}")
        return value

    @staticmethod
    def _arity(system: Any, ns: str, arity: int) -> list[tuple[int, ...]]:
        if ns not in system:
            return []
        block = system[ns]
        cols = ("atomi", "atomj", "atomk", "atoml")[:arity]
        missing = [c for c in cols if c not in block]
        if missing:
            raise KeyError(f"system[{ns!r}] missing columns {missing}")
        arrays = [block[c].detach().cpu().long().reshape(-1) for c in cols]
        n = int(arrays[0].shape[0])
        return [tuple(int(a[i]) for a in arrays) for i in range(n)]

    @staticmethod
    def _mass_list(masses: torch.Tensor | list[float] | None, n_atoms: int) -> list[float]:
        if masses is None:
            return [12.011] * n_atoms
        mass_t = torch.as_tensor(masses, dtype=torch.float64).reshape(-1)
        if mass_t.numel() != n_atoms:
            raise ValueError(f"masses length {mass_t.numel()} != n_atoms {n_atoms}")
        return [float(x) for x in mass_t.tolist()]

    def _charge_list(self, system: Any, n_atoms: int) -> list[float] | None:
        atoms = self.params("atoms")
        if atoms is not None and "q" in atoms:
            return [float(x) for x in atoms["q"].detach().cpu().reshape(-1).tolist()]
        if "atoms" in system and "q" in system["atoms"]:
            q = system["atoms"]["q"]
            return [float(x) for x in q.detach().cpu().reshape(-1).tolist()]
        return None

    def __repr__(self) -> str:
        styles = ", ".join(self.style_names()) or "(no styles)"
        filled = ", ".join(self._params.keys()) or "(no params)"
        return (
            f"{type(self).__name__}(name={self.name!r}, units={self.units!r}, "
            f"styles=[{styles}], params=[{filled}])"
        )


class ClassIForceField(ForceField):
    """Class-I bonded form: harmonic bond/angle + multi-term dihedral.

    Pair / Coulomb styles are optional extras — register them with
    :meth:`def_pairstyle` after construction. Subclasses pick the
    dihedral form via ``_dihedral_style``.
    """

    _dihedral_style = "fourier"

    def __init__(
        self,
        name: str = "class-i",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
    ) -> None:
        super().__init__(name, units, nonbonded_scaling=nonbonded_scaling)
        self.def_atomstyle("full")
        self.def_bondstyle("harmonic")
        self.def_anglestyle("harmonic")
        self.def_dihedralstyle(type(self)._dihedral_style)


class OPLSForceField(ClassIForceField):
    """Class-I with OPLS proper torsion (``k1..k4``)."""

    _dihedral_style = "opls"

    def __init__(
        self,
        name: str = "OPLS",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
    ) -> None:
        super().__init__(name, units, nonbonded_scaling=nonbonded_scaling)
