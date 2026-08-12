"""Potentials — assembled evaluable handle for one ForceField.

Pair cutoffs and electrostatic calculator options live on the **Potential
terms** (see :mod:`molpot.forcefield.pair_terms`), not on this container.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from molpot.forcefield.neighbor_list import NeighborListBuilder
from molpot.forcefield.pair_terms import (
    CoulombOpenPotential,
    CoulombPeriodicPotential,
    LJCutPotential,
)
from molpot.forcefield.schema import ForceFieldSchemaView, PotentialMeta
from molpot.ir.scaling import NonbondedScaling
from molpot.potentials import (
    AngleHarmonic,
    BondHarmonic,
    ImproperHarmonic,
    ImproperPeriodic,
    ProperTorsionPeriodic,
)
from molpot.potentials.elec.prefactors import kcalmol_A

if TYPE_CHECKING:
    from molpot.forcefield.container import ForceField

__all__ = ["Potentials"]


class Potentials:
    """Bound classical potentials assembled from a :class:`ForceField`.

    Bonded terms use existing harmonic / torsion kernels. Nonbonded terms are
    concrete Potential objects that own their cutoffs:

    * ``pair:lj/cut`` → :class:`LJCutPotential` (``cutoff`` on the Potential)
    * ``pair:coulomb`` open → :class:`CoulombOpenPotential`
    * ``pair:coulomb`` + cell → :class:`CoulombPeriodicPotential`
      (``calculator=ewald|pme|p3m`` on the Potential)

    ::

        ff.def_pairstyle("lj/cut", cutoff=12.0)
        ff.def_pairstyle("coulomb", calculator="pme", smearing=5.0)
        pots = ff.to_potentials()
        E = pots.calc_energy(system)
    """

    def __init__(self, forcefield: ForceField) -> None:
        self.forcefield = forcefield
        self._neighbors = NeighborListBuilder()
        self.lj: LJCutPotential | None = None
        self.coulomb_open: CoulombOpenPotential | None = None
        self.coulomb_periodic: CoulombPeriodicPotential | None = None
        self._assemble_pair_potentials()

    def _assemble_pair_potentials(self) -> None:
        ff = self.forcefield
        if "lj/cut" in ff.pair_styles():
            opts = ff.style_options("pair:lj/cut")
            self.lj = LJCutPotential(cutoff=float(opts.get("cutoff", 10.0)))
        if "coulomb" in ff.pair_styles() or ff.has_coulomb():
            opts = ff.style_options("pair:coulomb")
            # Periodic calculator is chosen when cell is present at energy time;
            # build both open + periodic potentials from style options.
            pref = float(opts.get("prefactor", kcalmol_A))
            self.coulomb_open = CoulombOpenPotential(
                cutoff=opts.get("cutoff"),
                prefactor=pref,
            )
            calc = str(opts.get("calculator", "ewald"))
            self.coulomb_periodic = CoulombPeriodicPotential(
                smearing=float(opts.get("smearing", 5.0)),
                calculator=calc,  # type: ignore[arg-type]
                lr_wavelength=float(opts.get("lr_wavelength", 1.0)),
                mesh_spacing=float(opts.get("mesh_spacing", 1.0)),
                prefactor=pref,
                real_space_cutoff=opts.get("real_space_cutoff"),
            )

    def get_schema(self) -> ForceFieldSchemaView:
        return PotentialMeta.schema_for(
            self.forcefield.style_names, units=self.forcefield.units
        )

    def calc_energy(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self._pos(system, pos)
        total = torch.zeros((), device=p.device, dtype=p.dtype)
        for e in self.term_energies(system, pos=p).values():
            total = total + e
        return total

    def calc_forces(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self._pos(system, pos).detach().requires_grad_(True)
        energy = self.calc_energy(system, pos=p)
        (grad,) = torch.autograd.grad(energy, p, create_graph=torch.is_grad_enabled())
        return -grad

    def calc_energy_forces(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._pos(system, pos).detach().requires_grad_(True)
        energy = self.calc_energy(system, pos=p)
        (grad,) = torch.autograd.grad(energy, p, create_graph=torch.is_grad_enabled())
        return energy, -grad

    def term_energies(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        p = self._pos(system, pos)
        ff = self.forcefield
        device = p.device
        dtype = p.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        terms: dict[str, torch.Tensor] = {}

        if ff.bond_style() is not None:
            bp = ff.params("bonds")
            bond_index = self._bond_index(system)
            if bp is not None and bond_index is not None and bond_index.size(1) > 0:
                n = bond_index.size(1)
                pot = BondHarmonic(k=bp["k"], r0=bp["r0"])
                terms["bonds"] = pot(
                    pos=p,
                    bond_index=bond_index,
                    bond_types=self._identity_types(n, device),
                )
            else:
                terms["bonds"] = zero

        if ff.angle_style() is not None:
            ap = ff.params("angles")
            angle_index = self._angle_index(system)
            if ap is not None and angle_index is not None and angle_index.size(1) > 0:
                n = angle_index.size(1)
                pot = AngleHarmonic(k=ap["k"], theta0=ap["theta0"])
                terms["angles"] = pot(
                    pos=p,
                    angle_index=angle_index,
                    angle_types=self._identity_types(n, device),
                )
            else:
                terms["angles"] = zero

        dstyle = ff.dihedral_style()
        if dstyle is not None:
            pp = ff.params("propers")
            proper_index = self._proper_index(system)
            if pp is not None and proper_index is not None and proper_index.size(1) > 0:
                n = proper_index.size(1)
                if dstyle == "opls" and "k1" in pp:
                    k_opls = torch.stack(
                        [pp["k1"], pp["k2"], pp["k3"], pp["k4"]], dim=-1
                    )
                    k = 0.5 * k_opls
                    phase = torch.zeros_like(k)
                    phase[:, 1] = math.pi
                    phase[:, 3] = math.pi
                    periodicity = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
                    idivf = pp.get("idivf", torch.ones(n, device=device, dtype=dtype))
                else:
                    k = pp["k"]
                    phase = pp["phase"]
                    periodicity = pp["periodicity"]
                    idivf = pp["idivf"]
                pot = ProperTorsionPeriodic(
                    k=k, periodicity=periodicity, phase=phase, idivf=idivf
                )
                terms["propers"] = pot(
                    pos=p,
                    proper_index=proper_index,
                    proper_types=self._identity_types(n, device),
                )
            else:
                terms["propers"] = zero

        if ff.improper_style() is not None:
            ip = ff.params("impropers")
            improper_index = self._improper_index(system)
            if ip is not None and improper_index is not None and improper_index.size(1) > 0:
                n = improper_index.size(1)
                if "chi0" in ip:
                    pot = ImproperHarmonic(k=ip["k"], chi0=ip["chi0"])
                    terms["impropers"] = pot(
                        pos=p,
                        improper_index=improper_index,
                        improper_types=self._identity_types(n, device),
                    )
                elif "periodicity" in ip:
                    pot = ImproperPeriodic(
                        k=ip["k"],
                        periodicity=ip["periodicity"],
                        phase=ip["phase"],
                        idivf=ip["idivf"],
                    )
                    terms["impropers"] = pot(
                        pos=p,
                        improper_index=improper_index,
                        improper_types=self._identity_types(n, device),
                    )
                else:
                    terms["impropers"] = zero
            else:
                terms["impropers"] = zero

        if self.lj is not None:
            terms["lj"] = self._lj_energy(system, p, zero)
        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            terms["coulomb"] = self._coulomb_energy(system, p, zero)
        return terms

    # ------------------------------------------------------------------
    # nonbonded
    # ------------------------------------------------------------------

    def _charges(self) -> torch.Tensor | None:
        atoms = self.forcefield.params("atoms")
        if atoms is None:
            return None
        return atoms.get("q", atoms.get("charge"))

    def _lj_params(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        atoms = self.forcefield.params("atoms")
        if atoms is None or "epsilon" not in atoms or "sigma" not in atoms:
            return None
        return atoms["epsilon"], atoms["sigma"]

    def _cell(self, system: Any) -> torch.Tensor | None:
        graphs = self._namespace(system, "graphs")
        if graphs is not None:
            cell = self._ns_tensor(graphs, "cell")
            if cell is not None:
                return cell[0] if cell.ndim == 3 else cell
        if isinstance(system, Mapping) and "cell" in system:
            return system["cell"]
        return None

    def _scaled_pairs(
        self,
        system: Any,
        pos: torch.Tensor,
        *,
        cutoff: float | None,
        scale_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        n = pos.shape[0]
        atom_batch = self._atom_batch(system, n, pos.device)
        cell = self._cell(system)
        built = self._neighbors.build(
            pos=pos, atom_batch=atom_batch, cell=cell, cutoff=cutoff
        )
        if built is None:
            return None
        edge_index, dist = built
        scale_mat = self._pair_scale_matrix(
            system,
            n,
            pos.device,
            pos.dtype,
            self.forcefield.nonbonded_scaling,
            scale_kind,
        )
        scales = scale_mat[edge_index[:, 0], edge_index[:, 1]]
        return edge_index, dist, scales

    def _lj_energy(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        assert self.lj is not None
        params = self._lj_params()
        if params is None:
            return zero
        epsilon, sigma = params
        pairs = self._scaled_pairs(
            system, pos, cutoff=self.lj.cutoff, scale_kind="lj"
        )
        if pairs is None:
            return zero
        edge_index, dist, scales = pairs
        return self.lj.energy(
            edge_index=edge_index,
            distances=dist,
            epsilon=epsilon,
            sigma=sigma,
            scales=scales,
        )

    def _coulomb_energy(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        q = self._charges()
        if q is None:
            return zero
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")

        cell = self._cell(system)
        if cell is None:
            assert self.coulomb_open is not None
            cut = self.coulomb_open.cutoff
            pairs = self._scaled_pairs(system, pos, cutoff=cut, scale_kind="q")
            if pairs is None:
                return zero
            edge_index, dist, scales = pairs
            return self.coulomb_open.energy(
                edge_index=edge_index,
                distances=dist,
                charges=q,
                scales=scales,
            )

        assert self.coulomb_periodic is not None
        # Neighbor list cutoff from the periodic Potential's real-space cutoff
        pairs = self._scaled_pairs(
            system,
            pos,
            cutoff=self.coulomb_periodic.real_space_cutoff,
            scale_kind="q",
        )
        if pairs is None:
            edge_index = torch.zeros(0, 2, dtype=torch.long, device=pos.device)
            dist = torch.zeros(0, dtype=pos.dtype, device=pos.device)
            scales = torch.zeros(0, dtype=pos.dtype, device=pos.device)
        else:
            edge_index, dist, scales = pairs
        return self.coulomb_periodic.energy(
            positions=pos,
            charges=q,
            cell=cell,
            edge_index=edge_index,
            distances=dist,
            scales=scales,
        )

    def _atom_batch(self, system: Any, n: int, device: torch.device) -> torch.Tensor:
        atoms = self._namespace(system, "atoms")
        if atoms is not None:
            b = self._ns_tensor(atoms, "batch")
            if b is not None:
                return b.long()
        return torch.zeros(n, dtype=torch.long, device=device)

    def _pair_scale_matrix(
        self,
        system: Any,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        scaling: NonbondedScaling,
        scale_kind: str,
    ) -> torch.Tensor:
        scale = torch.ones(n, n, device=device, dtype=dtype)
        if scale_kind == "q":
            s12, s13, s14 = (
                float(scaling.scale_q_12),
                float(scaling.scale_q_13),
                float(scaling.scale_q_14),
            )
        else:
            s12, s13, s14 = (
                float(scaling.scale_lj_12),
                float(scaling.scale_lj_13),
                float(scaling.scale_lj_14),
            )
        bond_index = self._bond_index(system)
        if bond_index is not None and bond_index.numel() > 0:
            i, j = bond_index[0].long(), bond_index[1].long()
            scale[i, j] = s12
            scale[j, i] = s12
        angle_index = self._angle_index(system)
        if angle_index is not None and angle_index.numel() > 0:
            i, k = angle_index[0].long(), angle_index[2].long()
            scale[i, k] = s13
            scale[k, i] = s13
        proper_index = self._proper_index(system)
        if proper_index is not None and proper_index.numel() > 0:
            i, ell = proper_index[0].long(), proper_index[3].long()
            free = scale[i, ell] == 1.0
            i14, l14 = i[free], ell[free]
            scale[i14, l14] = s14
            scale[l14, i14] = s14
        return scale

    # --- topology readers ---

    def _namespace(self, system: Any, name: str) -> Any | None:
        if system is None:
            return None
        if isinstance(system, (Mapping, TensorDict)):
            try:
                if name in system:
                    return system[name]
            except Exception:  # noqa: BLE001
                return None
        return None

    def _ns_tensor(self, ns: Any, *keys: str) -> torch.Tensor | None:
        if ns is None:
            return None
        for key in keys:
            try:
                if isinstance(ns, (Mapping, TensorDict)) and key in ns:
                    return ns[key]
            except Exception:  # noqa: BLE001
                continue
        return None

    def _pos(self, system: Any, pos: torch.Tensor | None) -> torch.Tensor:
        if pos is not None:
            return pos
        atoms = self._namespace(system, "atoms")
        if atoms is not None:
            p = self._ns_tensor(atoms, "pos")
            if p is not None:
                return p
        if isinstance(system, Mapping) and "pos" in system:
            return system["pos"]
        raise ValueError("calc_energy requires pos or system atoms.pos")

    def _bond_index(self, system: Any) -> torch.Tensor | None:
        bonds = self._namespace(system, "bonds")
        if bonds is not None:
            packed = self._ns_tensor(bonds, "bond_index")
            if packed is not None:
                return packed
            atomi = self._ns_tensor(bonds, "atomi")
            atomj = self._ns_tensor(bonds, "atomj")
            if atomi is not None and atomj is not None:
                return torch.stack([atomi.long(), atomj.long()], dim=0)
        if isinstance(system, Mapping) and "bond_index" in system:
            return system["bond_index"]
        return None

    def _angle_index(self, system: Any) -> torch.Tensor | None:
        angles = self._namespace(system, "angles")
        if angles is None:
            return None
        atomi = self._ns_tensor(angles, "atomi")
        atomj = self._ns_tensor(angles, "atomj")
        atomk = self._ns_tensor(angles, "atomk")
        if atomi is None or atomj is None or atomk is None:
            return None
        return torch.stack([atomi.long(), atomj.long(), atomk.long()], dim=0)

    def _proper_index(self, system: Any) -> torch.Tensor | None:
        propers = self._namespace(system, "propers")
        if propers is None:
            return None
        cols = [self._ns_tensor(propers, k) for k in ("atomi", "atomj", "atomk", "atoml")]
        if any(c is None for c in cols):
            return None
        return torch.stack([c.long() for c in cols], dim=0)  # type: ignore[union-attr]

    def _improper_index(self, system: Any) -> torch.Tensor | None:
        impropers = self._namespace(system, "impropers")
        if impropers is None:
            return None
        cols = [self._ns_tensor(impropers, k) for k in ("atomi", "atomj", "atomk", "atoml")]
        if any(c is None for c in cols):
            return None
        return torch.stack([c.long() for c in cols], dim=0)  # type: ignore[union-attr]

    @staticmethod
    def _identity_types(n: int, device: torch.device) -> torch.Tensor:
        return torch.arange(n, device=device, dtype=torch.long)

    def __repr__(self) -> str:
        return f"Potentials(forcefield={self.forcefield!r})"
