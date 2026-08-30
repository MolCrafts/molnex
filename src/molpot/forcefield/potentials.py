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

from molpot.derivation.force import autograd_forces_and_virial, autograd_forces_from_energy
from molpot.derivation.protocol import ENERGY_KEY, FORCES_KEY, write_virial
from molpot.forcefield.neighbor_list import NeighborListBuilder
from molpot.forcefield.pair_terms import (
    CoulombOpenPotential,
    CoulombPeriodicPotential,
    LJCutPotential,
)
from molpot.forcefield.schema import ForceFieldSchemaView, PotentialMeta
from molpot.ir import ProperTorsionBag
from molpot.ir.scaling import NonbondedScaling
from molpot.potentials import (
    AngleHarmonic,
    BondHarmonic,
    ImproperHarmonic,
    ImproperPeriodic,
    ProperTorsionPeriodic,
)
from molpot.potentials.base import BasePotential
from molpot.potentials.elec.prefactors import kcalmol_A

if TYPE_CHECKING:
    from molpot.forcefield.container import ForceField

__all__ = ["Potentials"]


class Potentials(BasePotential):
    """Classical Class-I energy as a molpot :class:`BasePotential` (``nn.Module``).

    Embed in a larger torch model, or reach MD through the owning
    :class:`~molpot.forcefield.container.ForceField` and
    :meth:`molix.md.MD.set_forcefield`. ``forward`` on a TensorDict
    writes ``graphs.energy`` and ``atoms.forces``.

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

    Args:
        forcefield: Filled :class:`~molpot.forcefield.ForceField` whose declared
            styles and parameter tables are assembled into torch terms here. Its
            unit convention (``units="real"`` by default: kcal/mol, Å) is the unit
            convention of every energy and force this container returns.
    """

    name = "class_i"
    type = "classical"

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__()
        self.forcefield = forcefield
        self._neighbors = NeighborListBuilder()
        self.lj: LJCutPotential | None = None
        self.coulomb_open: CoulombOpenPotential | None = None
        self.coulomb_periodic: CoulombPeriodicPotential | None = None
        self.mlip = getattr(forcefield, "_mlip", None)
        self._assemble_pair_potentials()
        self._assemble_bonded()

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
                calculator=calc,
                lr_wavelength=float(opts.get("lr_wavelength", 1.0)),
                mesh_spacing=float(opts.get("mesh_spacing", 1.0)),
                prefactor=pref,
                real_space_cutoff=opts.get("real_space_cutoff"),
            )

    def _assemble_bonded(self) -> None:
        ff = self.forcefield
        bp = ff.params("bonds")
        if bp is not None and "k" in bp and "r0" in bp:
            self.bond = BondHarmonic(k=bp["k"], r0=bp["r0"])
        else:
            self.bond = None
        ap = ff.params("angles")
        if ap is not None and "k" in ap and "theta0" in ap:
            self.angle = AngleHarmonic(k=ap["k"], theta0=ap["theta0"])
        else:
            self.angle = None
        pp = ff.params("propers")
        if pp is not None:
            self.proper = self._make_proper(pp)
        else:
            self.proper = None
        ip = ff.params("impropers")
        if ip is not None and "chi0" in ip:
            self.improper = ImproperHarmonic(k=ip["k"], chi0=ip["chi0"])
        elif ip is not None and "periodicity" in ip:
            self.improper = ImproperPeriodic(
                k=ip["k"],
                periodicity=ip["periodicity"],
                phase=ip["phase"],
                idivf=ip["idivf"],
            )
        else:
            self.improper = None

    def _make_proper(self, pp: dict[str, torch.Tensor]) -> ProperTorsionPeriodic:
        if "k1" in pp:
            k_opls = torch.stack([pp["k1"], pp["k2"], pp["k3"], pp["k4"]], dim=-1)
            k = 0.5 * k_opls
            phase = torch.zeros_like(k)
            phase[:, 1] = math.pi
            phase[:, 3] = math.pi
            n = k.shape[0]
            periodicity = torch.tensor([1, 2, 3, 4], device=k.device, dtype=torch.long)
            idivf = pp.get("idivf", torch.ones(n, device=k.device, dtype=k.dtype))
            return ProperTorsionPeriodic(k=k, periodicity=periodicity, phase=phase, idivf=idivf)
        pp = ProperTorsionBag._cosine_fields(pp)
        return ProperTorsionPeriodic(
            k=pp["k"],
            periodicity=pp["periodicity"],
            phase=pp["phase"],
            idivf=pp["idivf"],
        )

    def forward(self, data: Any = None, **kwargs: Any) -> Any:
        """Energy, or TensorDict with ``graphs.energy`` / ``atoms.forces``.

        A nested TensorDict is treated as an MD / composer batch: energy and
        forces are written in place and the batch is returned so the module
        composes into the MD driver's TensorDict force adapter.
        """
        if self.mlip is not None and isinstance(data, TensorDict) and kwargs.get("pos") is None:
            return self.mlip(data)
        energy, forces = self.calc_energy_forces(data, pos=kwargs.get("pos"))
        if isinstance(data, TensorDict):
            if "graphs" not in data.keys():
                data["graphs"] = TensorDict({}, batch_size=[])
            data[ENERGY_KEY] = energy
            data[FORCES_KEY] = forces
            return data
        return energy

    def get_schema(self) -> ForceFieldSchemaView:
        """Metadata of the declared styles: parameter names, units, energy formulas."""
        return PotentialMeta.schema_for(self.forcefield.style_names(), units=self.forcefield.units)

    def calc_energy(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total energy: the sum of every term the ForceField declares.

        Overrides :meth:`~molpot.potentials.base.BasePotential.calc_energy`, which
        returns a Python ``float``: this returns a **tensor** still attached to the
        autograd graph, because :meth:`calc_forces` and the Hessian path
        differentiate it.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``).
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Energy ``(B,)`` over the ``B`` molecules of the batch — shape ``()`` for
            a single molecule — in the ForceField's unit convention
            (``units="real"`` default: kcal/mol).
        """
        if self.mlip is not None and isinstance(system, TensorDict):
            if pos is not None:
                system["atoms", "pos"] = pos
            write = getattr(self.mlip, "_write_energy", None)
            if callable(write):
                write(system)
                return system[ENERGY_KEY]
            out = self.mlip(system)
            return out[ENERGY_KEY]
        p = self._pos(system, pos)
        num_graphs = self._num_graphs(system)
        total = torch.zeros((num_graphs,) if num_graphs > 1 else (), device=p.device, dtype=p.dtype)
        for e in self.term_energies(system, pos=p).values():
            total = total + e
        return total

    def calc_forces(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Atomic forces ``F = -∂E/∂pos``.

        Delegates to the canonical autograd backend
        :func:`~molpot.derivation.force.autograd_forces_from_energy` — no
        second force path lives here. Per-graph energies ``(B,)`` are summed to
        one scalar before differentiating, so a multi-molecule batch works.

        Positions are re-attached as a fresh ``requires_grad`` leaf taken from
        ``pos.detach()``: any autograd history the caller's ``pos`` carried is not
        differentiated through, while the force stays connected to the model
        parameters ``θ`` (the mixed second derivative ``∂²E/∂pos∂θ`` is kept)
        whenever grad is enabled, so a force loss can reach the parameter heads.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` and the topology.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Forces ``(N, 3)`` in the ForceField's unit convention
            (``units="real"`` default: kcal mol⁻¹ Å⁻¹).
        """
        p = self._pos(system, pos).detach().requires_grad_(True)
        energy = self.calc_energy(system, pos=p)
        return autograd_forces_from_energy(energy, p)

    def calc_virial(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Virial ``W = ∂E/∂ε`` at ``ε = 0``, shape ``(3, 3)``.

        Requires ``graphs.cell``. Imaged ``edges.edge_diff`` is strained with
        the cell when present so the pairwise (and ML) virial includes the
        minimum-image shift.

        Args:
            system: Batch carrying ``atoms.pos`` and ``graphs.cell``.
            pos: Optional positions overriding ``atoms.pos``.

        Returns:
            ``W`` in the ForceField energy unit.

        Raises:
            ValueError: If ``graphs.cell`` is missing.
        """
        if not isinstance(system, TensorDict):
            raise TypeError("calc_virial requires a TensorDict system")
        nested = system.keys(include_nested=True)
        if ("graphs", "cell") not in nested:
            raise ValueError("calc_virial requires graphs.cell")
        cell = system["graphs", "cell"]
        if cell.dim() == 3:
            cell = cell[0]
        p = self._pos(system, pos)
        cartesians: tuple[torch.Tensor, ...] = ()
        if ("edges", "edge_diff") in nested:
            cartesians = (system["edges", "edge_diff"],)

        def energy_fn(
            pos_s: torch.Tensor, cell_s: torch.Tensor, *strained: torch.Tensor
        ) -> torch.Tensor:
            system["atoms", "pos"] = pos_s
            system["graphs", "cell"] = cell_s
            if strained:
                system["edges", "edge_diff"] = strained[0]
                system["edges", "edge_dist"] = strained[0].norm(dim=-1)
            energy = self.calc_energy(system, pos=pos_s)
            return energy if energy.ndim == 0 else energy.sum()

        _, _, virial = autograd_forces_and_virial(energy_fn, p, cell, *cartesians)
        write_virial(system, virial)
        return virial

    def calc_energy_forces(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and forces from one energy evaluation.

        Bonded terms (bond / angle / proper / improper) and the LJ term
        evaluate their closed form on detached positions and return both
        energy and analytic forces, so no position leaf — and no
        ``create_graph`` double backward — is paid for them. Coulomb (open
        ``1/r`` or periodic Ewald/PME/P3M) shares the molpot kernel stack and
        has no cheap closed form, so it falls back to the autograd backend on
        a fresh position leaf.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` and the topology.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Energy ``(B,)`` — ``()`` for a single molecule — and forces ``(N, 3)``,
            in the ForceField's unit convention (``units="real"`` default:
            kcal/mol and kcal mol⁻¹ Å⁻¹).
        """
        p = self._pos(system, pos).detach()
        device = p.device
        dtype = p.dtype
        num_graphs = self._num_graphs(system)
        batched = num_graphs > 1
        zero = torch.zeros((num_graphs,) if batched else (), device=device, dtype=dtype)
        n_atoms = p.shape[0]
        atom_batch = self._atom_batch(system, n_atoms, device)

        energy = zero
        forces = p.new_zeros((n_atoms, 3))
        ff = self.forcefield

        if ff.bond_style() is not None:
            bond_index = self._bond_index(system)
            if self.bond is not None and bond_index is not None and bond_index.size(1) > 0:
                e, f = self.bond.energy_forces(
                    pos=p,
                    bond_index=bond_index,
                    bond_types=self._identity_types(bond_index.size(1), device),
                    term_batch=atom_batch[bond_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                e, f = zero, p.new_zeros((n_atoms, 3))
            energy = energy + e
            forces = forces + f

        if ff.angle_style() is not None:
            angle_index = self._angle_index(system)
            if self.angle is not None and angle_index is not None and angle_index.size(1) > 0:
                e, f = self.angle.energy_forces(
                    pos=p,
                    angle_index=angle_index,
                    angle_types=self._identity_types(angle_index.size(1), device),
                    term_batch=atom_batch[angle_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                e, f = zero, p.new_zeros((n_atoms, 3))
            energy = energy + e
            forces = forces + f

        if ff.dihedral_style() is not None:
            proper_index = self._proper_index(system)
            if self.proper is not None and proper_index is not None and proper_index.size(1) > 0:
                e, f = self.proper.energy_forces(
                    pos=p,
                    proper_index=proper_index,
                    proper_types=self._identity_types(proper_index.size(1), device),
                    term_batch=atom_batch[proper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                e, f = zero, p.new_zeros((n_atoms, 3))
            energy = energy + e
            forces = forces + f

        if ff.improper_style() is not None:
            improper_index = self._improper_index(system)
            if (
                self.improper is not None
                and improper_index is not None
                and improper_index.size(1) > 0
            ):
                e, f = self.improper.energy_forces(
                    pos=p,
                    improper_index=improper_index,
                    improper_types=self._identity_types(improper_index.size(1), device),
                    term_batch=atom_batch[improper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                e, f = zero, p.new_zeros((n_atoms, 3))
            energy = energy + e
            forces = forces + f

        if self.lj is not None:
            e, f = self._lj_energy_forces(system, p, zero, n_atoms)
            energy = energy + e
            forces = forces + f

        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            e, f = self._coulomb_energy_forces(system, p, zero, n_atoms)
            energy = energy + e
            forces = forces + f

        return energy, forces

    def term_energies(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Energy of each declared term, keyed by term name.

        One entry per style the ForceField declares — ``bonds`` / ``angles`` /
        ``propers`` / ``impropers`` / ``lj`` / ``coulomb`` — so a bonded-only
        ForceField yields no ``lj`` and no ``coulomb`` key even when ``epsilon`` /
        ``sigma`` or charges already sit in its parameter tables. A declared style
        whose topology or parameters are missing contributes an explicit zero rather
        than silently disappearing.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Term name → energy ``(B,)`` (``()`` for a single molecule) in the
            ForceField's unit convention (``units="real"`` default: kcal/mol).
        """
        p = self._pos(system, pos)
        ff = self.forcefield
        device = p.device
        dtype = p.dtype
        num_graphs = self._num_graphs(system)
        batched = num_graphs > 1
        zero = torch.zeros((num_graphs,) if batched else (), device=device, dtype=dtype)
        atom_batch = self._atom_batch(system, p.shape[0], device)
        terms: dict[str, torch.Tensor] = {}

        if ff.bond_style() is not None:
            bond_index = self._bond_index(system)
            if self.bond is not None and bond_index is not None and bond_index.size(1) > 0:
                n = bond_index.size(1)
                terms["bonds"] = self.bond(
                    pos=p,
                    bond_index=bond_index,
                    bond_types=self._identity_types(n, device),
                    term_batch=atom_batch[bond_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                terms["bonds"] = zero

        if ff.angle_style() is not None:
            angle_index = self._angle_index(system)
            if self.angle is not None and angle_index is not None and angle_index.size(1) > 0:
                n = angle_index.size(1)
                terms["angles"] = self.angle(
                    pos=p,
                    angle_index=angle_index,
                    angle_types=self._identity_types(n, device),
                    term_batch=atom_batch[angle_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                terms["angles"] = zero

        if ff.dihedral_style() is not None:
            proper_index = self._proper_index(system)
            if self.proper is not None and proper_index is not None and proper_index.size(1) > 0:
                n = proper_index.size(1)
                terms["propers"] = self.proper(
                    pos=p,
                    proper_index=proper_index,
                    proper_types=self._identity_types(n, device),
                    term_batch=atom_batch[proper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
            else:
                terms["propers"] = zero

        if ff.improper_style() is not None:
            improper_index = self._improper_index(system)
            if (
                self.improper is not None
                and improper_index is not None
                and improper_index.size(1) > 0
            ):
                n = improper_index.size(1)
                terms["impropers"] = self.improper(
                    pos=p,
                    improper_index=improper_index,
                    improper_types=self._identity_types(n, device),
                    term_batch=atom_batch[improper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                )
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
        return atoms.get("q")

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
        """Half neighbor list ``(edge_index, dist, scales)`` for one pair term."""
        atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
        cell = self._cell(system)
        edges = self._namespace(system, "edges")
        built = self._neighbors.build(
            pos=pos, atom_batch=atom_batch, cell=cell, cutoff=cutoff, edges=edges
        )
        if built is None:
            return None
        edge_index, dist = built
        scales = self._pair_scales(
            system,
            pos.shape[0],
            edge_index,
            pos.device,
            pos.dtype,
            self.forcefield.nonbonded_scaling,
            scale_kind,
        )
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
        # Keep LJ on its own cutoff list. Reusing an unlimited Coulomb half-list
        # forced ``_pair_scales`` (and distance storage) onto all intramolecular
        # pairs even though LJ only needs ``r < cutoff``.
        pairs = self._scaled_pairs(system, pos, cutoff=self.lj.cutoff, scale_kind="lj")
        if pairs is None:
            return zero
        edge_index, dist, scales = pairs
        num_graphs = self._num_graphs(system)
        atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
        return self.lj.energy(
            edge_index=edge_index,
            distances=dist,
            epsilon=epsilon,
            sigma=sigma,
            scales=scales,
            edge_batch=atom_batch[edge_index[:, 0]] if num_graphs > 1 else None,
            num_graphs=num_graphs,
        )

    def _lj_energy_forces(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lj is not None
        params = self._lj_params()
        if params is None:
            return zero, pos.new_zeros((n_atoms, 3))
        epsilon, sigma = params
        pairs = self._scaled_pairs(system, pos, cutoff=self.lj.cutoff, scale_kind="lj")
        if pairs is None:
            return zero, pos.new_zeros((n_atoms, 3))
        edge_index, dist, scales = pairs
        num_graphs = self._num_graphs(system)
        atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
        return self.lj.energy_forces(
            positions=pos,
            edge_index=edge_index,
            distances=dist,
            epsilon=epsilon,
            sigma=sigma,
            scales=scales,
            edge_batch=atom_batch[edge_index[:, 0]] if num_graphs > 1 else None,
            num_graphs=num_graphs,
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
            num_graphs = self._num_graphs(system)
            atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
            return self.coulomb_open.energy(
                edge_index=edge_index,
                distances=dist,
                charges=q,
                scales=scales,
                edge_batch=atom_batch[edge_index[:, 0]] if num_graphs > 1 else None,
                num_graphs=num_graphs,
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

    def _coulomb_energy_forces(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._charges()
        if q is None:
            return zero, pos.new_zeros((n_atoms, 3))
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")

        # Open Coulomb and Ewald/PME/P3M all share the molpot Coulomb kernel
        # stack; neither exposes a cheap closed form for forces, so the single
        # force path is one autograd backward on a fresh position leaf.
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            p_nb = pos.detach().requires_grad_(True)
            nb_energy = self._coulomb_energy(system, p_nb, zero)
            create_graph = grad_enabled and q.requires_grad
            nb_forces = (
                autograd_forces_from_energy(nb_energy, p_nb, create_graph=create_graph)
                if nb_energy.requires_grad
                else pos.new_zeros((n_atoms, 3))
            )
        # ``create_graph=False`` freed ``nb_energy``'s graph inside the grad call;
        # detach it so a later ``loss.backward()`` never touches the freed graph.
        return nb_energy if create_graph else nb_energy.detach(), nb_forces

    def _atom_batch(self, system: Any, n: int, device: torch.device) -> torch.Tensor:
        atoms = self._namespace(system, "atoms")
        if atoms is not None:
            b = self._ns_tensor(atoms, "batch")
            if b is not None:
                return b.long()
        return torch.zeros(n, dtype=torch.long, device=device)

    def _num_graphs(self, system: Any) -> int:
        graphs = self._namespace(system, "graphs")
        if isinstance(graphs, TensorDict) and graphs.batch_size:
            return int(graphs.batch_size[0])
        if isinstance(graphs, (Mapping, TensorDict)):
            num_atoms = self._ns_tensor(graphs, "num_atoms")
            if num_atoms is not None and num_atoms.ndim >= 1:
                return int(num_atoms.shape[0])
        atoms = self._namespace(system, "atoms")
        atom_batch = self._ns_tensor(atoms, "batch")
        if atom_batch is not None and atom_batch.numel():
            return int(atom_batch[-1].detach().to("cpu")) + 1
        return 1

    @staticmethod
    def _undirected_pair_keys(index: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """Map endpoint index ``(2|3|4, M)`` → undirected key ``min*n+max``."""
        lo = torch.minimum(index[0].long(), index[-1].long())
        hi = torch.maximum(index[0].long(), index[-1].long())
        return lo * n_atoms + hi

    @classmethod
    def _mask_pairs_in(
        cls,
        pair_keys: torch.Tensor,
        term_index: torch.Tensor,
        n_atoms: int,
    ) -> torch.Tensor:
        """Boolean mask of ``pair_keys`` that appear as undirected ``term_index`` ends."""
        keys = torch.unique(cls._undirected_pair_keys(term_index, n_atoms))
        if keys.numel() == 0:
            return torch.zeros(pair_keys.shape[0], dtype=torch.bool, device=pair_keys.device)
        keys, _ = torch.sort(keys)
        idx = torch.searchsorted(keys, pair_keys).clamp(max=keys.numel() - 1)
        return keys[idx] == pair_keys

    def _pair_scales(
        self,
        system: Any,
        n: int,
        edge_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        scaling: NonbondedScaling,
        scale_kind: str,
    ) -> torch.Tensor:
        scale = torch.ones(edge_index.shape[0], device=device, dtype=dtype)
        if edge_index.numel() == 0:
            return scale
        pair_lo = torch.minimum(edge_index[:, 0], edge_index[:, 1])
        pair_hi = torch.maximum(edge_index[:, 0], edge_index[:, 1])
        pair_keys = pair_lo * n + pair_hi

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
            scale[self._mask_pairs_in(pair_keys, bond_index, n)] = s12
        angle_index = self._angle_index(system)
        if angle_index is not None and angle_index.numel() > 0:
            scale[self._mask_pairs_in(pair_keys, angle_index, n)] = s13
        proper_index = self._proper_index(system)
        if proper_index is not None and proper_index.numel() > 0:
            proper_mask = self._mask_pairs_in(pair_keys, proper_index, n)
            scale[proper_mask & (scale == 1.0)] = s14
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
        return torch.stack([c.long() for c in cols], dim=0)

    def _improper_index(self, system: Any) -> torch.Tensor | None:
        impropers = self._namespace(system, "impropers")
        if impropers is None:
            return None
        cols = [self._ns_tensor(impropers, k) for k in ("atomi", "atomj", "atomk", "atoml")]
        if any(c is None for c in cols):
            return None
        return torch.stack([c.long() for c in cols], dim=0)

    @staticmethod
    def _identity_types(n: int, device: torch.device) -> torch.Tensor:
        return torch.arange(n, device=device, dtype=torch.long)

    def __repr__(self) -> str:
        return f"Potentials(forcefield={self.forcefield!r})"
