"""Potentials — assembled evaluable handle for one ForceField.

Pair cutoffs and electrostatic calculator options live on the **Potential
terms** (see :mod:`molpot.forcefield.pair_terms`), not on this container.

The energy is reported three ways over **one** evaluation:
:meth:`Potentials.calc_energy` (the total), :meth:`Potentials.term_energies`
(per named term — ``bonds``, ``lj_repulsion``, ``coulomb``, …) and
:meth:`Potentials.contribution_energies`, which groups those terms into the
S-decomposition ``E = E_local + E_disp + E_perm + E_pol + E_CT`` using the
label each style declares in its
:class:`~molpot.contract.InteractionContract`. Grouping, never a second energy
path, so the parts always sum back to the total.

The labels ``S1`` / ``S3`` / ``S5`` (the spec requirements this layer
implements: module contracts, no double counting, and the constraints — among
them the explicitly enforced total charge of :meth:`Potentials.bind_topology`)
and ``F2`` / ``F6`` (the two silent-zero findings the raises below close) are
defined in ``.claude/notes/composable-potentials.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from molix.schema import CellBound, CoreCompilable
from molpot.contract import Contribution, ExclusionClass
from molpot.derivation.force import autograd_forces_and_virial, autograd_forces_from_energy
from molpot.derivation.protocol import (
    ENERGY_KEY,
    FORCES_KEY,
    call_energy,
    write_energy,
    write_forces,
    write_virial,
)
from molpot.forcefield.exclusions import ExclusionTable
from molpot.forcefield.neighbor_list import NeighborListBuilder
from molpot.forcefield.pair_terms import (
    CoulombOpenPotential,
    CoulombPeriodicPotential,
    LJCutPotential,
)
from molpot.forcefield.schema import ForceFieldSchemaView, PotentialMeta
from molpot.ir import ProperTorsionBag
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

#: Largest per-graph deviation of ``Σ_g q`` from the declared
#: ``graphs.total_charge`` that :meth:`Potentials.bind_topology` accepts (e).
#: Sized for fp32 accumulation over a few thousand atoms, far below the
#: 0.01 e a drifting learned charge head produces.
_TOTAL_CHARGE_TOL = 1e-6


class Potentials(BasePotential):
    """Classical Class-I energy as a molpot :class:`BasePotential` (``nn.Module``).

    Embed in a larger torch model, or reach MD through the owning
    :class:`~molpot.forcefield.container.ForceField` and
    :meth:`molix.md.MD.set_forcefield`. ``forward`` on a TensorDict
    writes ``graphs.energy`` and ``atoms.forces``.

    Bonded terms use existing harmonic / torsion kernels. Nonbonded terms are
    concrete Potential objects that own their cutoffs:

    * ``pair:lj/cut`` → :class:`LJCutPotential`. Style options, both in Å and
      both owned by the Potential: ``cutoff`` — the radius ``r_c`` beyond
      which a pair contributes nothing (default ``10.0``) — and the optional
      ``switch_start`` ``r_s`` with ``0 < r_s < r_c``, which tapers the energy
      smoothly to zero over ``[r_s, r_c]`` with the quintic switch instead of
      truncating it at ``r_c``. Omitting ``switch_start`` keeps the hard
      truncation.
    * ``pair:coulomb`` open → :class:`CoulombOpenPotential` (optional
      ``cutoff`` in Å; ``prefactor`` in kcal Å mol⁻¹ e⁻²)
    * ``pair:coulomb`` + cell → :class:`CoulombPeriodicPotential`
      (``calculator=ewald|pme|p3m``, ``smearing`` in Å, on the Potential)

    ::

        ff.def_pairstyle("lj/cut", cutoff=12.0, switch_start=10.0)
        ff.def_pairstyle("coulomb", calculator="pme", smearing=5.0)
        pots = ff.to_potentials()
        E = pots.calc_energy(system)

    Args:
        forcefield: Filled :class:`~molpot.forcefield.ForceField` whose declared
            styles and parameter tables are assembled into torch terms here. Its
            unit convention (``units="real"`` by default: kcal/mol, Å) is the unit
            convention of every energy and force this container returns.

    Raises:
        ValueError: The declared styles do not compose, checked before a single
            term is assembled: a style is not registered in
            :class:`~molpot.forcefield.schema.PotentialMeta`; the combination is
            refused by
            :meth:`~molpot.forcefield.schema.PotentialMeta.check_composition`
            (more than one polarization style, or a polarization style with no
            permanent-electrostatics partner); or a style consumes the
            induced-response exclusion table that no ForceField carries yet.
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
        self._exclusions: ExclusionTable | None = None
        self._bound_topology: tuple[int, ...] | None = None
        self._check_composition()
        self._assemble_pair_potentials()
        self._assemble_bonded()

    def _check_composition(self) -> None:
        """Refuse a style combination before a single term is assembled.

        Runs ahead of :meth:`_assemble_pair_potentials` /
        :meth:`_assemble_bonded` so a refused ForceField never leaves a
        half-built container behind.

        Raises:
            ValueError: A declared style is not registered (a typo used to
                assemble into a silent no-op), the combination is refused by
                :meth:`~molpot.forcefield.schema.PotentialMeta.check_composition`,
                or a style consumes the induced-exclusion table that no
                ForceField carries yet.
        """
        keys = list(self.forcefield.style_names())
        for key in keys:
            if key.startswith("atom:") or key in PotentialMeta._REGISTRY:
                continue
            raise ValueError(
                f"ForceField declares an unregistered style {key!r}; registered styles: "
                f"{sorted(PotentialMeta._REGISTRY)}"
            )
        PotentialMeta.check_composition(keys)
        for key in keys:
            if key.startswith("atom:"):
                continue
            if PotentialMeta.contract_for(key).exclusion is ExclusionClass.INDUCED and not hasattr(
                self.forcefield, "induced_scaling"
            ):
                raise ValueError(
                    f"style {key!r} consumes the induced-response exclusion table, but this "
                    "ForceField has no 'induced_scaling' (arrives with composable-a1)"
                )

    def _assemble_pair_potentials(self) -> None:
        ff = self.forcefield
        if "lj/cut" in ff.pair_styles():
            opts = ff.style_options("pair:lj/cut")
            # ``switch_start`` is the second half of the ``lj/cut`` cutoff option
            # and travels with it: dropped here, the quintic taper on
            # :class:`LJCutPotential` would be unreachable from any ForceField and
            # every assembled Class-I model would keep the hard truncation.
            # ``None`` (no window) stays ``None``; the ``0 < r_s < r_c`` check is
            # the term's, so a bad window raises there rather than twice.
            switch_start = opts.get("switch_start")
            self.lj = LJCutPotential(
                cutoff=float(opts.get("cutoff", 10.0)),
                switch_start=None if switch_start is None else float(switch_start),
            )
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

    def _require_bonded_params(
        self,
        category: str,
        table: str,
        term: BasePotential | None,
    ) -> None:
        """Refuse a declared bonded style whose parameter table never filled its term.

        The bonded counterpart of :meth:`_charges` / :meth:`_lj_params`, and the
        same F2 rule: :meth:`_assemble_bonded` leaves ``term`` at ``None`` when
        the style's table is absent or misses a parameter it reads, and an
        unparameterised declaration then contributed a zero indistinguishable
        from a molecule whose bonds genuinely cost nothing. ``term is not None``
        is the whole fast path — the schema lookup happens only on the way to
        the exception.

        Absent *topology* is a different fact and stays a zero: it is checked by
        the caller *after* this guard, so a filled table with no bonds in the
        batch still contributes an explicit zero.

        Args:
            category: Style category of the branch — ``"bond"``, ``"angle"``,
                ``"dihedral"`` or ``"improper"``. Called only from a branch that
                already found a declared style of this category.
            table: Parameter table the style reads (``"bonds"``, ``"angles"``,
                ``"propers"``, ``"impropers"``).
            term: Term assembled by :meth:`_assemble_bonded`, ``None`` when the
                parameter tables could not fill it.

        Raises:
            ValueError: ``term`` is ``None``; the message names the style key,
                the parameters missing from the table, and the table.
        """
        if term is not None:
            return
        style_key, schema = next(
            (key, style)
            for key, style in self.get_schema().styles.items()
            if style.category == category
        )
        declared = tuple(spec.name for spec in schema.params)
        filled = self.forcefield.params(table) or {}
        missing = tuple(name for name in declared if name not in filled) or declared
        raise ValueError(
            f"style {style_key!r} is declared but its parameter table is absent: "
            f"no {' / '.join(missing)} under {table}"
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

    def bind_topology(self, system: Any, *, pos: torch.Tensor | None = None) -> None:
        """Build this topology's exclusion table, and check the declared net charge.

        Called **once per topology** — the connectivity analogue of
        :meth:`molix.schema.CellBound.bind_cell`, which is called once per
        cell. :meth:`_scaled_pairs` calls it lazily on the first nonbonded
        evaluation, so MD and a single-batch score need no explicit call; a
        caller that reuses one ``Potentials`` over a **different** topology
        must call it again, because positions may change freely between energy
        calls but the bond / angle / torsion columns may not. Repeated calls
        are harmless: the table is rebuilt from scratch each time.

        The bind records a **shape-level** fingerprint of what it read — the
        atom count and the column counts of ``bonds`` / ``angles`` /
        ``propers``, Python ints off ``.shape``, no tensor content and no host
        sync — and every later energy call compares it (:meth:`_scaled_pairs`)
        and raises rather than answering out of a stale table. Its limit is the
        one that keeps positions free: a different topology of the *same*
        shape (a re-indexed molecule, another conformer's neighbour of equal
        column counts) is indistinguishable here and remains the caller's
        contract. Never silently rebinding is the point — a rebuilt table
        would be one more quietly wrong energy.

        Compiling the energy path? Bind before entering the compiled region —
        the lazy call inside would trace the build (and the host-side check
        below) as a graph break on the first step.

        The total-charge check lives here rather than in the force loop
        precisely because here it is affordable: eager, per graph and
        host-side, once per topology instead of once per step.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers``).
                ``graphs.total_charge`` ``(B,)`` in e is checked when present.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``. Only
                their count and device are read.

        Raises:
            ValueError: The ForceField carries a ``q`` table and the batch
                declares ``graphs.total_charge``, but some graph's ``Σ_g q``
                misses its declared total by more than
                :data:`_TOTAL_CHARGE_TOL` (S5, fail-closed). A charge head that
                drifts off the constraint changes the molecule's net charge,
                and under PBC an Ewald background term absorbs that silently
                instead of reporting it.
        """
        p = self._pos(system, pos)
        self._bound_topology = self._topology_fingerprint(system, p.shape[0])
        self._exclusions = ExclusionTable(
            n_atoms=p.shape[0],
            scaling=self.forcefield.nonbonded_scaling,
            bond_index=self._bond_index(system),
            angle_index=self._angle_index(system),
            proper_index=self._proper_index(system),
            device=p.device,
        )
        charges = (self.forcefield.params("atoms") or {}).get("q")
        declared = self._ns_tensor(self._namespace(system, "graphs"), "total_charge")
        if charges is None or declared is None or declared.numel() == 0:
            return
        q = charges.detach()
        atom_batch = self._atom_batch(system, p.shape[0], p.device)
        summed = torch.zeros(declared.shape[0], dtype=q.dtype, device=q.device).index_add_(
            0, atom_batch, q
        )
        drift = (summed - declared.detach().to(summed)).abs()
        graph = int(torch.argmax(drift))
        if float(drift[graph]) > _TOTAL_CHARGE_TOL:
            raise ValueError(
                f"graph {graph}: sum of atoms.q is {float(summed[graph]):.8g} e but "
                f"graphs.total_charge declares {float(declared[graph]):.8g} e "
                f"(tolerance {_TOTAL_CHARGE_TOL:g} e)"
            )

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Hand the live simulation cell to the mlip, if it keeps cell state.

        One layer of delegation, guarded by :class:`molix.schema.CellBound`.
        The guard is not decoration: ``def_pairstyle`` accepts *any*
        ``nn.Module`` as the mlip, so an mlip with no ``bind_cell`` at all is a
        supported shape and an unconditional call would break every such model.

        A **documented no-op** when there is no mlip — the classical Class-I
        terms hold nothing sized by the lattice, and the neighbour list owns its
        own rebinding — and equally when the mlip does not satisfy
        ``CellBound``. Neither case raises.

        Args:
            cell: Lattice vectors ``(3, 3)`` — or ``(1, 3, 3)`` — in Å, rows
                being the lattice vectors. Passed through untouched, storage
                included, so an implementer that re-reads it sees the barostat's
                in-place updates.
        """
        if isinstance(self.mlip, CellBound):
            self.mlip.bind_cell(cell)

    def compile_core(self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        """Ask the mlip to install a compiled build of its own hot core.

        One layer of delegation, guarded by
        :class:`molix.schema.CoreCompilable`. The compiled core is installed on
        the **mlip**, never on this container: ``to_potentials()`` builds a
        fresh ``Potentials`` on every ``MD`` setter call, while the mlip is the
        one ``nn.Module`` the ForceField holds throughout — so a core installed
        here would be silently discarded by the next ``set_*``, and one
        installed on the mlip survives every rebuild. This class therefore
        keeps no ``_compiled_core`` and has no second evaluation path;
        :meth:`calc_energy` cannot tell whether a core is installed.

        ``compiler`` is structural — any ``callable -> callable``, such as
        :class:`molix.compile.Compiler`, which molpot must not import.

        Args:
            compiler: Applied by the mlip to whichever flat method it nominates
                as its core.

        Raises:
            RuntimeError: If there is no mlip, or the mlip does not satisfy
                ``CoreCompilable``. Failing loudly beats compiling nothing and
                reporting success.
        """
        if not isinstance(self.mlip, CoreCompilable):
            raise RuntimeError(
                "compile_core: this mlip has no compilable core "
                f"({type(self.mlip).__name__} does not implement compile_core); "
                "only an mlip satisfying molix.schema.CoreCompilable can be compiled"
            )
        self.mlip.compile_core(compiler)

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

        **The mlip branch.** When :attr:`mlip` is not ``None`` (``def_pairstyle``
        was given an ``nn.Module`` instead of a classical style name) and
        ``system`` is a ``TensorDict``, this method never sums the classical
        term energies below — it hands the whole batch to
        :func:`molpot.derivation.protocol.call_energy`, which is the single
        place that decides whether the mlip's ``_write_energy`` protocol hook or
        a plain ``model(batch)`` forward runs. Whether the mlip has an installed
        compiled core (see :meth:`compile_core`) is invisible here too — the
        mlip decides, not this branch. A ``pos`` override is written onto
        ``system["atoms", "pos"]`` *before* that call, since the mlip only ever
        reads positions off the batch — writing it after would be ignored. With
        an mlip but a non-``TensorDict`` ``system``, this falls through to the
        classical path below instead (which, with no classical styles
        declared, contributes an empty sum).

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``). Must be a ``TensorDict`` to take the mlip
                branch above.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Energy ``(B,)`` over the ``B`` molecules of the batch — shape ``()`` for
            a single molecule. Classical path: the ForceField's unit convention
            (``units="real"`` default: kcal/mol). mlip path: whatever
            :func:`~molpot.derivation.protocol.call_energy` wrote under
            ``ENERGY_KEY`` — the mlip's native energy unit (eV for the MACE
            family) — which the ForceField's declared ``units`` (``"metal"``
            for an MLIP pair style) is expected to already match; neither
            branch performs a unit conversion here.
        """
        if self.mlip is not None and isinstance(system, TensorDict):
            # The pos override lands on the batch first: the mlip reads its
            # positions off ``atoms.pos``, so writing after the call would be
            # ignored. Which entry of the mlip evaluates the energy is
            # ``call_energy``'s policy alone — this branch must not reimplement
            # it by probing the mlip. Whether a compiled core is installed is
            # likewise the mlip's own business and invisible here.
            if pos is not None:
                system["atoms", "pos"] = pos
            out = call_energy(self.mlip, system)
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

        The ambient ``torch.no_grad()`` an MD loop installs is expected: the
        energy evaluation is re-enabled locally so it builds the graph the force
        needs. ``create_graph`` is deliberately left to resolve from the
        *ambient* mode inside the backend — under an MD ``no_grad`` it stays
        ``False``. A compiled core serves both modes because buffer donation is
        off process-wide (:func:`molix.compile.register_inductor_fallbacks`);
        ``Compiler(donated_buffer=True)`` trades that back for MD-only reuse.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` and the topology.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Forces ``(N, 3)`` in the ForceField's unit convention
            (``units="real"`` default: kcal mol⁻¹ Å⁻¹).
        """
        p = self._pos(system, pos).detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.calc_energy(system, pos=p)
        return autograd_forces_from_energy(energy, p)

    def calc_energy_forces_virial(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy, forces and virial from one strained autograd backward.

        MD NPT calls this once per step. ``create_graph`` follows the ambient
        grad mode — wrap the run in ``torch.no_grad()`` so the tape is not
        kept for parameter derivatives.

        Args:
            system: Batch carrying ``atoms.pos`` and ``graphs.cell``.
            pos: Optional positions overriding ``atoms.pos``.

        Returns:
            ``(energy, forces, virial)`` with virial ``W = ∂E/∂ε`` shape
            ``(3, 3)`` in the ForceField energy unit.

        Raises:
            TypeError: If ``system`` is not a TensorDict.
            ValueError: If ``graphs.cell`` is missing.
        """
        if not isinstance(system, TensorDict):
            raise TypeError("calc_energy_forces_virial requires a TensorDict system")
        nested = system.keys(include_nested=True)
        if ("graphs", "cell") not in nested:
            raise ValueError("calc_energy_forces_virial requires graphs.cell")
        cell_buf = system["graphs", "cell"]
        batched_cell = cell_buf.dim() == 3
        cell_3 = cell_buf[0] if batched_cell else cell_buf
        pos_buf = self._pos(system, pos)
        edge_index = system["edges", "edge_index"] if ("edges", "edge_index") in nested else None
        edge_diff_buf = system["edges", "edge_diff"] if ("edges", "edge_diff") in nested else None
        # Strain the Cartesian periodic remainder, not a copy of edge_diff.
        # Passing frozen edge_diff as an independent cartesian makes local
        # graph energy (MACE / Polar) a function of strain only, so
        # ∂E/∂pos drops the short-range forces and NPT heats.
        shifts_buf = None
        if ("edges", "shifts") in nested:
            shifts_buf = system["edges", "shifts"]
        elif edge_diff_buf is not None and edge_index is not None:
            src0 = edge_index[:, 0]
            tgt0 = edge_index[:, 1]
            shifts_buf = edge_diff_buf - (pos_buf[tgt0] - pos_buf[src0])
        cartesians: tuple[torch.Tensor, ...] = (shifts_buf,) if shifts_buf is not None else ()

        def energy_fn(
            pos_s: torch.Tensor, cell_s: torch.Tensor, *strained: torch.Tensor
        ) -> torch.Tensor:
            system["atoms", "pos"] = pos_s
            system["graphs", "cell"] = cell_s.unsqueeze(0) if batched_cell else cell_s
            if edge_index is not None:
                src = edge_index[:, 0]
                tgt = edge_index[:, 1]
                diff = pos_s[tgt] - pos_s[src]
                if strained:
                    diff = diff + strained[0]
                system["edges", "edge_diff"] = diff
                system["edges", "edge_dist"] = diff.norm(dim=-1)
            energy = self.calc_energy(system, pos=pos_s)
            return energy if energy.ndim == 0 else energy.sum()

        energy, forces, virial = autograd_forces_and_virial(energy_fn, pos_buf, cell_3, *cartesians)
        # Strain writes replace graphs.cell / pos / edge_diff with graph
        # tensors. Put the MD buffers back so the barostat copy_ and the
        # next neighbour update still see the same storage.
        system["atoms", "pos"] = pos_buf
        system["graphs", "cell"] = cell_buf
        if edge_diff_buf is not None:
            system["edges", "edge_diff"] = edge_diff_buf
            system["edges", "edge_dist"] = edge_diff_buf.norm(dim=-1)
        graphs_bs = system["graphs"].batch_size
        stored = energy.detach().reshape(*graphs_bs) if graphs_bs else energy.detach().reshape(())
        write_energy(system, stored)
        write_forces(system, forces)
        write_virial(system, virial)
        return energy, forces, virial

    def calc_virial(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Virial ``W = ∂E/∂ε`` at ``ε = 0``, shape ``(3, 3)``.

        Thin wrapper over :meth:`calc_energy_forces_virial`. Prefer that
        method when forces are needed in the same step.
        """
        _, _, virial = self.calc_energy_forces_virial(system, pos=pos)
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
            kcal/mol and kcal mol⁻¹ Å⁻¹). mlip path: the mlip's own energy
            unit (eV for the MACE family), never a sum with classical terms.

        Raises:
            ValueError: A declared style's parameter table is absent — the same
                contract as :meth:`term_energies`, checked branch for branch so
                the two routes agree on which ForceFields evaluate at all.
        """
        if self.mlip is not None and isinstance(system, TensorDict):
            # Same branch policy as ``calc_energy``: the mlip owns the whole
            # energy. Without this, an mlip ForceField — which declares no
            # classical style — falls through the term loop below and returns a
            # silent ``(0, zeros)`` instead of its energy and forces.
            leaf = self._pos(system, pos).detach().requires_grad_(True)
            with torch.enable_grad():
                energy = self.calc_energy(system, pos=leaf)
            return energy, autograd_forces_from_energy(energy, leaf)

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
            self._require_bonded_params("bond", "bonds", self.bond)
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
            self._require_bonded_params("angle", "angles", self.angle)
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
            self._require_bonded_params("dihedral", "propers", self.proper)
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
            self._require_bonded_params("improper", "impropers", self.improper)
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
            (rep, disp), f = self._lj_energy_forces(system, p, zero, n_atoms)
            energy = energy + rep + disp
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

        The keys are exactly the :class:`~molpot.contract.TermSpec` keys of the
        declared styles' contracts — ``bonds`` / ``angles`` / ``propers`` /
        ``impropers`` / ``lj_repulsion`` / ``lj_dispersion`` / ``coulomb`` — so a
        bonded-only ForceField yields no LJ and no ``coulomb`` key even when
        ``epsilon`` / ``sigma`` or charges already sit in its parameter tables.
        ``pair:lj/cut`` contributes **two** keys from one evaluation (S3): the
        repulsive wall ``4ε(σ/r)¹²`` and the dispersive tail ``−4ε(σ/r)⁶``, whose
        sum is the LJ energy. A declared style whose *topology* is empty
        contributes an explicit zero; a declared style whose *parameter table* is
        absent raises instead of contributing a silent zero.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Term name → energy ``(B,)`` (``()`` for a single molecule) in the
            ForceField's unit convention (``units="real"`` default: kcal/mol).

        Raises:
            ValueError: A declared style's parameter table is absent.
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
            self._require_bonded_params("bond", "bonds", self.bond)
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
            self._require_bonded_params("angle", "angles", self.angle)
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
            self._require_bonded_params("dihedral", "propers", self.proper)
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
            self._require_bonded_params("improper", "impropers", self.improper)
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
            terms["lj_repulsion"], terms["lj_dispersion"] = self._lj_energy(system, p, zero)
        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            terms["coulomb"] = self._coulomb_energy(system, p, zero)
        return terms

    def contribution_energies(
        self,
        system: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> dict[Contribution, torch.Tensor]:
        """Term energies grouped into the S-decomposition buckets.

        The physical partition is ``E = E_local + E_disp + E_perm + E_pol +
        E_CT``. Each declared style's contract labels its terms, and this method
        only *sums* :meth:`term_energies` by that label — it is a grouping of the
        one evaluation, never a second energy path, so the returned values add up
        to :meth:`calc_energy` by construction. In tree today: ``bonds`` /
        ``angles`` / ``propers`` / ``impropers`` and ``lj_repulsion`` →
        :attr:`~molpot.contract.Contribution.LOCAL`, ``lj_dispersion`` →
        :attr:`~molpot.contract.Contribution.DISP`, ``coulomb`` →
        :attr:`~molpot.contract.Contribution.PERM`.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Contribution → summed energy ``(B,)`` (``()`` for a single molecule)
            in the ForceField's unit convention (``units="real"`` default:
            kcal/mol). Only contributions with at least one declared term appear,
            so a bonded-only ForceField returns ``{LOCAL: ...}`` alone.

        Raises:
            ValueError: A declared style's parameter table is absent (from
                :meth:`term_energies`).
        """
        labels = {
            spec.key: spec.contribution
            for key in self.forcefield.style_names()
            if not key.startswith("atom:")
            for spec in PotentialMeta.contract_for(key).terms
        }
        grouped: dict[Contribution, torch.Tensor] = {}
        for name, energy in self.term_energies(system, pos=pos).items():
            contribution = labels[name]
            previous = grouped.get(contribution)
            grouped[contribution] = energy if previous is None else previous + energy
        return grouped

    # ------------------------------------------------------------------
    # nonbonded
    # ------------------------------------------------------------------

    def _charges(self) -> torch.Tensor:
        """Partial charges ``(N,)`` of the declared ``pair:coulomb`` style.

        Raises:
            ValueError: ``atoms.q`` is absent. A declared style with no table
                contributed a silent zero before (F2).
        """
        atoms = self.forcefield.params("atoms") or {}
        if "q" not in atoms:
            raise ValueError(
                "style 'pair:coulomb' is declared but its parameter table is absent: "
                "no q (partial charges) under atoms"
            )
        return atoms["q"]

    def _lj_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom ``(epsilon, sigma)`` of the declared ``pair:lj/cut`` style.

        Raises:
            ValueError: ``atoms.epsilon`` or ``atoms.sigma`` is absent.
        """
        atoms = self.forcefield.params("atoms") or {}
        if "epsilon" not in atoms or "sigma" not in atoms:
            missing = [name for name in ("epsilon", "sigma") if name not in atoms]
            raise ValueError(
                "style 'pair:lj/cut' is declared but its parameter table is absent: "
                f"no {' / '.join(missing)} under atoms"
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Half neighbor list ``(edge_index, dist, scales, live)`` for one pair term.

        The list keeps the capacity the builder returned it at — ``live`` is the
        boolean mask of the rows that contribute, not a selection to apply.
        ``None`` when there is no list at all (fewer than two atoms, or an
        enumeration that produced no pair). ``scales`` is the exclusion factor
        of every row including the dead ones (which read ``1.0``), so the term
        multiplies instead of slicing.

        The one seam every pair term reaches for, so it is where the bound
        table is checked against the batch. Binds the topology on first use
        (:meth:`bind_topology`) — eager convenience, not a rebind — and from
        then on compares the batch's shape-level fingerprint against the bound
        one, raising when they differ instead of scaling this batch's pairs by
        the previous topology's relations. A ``Potentials`` reused over a
        different topology must be rebound explicitly.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` and the topology
                namespaces, as :meth:`bind_topology` reads them.
            pos: Positions ``(N, 3)`` in Å the list is enumerated from.
            cutoff: Real-space cutoff ``r_c`` in Å of the asking term, or
                ``None`` to keep every pair the builder produced.
            scale_kind: Interaction class of the asking term — one of the
                ForceField's :meth:`~molpot.ir.NonbondedScaling.kinds`
                (``"q"`` for Coulomb, ``"lj"`` for Lennard-Jones).

        Returns:
            ``(edge_index, dist, scales, live)`` — pairs ``(E, 2)``,
            separations ``(E,)`` in Å, dimensionless scale factors ``(E,)``
            and the boolean liveness mask ``(E,)`` — or ``None`` when the
            builder produced no list.

        Raises:
            ValueError: The batch's topology has a different shape than the
                bound one (atom count, or the ``bonds`` / ``angles`` /
                ``propers`` column counts). The message names
                :meth:`bind_topology`, which is the fix.
        """
        if self._exclusions is None:
            self.bind_topology(system, pos=pos)
        else:
            current = self._topology_fingerprint(system, pos.shape[0])
            if current != self._bound_topology:
                raise ValueError(
                    "this Potentials is bound to a topology with fingerprint "
                    f"{self._bound_topology} but the batch has {current} "
                    "(n_atoms, bonds, angles, propers; -1 = namespace absent); its "
                    "exclusion table is never rebuilt implicitly — call "
                    "bind_topology(system) for the new topology"
                )
        assert self._exclusions is not None
        atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
        cell = self._cell(system)
        edges = self._namespace(system, "edges")
        built = self._neighbors.build(
            pos=pos, atom_batch=atom_batch, cell=cell, cutoff=cutoff, edges=edges
        )
        if built is None:
            return None
        edge_index, dist, live = built
        scales = self._exclusions.lookup(edge_index, scale_kind, dtype=pos.dtype)
        return edge_index, dist, scales, live

    def _lj_energy(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(repulsion, dispersion)`` of the LJ style — zeros when no pair is in range."""
        assert self.lj is not None
        epsilon, sigma = self._lj_params()
        # Keep LJ on its own cutoff list. Reusing an unlimited Coulomb half-list
        # forced the exclusion lookup (and distance storage) onto all
        # intramolecular pairs even though LJ only needs ``r < cutoff``.
        pairs = self._scaled_pairs(system, pos, cutoff=self.lj.cutoff, scale_kind="lj")
        if pairs is None:
            return zero, zero
        edge_index, dist, scales, live = pairs
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
            edge_mask=live,
        )

    def _lj_energy_forces(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
        n_atoms: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """``((repulsion, dispersion), forces)`` of the LJ style."""
        assert self.lj is not None
        epsilon, sigma = self._lj_params()
        pairs = self._scaled_pairs(system, pos, cutoff=self.lj.cutoff, scale_kind="lj")
        if pairs is None:
            return (zero, zero), pos.new_zeros((n_atoms, 3))
        edge_index, dist, scales, live = pairs
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
            edge_mask=live,
        )

    def _coulomb_energy(
        self,
        system: Any,
        pos: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        q = self._charges()
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")

        cell = self._cell(system)
        if cell is None:
            assert self.coulomb_open is not None
            cut = self.coulomb_open.cutoff
            pairs = self._scaled_pairs(system, pos, cutoff=cut, scale_kind="q")
            if pairs is None:
                return zero
            edge_index, dist, scales, live = pairs
            num_graphs = self._num_graphs(system)
            atom_batch = self._atom_batch(system, pos.shape[0], pos.device)
            return self.coulomb_open.energy(
                edge_index=edge_index,
                distances=dist,
                charges=q,
                scales=scales,
                edge_batch=atom_batch[edge_index[:, 0]] if num_graphs > 1 else None,
                num_graphs=num_graphs,
                edge_mask=live,
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
            edge_index, dist, scales, live = pairs
            # The periodic term still selects its real-space rows by
            # ``scales > 0``; folding liveness into the scale drops the dead
            # rows through that same test (a0-03 rewrites the term itself).
            scales = scales * live.to(scales.dtype)
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

    def _topology_fingerprint(self, system: Any, n_atoms: int) -> tuple[int, ...]:
        """Shape-level identity of a batch's topology: ``(N, n_b, n_a, n_p)``.

        Column *counts* only — the last dimension of each index block, or
        ``-1`` when the namespace is absent — read off ``.shape``, so nothing
        here touches a tensor value: positions move every MD step and a fresh
        batch of the same molecule brings new tensors, and both must keep the
        bound table. It is what :meth:`_scaled_pairs` compares, and it cannot
        distinguish two topologies of equal shape (see :meth:`bind_topology`).

        Built through the same readers :meth:`bind_topology` hands the table,
        so the guard can never disagree with the table it guards about what a
        bond / angle / proper column is. They stack their columns; only the
        shape of the result is read here, which a compiled trace folds to a
        constant and drops.

        Args:
            system: The batch whose ``bonds`` / ``angles`` / ``propers``
                namespaces are read, exactly as :meth:`bind_topology` reads them.
            n_atoms: Atom count ``N`` the table would be keyed against.

        Returns:
            ``(n_atoms, n_bonds, n_angles, n_propers)``, each ``-1`` when that
            namespace carries no usable index block.
        """
        counts = [int(n_atoms)]
        for reader in (self._bond_index, self._angle_index, self._proper_index):
            index = reader(system)
            counts.append(-1 if index is None else int(index.shape[-1]))
        return tuple(counts)

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
