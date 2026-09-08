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
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict, TensorDictBase

from molix.batch import TensorFrame, as_tensor_frame
from molix.schema import CellBound, CoreCompilable
from molpot.contract import Contribution, ExclusionClass
from molpot.derivation.force import (
    autograd_forces_and_virial,
    autograd_forces_from_energy,
    position_leaf,
)
from molpot.derivation.protocol import (
    ENERGY_KEY,
    FORCES_KEY,
    call_energy,
    write_energy,
    write_forces,
    write_virial,
)
from molpot.forcefield.exclusions import ExclusionTable
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
from molpot.potentials.vdw.lj126 import MixingRule

if TYPE_CHECKING:
    from molpot.forcefield.container import ForceField

__all__ = ["PAIR_STYLE_DEFAULTS", "PairStyleDefaults", "Potentials"]


@dataclass(frozen=True)
class PairStyleDefaults:
    """Values :meth:`Potentials._assemble_pair_potentials` falls back to.

    A pair style declared bare (``def_pairstyle("lj/cut")``) still runs at some
    cutoff and through some solver. Those values are model identity, so they
    live here once and are read in exactly one place:
    :meth:`ForceField._nonbonded_method_record
    <molpot.forcefield.container.ForceField._nonbonded_method_record>`, whose
    :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` both
    :meth:`Potentials._assemble_pair_potentials` and the recording path
    (:meth:`ForceField.to_ir <molpot.forcefield.ForceField.to_ir>`,
    :meth:`ForceField.manifest <molpot.forcefield.ForceField.manifest>`) then
    consume. A second reading of these defaults on either side would let a
    manifest report a model that was never assembled.

    Only the options that *have* a fallback appear. The Lennard-Jones switching
    window and the open-boundary Coulomb cutoff have none: unset they mean "no
    window" and "no truncation", which are absences rather than radii. The
    real-space split radius does have one, but a *derived* one — it is a
    multiple of the smearing, so it appears below as
    :attr:`real_space_cutoff_factor` rather than as an absolute length.

    Attributes:
        lj_cutoff: Lennard-Jones cutoff radius in Å.
        lj_mixing: Combining rule a ``pair:lj/cut`` style declared without a
            ``mixing`` option runs with — the value of a
            :class:`~molpot.potentials.vdw.lj126.MixingRule`. Lorentz–Berthelot
            (``σ`` arithmetic, ``ε`` geometric), the AMBER / CHARMM rule; a
            product form parameterized the other way declares its own, as
            :class:`~molpot.forcefield.OPLSAAForceField` does with
            ``"geometric"``.
        coulomb_calculator: Reciprocal-space algorithm under a cell. The mesh
            is the default; the explicit ``"ewald"`` k-vector sum is the
            small-box reference, not what a real periodic system should pay.
        smearing: Gaussian charge-splitting width of the Ewald decomposition,
            in Å.
        real_space_cutoff_factor: Multiple of :attr:`smearing` at which the
            real-space half of the Ewald split is truncated when no
            ``real_space_cutoff`` option is declared (dimensionless). At three
            widths the complementary error function has fallen to ``~2e-5`` of
            the bare ``1/r``, so what is dropped there is below the accuracy of
            the reciprocal sum that replaces it.
        lr_wavelength: Shortest reciprocal wavelength resolved by ``"ewald"``,
            in Å.
        mesh_spacing: Grid spacing of the ``"pme"`` / ``"p3m"`` mesh, in Å.
        capacity_factor: Dimensionless head-room on the frozen reciprocal
            shape; ``1.0`` sizes for the bound cell exactly.
    """

    lj_cutoff: float = 10.0
    lj_mixing: str = MixingRule.LORENTZ_BERTHELOT
    coulomb_calculator: str = "pme"
    smearing: float = 5.0
    real_space_cutoff_factor: float = 3.0
    lr_wavelength: float = 1.0
    mesh_spacing: float = 1.0
    capacity_factor: float = 1.0


#: The single instance of :class:`PairStyleDefaults` the resolver reads.
PAIR_STYLE_DEFAULTS = PairStyleDefaults()

#: Largest per-graph deviation of ``Σ_g q`` from the declared
#: ``graphs.total_charge`` that :meth:`Potentials.bind_topology` accepts (e).
#: Sized for fp32 accumulation over a few thousand atoms, far below the
#: 0.01 e a drifting learned charge head produces.
_TOTAL_CHARGE_TOL = 1e-6

#: Floor applied to a squared separation before its ``sqrt`` in
#: :meth:`Potentials._placed_pairs` (Å²). ``sqrt``'s gradient at zero is
#: ``inf``/``NaN`` and a padding self-loop is exactly a zero vector, so the
#: clamp has to happen *before* the root — no later multiply-by-zero mask can
#: undo a NaN. Far below any physical separation, so no live pair is moved.
_SQ_DIST_CLAMP_MIN = 1e-24


def _resolved(value: float | None, option: str) -> float:
    """One numeric field of a declared style's nonbonded record, as a number.

    Every numeric field of
    :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` is filled together with
    the style it belongs to, so a ``None`` reaching a term the same record just
    said to build means the resolver and the assembly disagree.

    Args:
        value: The record field.
        option: Its style-option name, for the message.

    Returns:
        ``value``.

    Raises:
        ValueError: ``value`` is ``None`` — a defect in
            :meth:`ForceField._nonbonded_method_record
            <molpot.forcefield.container.ForceField._nonbonded_method_record>`
            rather than a user error, raised here instead of being defaulted
            back to a number the record never reported.
    """
    if value is None:
        raise ValueError(
            f"the nonbonded record declares this pair style but leaves {option!r} unset; "
            "ForceField._nonbonded_method_record resolves every field of a declared style"
        )
    return value


class Potentials(BasePotential):
    """Classical Class-I energy as a molpot :class:`BasePotential` (``nn.Module``).

    Embed in a larger torch model, or reach MD through the owning
    :class:`~molpot.forcefield.container.ForceField` and
    :meth:`molix.md.MD.set_forcefield`. ``forward`` on a TensorDict
    writes ``graphs.energy`` and ``atoms.forces``.

    Those forces come from :meth:`calc_energy_forces`, which is **mixed**: the
    bonded terms and ``pair:lj/cut`` evaluate a closed-form
    ``F = -∂E/∂pos`` (the *analytic* branch — no ``requires_grad`` position
    leaf, no backward pass), while Coulomb and an mlip pair style have no
    cheap closed form and fall back to one ``torch.autograd.grad`` on a fresh
    position leaf. :meth:`calc_forces` is the all-autograd route and
    :meth:`calc_energy_forces_virial` the strained one NPT uses; the three
    agree to numerical precision on the same batch, so the analytic branch is a
    cost choice and never a physics choice. It does constrain what a batch must
    carry: an analytic pair force needs the minimum-image separation *vector*,
    which :meth:`_placed_pairs` forms and hands to
    :meth:`LJCutPotential.energy_forces
    <molpot.forcefield.pair_terms.LJCutPotential.energy_forces>` together with
    its norm.

    Bonded terms use existing harmonic / torsion kernels. Nonbonded terms are
    concrete Potential objects that own their cutoffs:

    * ``pair:lj/cut`` → :class:`LJCutPotential`. Style options, both in Å and
      both owned by the Potential: ``cutoff`` — the radius ``r_c`` beyond
      which a pair contributes nothing (default
      :attr:`PairStyleDefaults.lj_cutoff`) — and the optional
      ``switch_start`` ``r_s`` with ``0 < r_s < r_c``, which tapers the energy
      smoothly to zero over ``[r_s, r_c]`` with the quintic switch instead of
      truncating it at ``r_c``. Omitting ``switch_start`` keeps the hard
      truncation. Its third option is dimensionless: ``mixing``, the combining
      rule making per-pair ``ε`` / ``σ`` out of the per-atom tables
      (``"lorentz-berthelot"``, the default, or ``"geometric"`` — see
      :class:`~molpot.potentials.vdw.lj126.MixingRule`).
    * ``pair:coulomb`` open → :class:`CoulombOpenPotential` (optional
      ``cutoff`` in Å; ``prefactor`` in kcal Å mol⁻¹ e⁻²)
    * ``pair:coulomb`` + cell → :class:`CoulombPeriodicPotential`, whose style
      options are ``smearing`` (Gaussian splitting width, Å),
      ``real_space_cutoff`` (Å, default
      :attr:`PairStyleDefaults.real_space_cutoff_factor` × ``smearing``),
      ``capacity_factor`` (dimensionless head-room on the frozen reciprocal
      shape), ``lr_wavelength`` (Å, ``ewald`` only), ``mesh_spacing`` (Å,
      ``pme`` / ``p3m`` only), ``prefactor`` (kcal Å mol⁻¹ e⁻²) and
      ``calculator=pme|ewald|p3m``, defaulting to
      :attr:`PairStyleDefaults.coulomb_calculator` (the ``pme`` mesh) — the
      explicit ``O(N²)`` ``ewald`` k-vector sum is the small-box reference,
      not what a real periodic system should get by default

    Every option left undeclared takes its value from
    :data:`PAIR_STYLE_DEFAULTS`, applied in one place only
    (:meth:`ForceField._nonbonded_method_record
    <molpot.forcefield.container.ForceField._nonbonded_method_record>`), so the
    numbers above are stated there and not duplicated per term.

    Every pair term above reads the ``edges`` neighbour list already on the
    batch; molpot never enumerates pairs of its own, so a batch evaluated by a
    ForceField with any pair style must carry one (:meth:`_placed_pairs`). The
    radius that list must reach is the field's, not the caller's, so the field
    hands it out (:meth:`ForceField.neighbor_list
    <molpot.forcefield.container.ForceField.neighbor_list>`)::

        ff.def_pairstyle("lj/cut", cutoff=12.0, switch_start=10.0)
        ff.def_pairstyle("coulomb", smearing=5.0, capacity_factor=1.2)
        pots = ff.to_potentials()
        # molix builds the cutoff graph; Potentials only prices it. Under a
        # cell the radius is the Ewald split's 3 x smearing = 15 A, not the
        # Lennard-Jones 12 A — the field resolves which of the two is larger.
        system = ff.neighbor_list(pbc=True).execute(system)   # system has graphs.cell
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
        self.lj: LJCutPotential | None = None
        self.coulomb_open: CoulombOpenPotential | None = None
        self.coulomb_periodic: CoulombPeriodicPotential | None = None
        self.mlip = getattr(forcefield, "_mlip", None)
        self._exclusions: ExclusionTable | None = None
        self._bound_topology: tuple[int, ...] | None = None
        # How many graphs the bound batch holds, resolved once at
        # :meth:`bind_topology` because reading it off ``atoms.batch`` is a host
        # sync and therefore a graph break on a compiled core (:meth:`_num_graphs`).
        self._bound_num_graphs: int | None = None
        # The frame whose geometry last passed :meth:`_check_bonds_unwrapped`:
        # a weak reference to the batch object and its topology fingerprint.
        # Separate from ``_bound_topology`` because a bonded-only field never
        # binds an exclusion table and would otherwise never be checked at all,
        # and keyed on the batch rather than on its shape because a new frame of
        # the same shape is a new geometry (:meth:`_check_geometry`).
        self._checked_geometry: tuple[weakref.ref[Any], tuple[int, ...]] | None = None
        # The typed view last built for a batch, keyed on that batch object by
        # weak reference (:meth:`_frame`). Not a cache for speed: the view's
        # *identity* is what the geometry memo above is keyed on, so a fresh
        # wrapper per call would re-run the F29 host sync on every MD step and
        # put it inside any trace that reaches it.
        self._frame_view: tuple[weakref.ref[Any], TensorFrame] | None = None
        # The view of "no system at all" — the ``BasePotential`` ``pos=``-only
        # call — built once so that path is memoized like any other frame.
        self._empty_frame: TensorFrame | None = None
        self._bonded_styles: frozenset[str] = frozenset()
        # The compiled build of :meth:`_classical_core`, installed by
        # :meth:`compile_core` on a field with no mlip. ``None`` means "call the
        # eager bound method"; it is the *same* function either way, so there is
        # no second evaluation path (see :meth:`compile_core`).
        self._compiled_classical: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
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
        """Build the pair terms from the ForceField's resolved nonbonded record.

        :meth:`ForceField._nonbonded_method_record
        <molpot.forcefield.container.ForceField._nonbonded_method_record>` is
        the **one** resolver of the pair-style options: it applies
        :data:`PAIR_STYLE_DEFAULTS` to whatever ``def_pairstyle`` declared, and
        both the terms built here and the record ``to_ir`` and the manifest
        carry come out of that single reading. Resolving the options a second
        time here is one edit away from a manifest that describes a model which
        was never assembled.

        Each field of the record is ``None`` exactly when the style it belongs
        to was never declared, so the two branches below read "the ForceField
        declares this style, and here is what it resolved to".

        The Coulomb ``prefactor`` is the one style option with no field on the
        record: ``k_e`` is a unit-system constant, not a nonbonded *method*, so
        it stays a direct read of the style options.
        """
        ff = self.forcefield
        method = ff._nonbonded_method_record()
        if method.lj_cutoff is not None:
            # ``switch_start`` is the second half of the ``lj/cut`` cutoff option
            # and travels with it: dropped here, the quintic taper on
            # :class:`LJCutPotential` would be unreachable from any ForceField and
            # every assembled Class-I model would keep the hard truncation.
            # ``None`` (no window) stays ``None``; the ``0 < r_s < r_c`` check is
            # the term's, so a bad window raises there rather than twice.
            # The combining rule travels with them: it is model identity (an
            # OPLS-AA table mixed Lorentz–Berthelot is a different force field
            # on every cross pair), and the record is where the declaration
            # already resolved it. ``MixingRule`` refuses anything else, so a
            # record naming a cutoff but no rule fails here rather than
            # silently mixing by whichever kernel default is current.
            self.lj = LJCutPotential(
                cutoff=method.lj_cutoff,
                switch_start=method.lj_switch_start,
                mixing=MixingRule(method.lj_mixing),
            )
        if method.coulomb_calculator is not None:
            # Periodic calculator is chosen when a cell is present at energy
            # time; build both the open and the periodic term here.
            pref = float(ff.style_options("pair:coulomb").get("prefactor", kcalmol_A))
            self.coulomb_open = CoulombOpenPotential(
                cutoff=method.coulomb_cutoff,
                prefactor=pref,
            )
            # The mesh method is the default under a cell: ``ewald`` sums every
            # k-vector explicitly, which is the reference a small-box finite
            # difference wants and not what a real periodic system should pay.
            self.coulomb_periodic = CoulombPeriodicPotential(
                smearing=_resolved(method.smearing, "smearing"),
                calculator=method.coulomb_calculator,
                lr_wavelength=_resolved(method.lr_wavelength, "lr_wavelength"),
                mesh_spacing=_resolved(method.mesh_spacing, "mesh_spacing"),
                prefactor=pref,
                # Explicit, so the term's own ``3 × smearing`` fallback is not
                # what the assembled model silently runs at while the record
                # reports the same number from the other side of the seam.
                real_space_cutoff=_resolved(method.real_space_cutoff, "real_space_cutoff"),
                capacity_factor=_resolved(method.capacity_factor, "capacity_factor"),
            )

    def _assemble_bonded(self) -> None:
        ff = self.forcefield
        # Which bonded categories the ForceField declares, resolved **once**.
        # Every energy path branches on this, and asking the ForceField again
        # would walk its molrs-side style registry — a call dynamo cannot
        # trace, i.e. one graph break per branch on the hot path. The answer is
        # exactly as stale-proof as the terms assembled just below: both are
        # frozen here, and ``to_potentials()`` builds a fresh container.
        self._bonded_styles = frozenset(
            category
            for category, declared in (
                ("bond", ff.bond_style()),
                ("angle", ff.angle_style()),
                ("dihedral", ff.dihedral_style()),
                ("improper", ff.improper_style()),
            )
            if declared is not None
        )
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
        """Energy, or the batch with ``graphs.energy`` / ``atoms.forces`` written.

        A batch is treated as an MD / composer system: energy and forces are
        written in place and the batch is returned so the module composes into
        the MD driver's TensorDict force adapter. "A batch" is the post-collate
        container in any of its spellings — the plain ``TensorDict`` and its
        :class:`~molix.batch.TensorFrame` view alike, since the view *is* the
        caller's leaves (:meth:`_frame`) and a write through it lands in the
        caller's own storage. Answering a view with a bare energy tensor
        instead would leave that caller's batch with no forces in it and
        nothing raised.

        The energy is reshaped to the ``graphs`` namespace's own batch size on
        the way in — the same defence :meth:`calc_energy_forces_virial` applies
        before its ``write_energy``. :meth:`calc_energy_forces` answers a
        single-molecule batch with a **scalar** ``()``, which is the right shape
        for the sum it is, while a collated batch of one graph carries
        ``graphs`` at ``batch_size=[1]`` and a TensorDict refuses to store a
        ``()`` leaf under it. Every collate in molnex produces exactly that
        batch, so without the reshape the most ordinary case is the one that
        raises.
        """
        writes_back = isinstance(data, TensorDictBase)
        frame = self._frame(data)
        if self.mlip is not None and writes_back and kwargs.get("pos") is None:
            return self.mlip(frame)
        energy, forces = self.calc_energy_forces(frame, pos=kwargs.get("pos"))
        if writes_back:
            if "graphs" not in frame.keys():
                frame["graphs"] = TensorDict({}, batch_size=[])
            graphs_bs = frame.graphs.batch_size
            # A ``batch_size=[]`` graphs namespace constrains nothing, so the
            # energy is stored as produced; a sized one is the shape to answer in.
            frame[ENERGY_KEY] = energy.reshape(*graphs_bs) if graphs_bs else energy
            frame[FORCES_KEY] = forces
            # The caller's own container back, not the view: the writes above
            # went into its storage, so the two carry the same batch and the
            # one that was handed in is the one that composes onward.
            return data
        return energy

    def bind_topology(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
        *,
        pos: torch.Tensor | None = None,
    ) -> None:
        """Build this topology's exclusion table, and check the declared net charge.

        Called **once per topology** — the connectivity analogue of
        :meth:`molix.schema.CellBound.bind_cell`, which is called once per
        cell. :meth:`_scaled_pairs` calls it lazily on the first nonbonded
        evaluation, so MD and a single-batch score need no explicit call; a
        caller that reuses one ``Potentials`` over a **different** topology
        must call it again, because positions may change freely between energy
        calls but the bond / angle / torsion columns may not. Repeated calls
        are harmless: the table is rebuilt from scratch each time.

        **All or nothing.** Every check below runs before a single field of this
        container is written, so a bind that raises leaves the container exactly
        as it found it — still unbound if it was unbound, still on the previous
        topology if it was bound. That is not tidiness: the state a half-bind
        used to leave behind was *load-bearing*, because
        :meth:`calc_energy_forces` binds only when ``_exclusions`` is ``None``.
        A refused bind that had already installed the table therefore disarmed
        the very guard that would have refused the next call, and the batch the
        container had just rejected got a number back. It is
        :meth:`_check_composition`'s principle — a refused ForceField never
        leaves a half-built container behind — applied one stage later.

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
        host-side, once per topology instead of once per step. The
        **unwrapped-molecule** check (finding F29) has the same cost model but
        not the same home: a bonded-only field never binds, and it is exactly
        the composition that check exists for, so it belongs to
        :meth:`_check_geometry` — its own once-per-frame gate, which every
        public entry point calls and which this bind calls too. Under a
        ``graphs.cell`` every bond is measured against half the cell's minimum
        perpendicular width, and a molecule folded across a box face — every
        bonded term of which would then be priced across the whole box — is
        refused instead of answered.

        What that check can vouch for ends at the frame it read. Its record,
        like the fingerprint above, is shape-level, so positions stay free
        afterwards. MD is covered by construction — the integrator never wraps
        in place, so a rollout whose initial frame is whole stays whole — while
        a **training batch** is vouched for only at the first call that read
        its shape: a data pipeline that wraps frames (the normal state of a
        trajectory read back from disk) must unwrap them before they arrive,
        and no later stage will notice if it does not.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers``).
                ``graphs.total_charge`` ``(B,)`` in e is checked when present,
                and so is the orientation of ``edges.edge_index``
                (:meth:`_check_pair_orientation`) — bind time is the one place
                that check is affordable, and a mis-oriented list is otherwise
                a silent zero rather than an error. ``graphs.cell`` ``(3, 3)``
                or ``(B, 3, 3)`` in Å, with ``bonds``, additionally puts the
                frame through :meth:`_check_bonds_unwrapped`.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``. Their
                count and device are read, and — under a cell — their bonded
                displacements, since they are the geometry the energy would be
                priced at.

        Raises:
            ValueError: The ForceField carries a ``q`` table and the batch
                declares ``graphs.total_charge``, but some graph's ``Σ_g q``
                misses its declared total by more than
                :data:`_TOTAL_CHARGE_TOL` (S5, fail-closed). A charge head that
                drifts off the constraint changes the molecule's net charge,
                and under PBC an Ewald background term absorbs that silently
                instead of reporting it. Or the batch's placed edge list holds
                pair rows but not one with ``source < target`` — the half that
                :meth:`_placed_pairs` prices — so every pair term would cost
                exactly zero. Or, under a cell, some bond is longer than half
                its cell's minimum perpendicular width
                (:meth:`_check_bonds_unwrapped`).
        """
        # Every refusal first, then every mutation — the same rule
        # :meth:`_check_composition` applies to assembly, applied to binding.
        # Assigning as we went left a refused bind *armed*: the fingerprint, the
        # graph count and the exclusion table were already installed when the
        # charge check raised, so the very next ``calc_energy`` on that batch
        # found ``_exclusions`` set, skipped its own bind and answered — 97.7
        # kcal/mol out of a container that had just refused the frame. So the
        # three bound fields are assigned at the very bottom, after everything
        # that can raise. (The geometry record in between is the gate's own
        # bookkeeping, not a bind: it says a frame passed, which it did.)
        frame = self._frame(system)
        p = self._pos(frame, pos)
        self._check_pair_orientation(frame)
        charges = (self.forcefield.params("atoms") or {}).get("q")
        declared = frame.graphs.get("total_charge", None) if "graphs" in frame.keys() else None
        if charges is not None and declared is not None and declared.numel() != 0:
            q = charges.detach()
            atom_batch = self._atom_batch(frame, p.shape[0], p.device)
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
        # An explicit bind rebuilds everything from scratch, the geometry record
        # included: rebinding the *same batch object* into a new lattice (its
        # cell written in place) is the one call that would otherwise ride the
        # memo, and it is exactly the call that means "check this again". The
        # gate hands back the fingerprint it built, so the bind does not build a
        # second one.
        self._checked_geometry = None
        fingerprint = self._check_geometry(frame, p)
        # Host-side, once: how many graphs this batch holds. Off the hot path
        # it is the one number an evaluation cannot read without a device sync
        # (see :meth:`_num_graphs`), and a bind is where syncs are allowed.
        num_graphs = self._count_graphs(frame)
        exclusions = ExclusionTable(
            n_atoms=p.shape[0],
            scaling=self.forcefield.nonbonded_scaling,
            bond_index=self._bond_index(frame),
            angle_index=self._angle_index(frame),
            proper_index=self._proper_index(frame),
            device=p.device,
        )
        self._bound_topology = fingerprint
        self._bound_num_graphs = num_graphs
        self._exclusions = exclusions

    def _check_pair_orientation(self, frame: TensorFrame) -> None:
        """Refuse a placed list that :meth:`_placed_pairs` would read as all dead.

        :meth:`_placed_pairs` prices the ``source < target`` half of the
        bidirectional list :class:`molix.batch.NeighborList` writes at its
        ``symmetry=True`` default — one orientation of each pair, chosen by the
        rule rather than by the order the rows happen to be in. Its
        ``symmetry=False`` mode emits the *other* half (``source > target``), a
        list every pair term then multiplies by an all-false mask: no raise, no
        missing key, every nonbonded energy exactly zero.

        Called from :meth:`bind_topology`, which is host-side and runs once per
        topology, so the single ``.any()`` sync it costs never reaches a
        compiled step. Self-loop rows (a capacity list's dead padding) are not
        evidence either way, which is why the refusal needs a row that actually
        runs ``source > target``.

        Args:
            frame: The batch's typed view, optionally carrying
                ``edges.edge_index`` ``(E, 2)``. No list, or a list of no rows,
                is nothing to check.

        Raises:
            ValueError: Some row runs ``source > target`` and none runs
                ``source < target``.
        """
        edges = frame.edges if "edges" in frame.keys() else None
        edge_index = None if edges is None else edges.get("edge_index", None)
        if edge_index is None or edge_index.shape[0] == 0:
            return
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        # One host sync, folded into a single boolean: "reversed rows exist and
        # forward ones do not".
        if bool((src > tgt).any() & ~(src < tgt).any()):
            raise ValueError(
                "this batch's edges.edge_index carries only source > target rows: "
                "Potentials prices the source < target half of a bidirectional list, so "
                "every pair term would silently cost zero on this one. "
                "molix.batch.NeighborList(symmetry=False) produces exactly this "
                "upper-triangle-reversed half list — build it with the symmetry=True "
                "default (both orientations, one of them priced), or transpose the two "
                "columns before binding."
            )

    def _check_geometry(self, frame: TensorFrame, pos: torch.Tensor) -> tuple[int, ...]:
        """Put this batch's geometry through :meth:`_check_bonds_unwrapped`, once.

        The **gate every public entry point goes through** — :meth:`calc_energy`,
        :meth:`calc_energy_forces` and :meth:`calc_energy_forces_virial` all call
        it before pricing anything — and the reason the F29 refusal is not
        :meth:`bind_topology`'s alone. A bind happens only where there is an
        exclusion table to build, i.e. only for a field with a nonbonded term
        (:meth:`_rebuilds_pairs`); a **bonded-only** field under a
        ``graphs.cell`` binds nothing, so hanging the check on the bind left
        exactly the composition whose energy the check exists for — bonds
        measured on plain displacements — as the one that never got it. The
        guard therefore belongs to a seam of its own, called wherever a public
        entry resolves positions, and :meth:`bind_topology` calls it too rather
        than the underlying check.

        **Once per frame, not once per step.** The memo is keyed on the batch
        *object*, held by weak reference, and a repeated call on that same
        object is a pointer comparison and nothing else — no fingerprint is
        rebuilt. Rebuilding one stacks the bond / angle / proper index columns,
        which is tens of microseconds per step (25 µs measured on the butane
        topology, against 0.09 µs for the comparison) sitting directly in front
        of a compiled core — the whole saving spent on deciding not to work.
        The fingerprint of the frame that passed is cached in the record
        instead, and handed back to :meth:`bind_topology` so the bind does not
        compute a second one either.

        That the key is the object is the split the physics asks for. A
        **frame** is a geometry, and the check is about geometry, so every new
        frame is checked: a training container prices batch after batch of the
        same shape, and one whole molecule at the front of that stream says
        nothing about the folded one behind it. Within one bound frame, on the
        other hand, positions are deliberately free — that is the MD rollout,
        which cannot pay a host sync per step and does not need to, since an
        integrator never wraps in place and a barostat scales the cell and the
        coordinates together, so a molecule that starts whole stays whole.
        Keying on identity rather than on shape is what makes those two cases
        separable at all; they are indistinguishable by fingerprint. The weak
        reference is what makes the key safe — a freed batch cannot have its
        address reused into a false match, because the dead reference misses
        instead. The record is written only after the check passes, so a refused
        frame is refused again on the next call rather than accepted by its own
        failure, and :meth:`bind_topology` clears it first so an explicit rebind
        — the deliberate "same molecule, new box", and equally a topology
        rewritten into the same object — is always an explicit re-check.

        **A miss runs the check in every mode — there is no "we are tracing"
        escape hatch, because one here failed open.** Refusing a frame means
        reading a comparison back to Python and branching on it, so this method
        cannot be *traced* into a graph. The tempting conclusion — skip it under
        ``torch.compiler.is_compiling()`` — is wrong twice over. It is wrong in
        principle, because "the check cannot be compiled" is not "the frame need
        not be checked", and a folded molecule priced through a compiled entry
        is the same finite, differentiable, wrong energy as through an eager
        one, with nothing downstream to catch it. And it is wrong in fact:
        measured, ``is_compiling()`` reads ``False`` by the time this method
        runs under ``torch.compile``, ``torch._dynamo.explain`` and
        ``fullgraph=True`` alike — dynamo has already given up on the frame and
        is re-running it eagerly — so such a guard is not a trade-off, it is
        dead code that silences the check on the *eager* re-run.

        What actually happens on a miss under a compiler is therefore the right
        thing, and it is measured rather than assumed: dynamo bails out of the
        frame, the check runs for real, and a wrapped molecule raises its
        ``ValueError`` out through the compiled call. Under ``fullgraph=True``,
        where bailing is not allowed, dynamo refuses to compile the
        data-dependent comparison instead. Both are refusals; neither is a
        number.

        The cost of a miss under a compiler is graph breaks, and that is the
        honest signal: it means a whole public entry point was wrapped in a
        compiler on a frame nothing had vouched for. The composition that
        avoids it is the one :meth:`bind_topology` prescribes for a compiled
        path — bind, or evaluate once eagerly, *then* compile — after which
        every later call on that batch is the pointer comparison above and the
        trace is clean. Nothing molnex compiles itself pays it: the compiled
        region a force field runs is :meth:`_classical_core`, and
        :meth:`calc_energy_forces` calls this **before** entering it, eagerly,
        every step; :meth:`calc_energy_forces_virial` has no compiled build at
        all (finding F34, failed closed at :meth:`molix.md.MD.set_compiler`).

        Args:
            frame: The typed view of the batch being priced, as
                :meth:`_check_bonds_unwrapped` reads it (``graphs.cell``, the
                ``bonds`` namespace, ``atoms.batch``). Its identity **is** the
                memo key — which is why :meth:`_frame` hands back the same view
                for the same batch object, rather than a fresh wrapper per
                call. A system that cannot be weakly referenced (a plain
                ``dict``) is simply never memoized and is checked on every call.
            pos: Positions ``(N, 3)`` Å the energy would be priced at — the
                caller's ``pos`` override when there is one, since that is the
                geometry that gets measured. It is deliberately **not** part of
                the key: positions are free within one bound frame (see above),
                so a second call on the same batch with different ``pos`` is
                passed through. Every in-tree caller hands the same geometry the
                batch carries, as a leaf or a cast of it, so this costs nothing
                today; a caller that scores genuinely different coordinates on
                one batch object wants a bind between them.

        Returns:
            The :meth:`_topology_fingerprint` of the frame now vouched for —
            cached on a hit, computed on a miss.

        Raises:
            ValueError: Propagated from :meth:`_check_bonds_unwrapped` — some
                bond is longer than half the minimum perpendicular width of its
                graph's cell (F29). Raised from a compiled call too, not only an
                eager one (see above).
        """
        checked = self._checked_geometry
        if checked is not None and checked[0]() is frame:
            return checked[1]
        # No "are we tracing?" escape hatch here, deliberately — see the
        # docstring. A miss runs the check in every mode, because the one thing
        # this method may never do is return without an answer.
        fingerprint = self._topology_fingerprint(frame, pos.shape[0])
        self._check_bonds_unwrapped(frame, pos)
        try:
            self._checked_geometry = (weakref.ref(frame), fingerprint)
        except TypeError:
            # A mapping that cannot be weakly referenced (a plain dict) is
            # checked on every call rather than memoized against an address
            # that may be reused by the next frame.
            self._checked_geometry = None
        return fingerprint

    def _check_bonds_unwrapped(self, frame: TensorFrame, pos: torch.Tensor) -> None:
        """Refuse a periodic frame whose molecules are folded across a box face.

        This is the **one** place the precondition every public entry point
        states (:meth:`calc_energy`, :meth:`calc_energy_forces`,
        :meth:`calc_energy_forces_virial`) is spelled out and enforced.

        The two halves of an assembled energy read geometry differently: the
        bonded terms take plain ``pos_j - pos_i`` off their own index columns,
        while the pair terms take *minimum-image* separations
        (:meth:`_placed_pairs`; :meth:`_rebuilds_pairs` says which terms belong
        to which group). Wrapping a periodic frame into the box is
        therefore invisible to one half and fatal to the other — an atom folded
        across a face keeps its pair energy exact, the minimum image being
        wrap-invariant, and has its bonds, angles and torsions measured right
        across the box. Nothing downstream can spot it: a stretched bond is a
        legal configuration, and the number that comes back is finite,
        differentiable and wrong. It is finding **F29** in
        ``.claude/notes/composable-potentials.md``, and a frame read back from a
        wrapped trajectory is its normal source.

        Bonds alone decide it. A molecule whose every covalent bond is shorter
        than half its cell's minimum perpendicular width is contiguous in the
        coordinates as given, so every displacement built on top of the bond
        graph — angle, proper, improper — is the intramolecular one too; a
        molecule folded across a face has to leave at least one bond nearly a
        box long.

        **The limit, in symbols.** Write the cell's three lattice vectors as
        ``a_1``, ``a_2``, ``a_3`` (rows of ``graphs.cell``, Å) and its volume as
        ``V = |det(cell)|`` (Å³). The perpendicular width between the pair of
        faces spanned by ``a_i`` and ``a_j`` is the volume divided by that
        face's area, ``V / |a_i × a_j|``; the **smallest** of the three widths
        therefore uses the **largest** face area, so

            ``w_min = V / max over the three face pairs of |a_i × a_j|``

        (``w_min = L`` for a cube of edge ``L``). A bond is refused when its
        length exceeds ``w_min / 2`` — ``L/2`` for that cube — which is the code
        below: ``half_width = 0.5 * volume / widest_face``. It is *not* half the
        shortest lattice vector: a flattened triclinic cell can have long edges
        and a thin slab between two of its faces, and it is the slab that limits
        how far a molecule can reach before its image is nearer than itself.

        Reached only through :meth:`_check_geometry`, which is host-side and
        admits one frame per topology, so the single ``argmax`` sync this costs
        is paid once and never once per step — the force path stays free of it,
        which is the whole reason the check lives out here rather than in a
        term. A batch with no ``graphs.cell`` (an open boundary, nothing to
        wrap into), no bonds, or a degenerate lattice has no limit to exceed and
        is passed through untouched.

        Args:
            frame: The batch's typed view, optionally carrying
                ``graphs.cell`` ``(3, 3)`` or ``(B, 3, 3)`` in Å and the
                ``bonds`` namespace. ``atoms.batch`` picks each bond's cell out
                of a batched one.
            pos: Positions ``(N, 3)`` in Å the energy would be priced at — the
                ``pos`` override when :meth:`bind_topology` was handed one,
                since that is the geometry, not ``atoms.pos``. Read detached.

        Raises:
            ValueError: Some bond is longer than half the minimum perpendicular
                width of its graph's cell, naming that bond, its length, the
                limit and the remedy.
        """
        cell = frame.graphs.get("cell", None) if "graphs" in frame.keys() else None
        bond_index = self._bond_index(frame)
        if cell is None or bond_index is None or bond_index.shape[-1] == 0:
            return
        cells = cell.detach().to(pos)
        if cells.ndim == 2:
            cells = cells.unsqueeze(0)
        a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
        # Perpendicular width between two faces: the volume divided by the area
        # of the face pair they span. The *widest* face gives the *thinnest*
        # slab, which is the one a molecule folds across first.
        widest_face = torch.stack(
            [
                torch.linalg.cross(b, c).norm(dim=-1),
                torch.linalg.cross(c, a).norm(dim=-1),
                torch.linalg.cross(a, b).norm(dim=-1),
            ],
            dim=-1,
        ).amax(dim=-1)
        volume = torch.linalg.det(cells).abs()
        half_width = torch.where(
            volume > 0.0,
            0.5 * volume / widest_face.clamp_min(torch.finfo(cells.dtype).tiny),
            torch.inf,
        )
        source, target = bond_index[0].long(), bond_index[1].long()
        p = pos.detach()
        length = (p[target] - p[source]).norm(dim=-1)
        graph = (
            self._atom_batch(frame, p.shape[0], p.device)[source]
            if half_width.shape[0] > 1
            else torch.zeros_like(source)
        )
        limit = half_width[graph]
        # One host sync on the passing path, the same shape as the charge check
        # above: pick the worst offender, then ask about that one alone.
        worst = int(torch.argmax(length - limit))
        if float(length[worst] - limit[worst]) > 0.0:
            raise ValueError(
                f"bond {worst} (atoms {int(source[worst])}-{int(target[worst])}) spans "
                f"{float(length[worst]):.6g} Å, over the {float(limit[worst]):.6g} Å half-width "
                "of its cell: this molecule is wrapped across a box face. "
                "molpot prices bonded terms on plain displacements (F29); unwrap molecules "
                "before evaluation — MD never wraps in place, so a rollout only needs its "
                "initial frame unwrapped."
            )

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Hand the live simulation cell to every term that keeps cell state.

        Two of them can: the periodic Coulomb term, which freezes its
        reciprocal-space shape here, and the mlip — the latter guarded by
        :class:`molix.schema.CellBound`, because ``def_pairstyle`` accepts
        *any* ``nn.Module`` as the mlip and an unconditional call would break
        every model without the method. The bonded terms and the
        Lennard-Jones one hold nothing sized by the lattice, and the neighbour
        list owns its own rebinding, so a ForceField of only those makes this
        a **documented no-op** — as does an mlip that does not satisfy
        ``CellBound``. Neither case raises.

        The ``(1, 3, 3)`` batched layout is resolved here, once, for the
        classical term — the protocol puts that on the caller — while the mlip
        keeps receiving the caller's own storage untouched.

        Args:
            cell: Lattice vectors ``(3, 3)`` — or ``(1, 3, 3)`` — in Å, rows
                being the lattice vectors. Passed to the mlip untouched, storage
                included, so an implementer that re-reads it sees the barostat's
                in-place updates.

        Raises:
            RuntimeError: Propagated from
                :meth:`~molpot.forcefield.pair_terms.CoulombPeriodicPotential.bind_cell`
                when the cell has outgrown the reciprocal shape a previous bind
                froze (raise ``capacity_factor``), or from a ``CellBound``
                mlip's own capacity guard.
        """
        if self.coulomb_periodic is not None:
            self.coulomb_periodic.bind_cell(cell[0] if cell.dim() == 3 else cell)
        if isinstance(self.mlip, CellBound):
            self.mlip.bind_cell(cell)

    def compile_core(self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        """Install a compiled build of this field's hot core — mlip or classical.

        **Which core.** An mlip field has exactly one: the mlip's own, and this
        method is one layer of delegation to it, guarded by
        :class:`molix.schema.CoreCompilable`. That build is installed on the
        **mlip**, never here — ``to_potentials()`` returns a fresh
        ``Potentials`` on every ``MD`` setter call, while the mlip is the one
        ``nn.Module`` the ForceField holds throughout, so a core installed on
        the container would be silently discarded by the next ``set_*``.
        (:meth:`molix.md.MD.set_compiler` closes that gap from the other side:
        it re-applies this method after every rebuild.)

        A classical field's core is :meth:`_classical_core` — the whole
        bonded + LJ + Coulomb ``(system, pos) -> (energy, forces)`` evaluation
        that :meth:`calc_energy_forces` is. The compiled build is stored in
        ``_compiled_classical`` and :meth:`calc_energy_forces` routes through
        it when it is there. It is the *same* function either way, so there is
        still no second evaluation path and no "is it compiled?" physics
        branch: eager and compiled differ only in reduction order.

        **The seam.** The compiled region is the whole classical core except
        one known break: a declared ``pair:coulomb`` (open or periodic) has no
        closed-form force, so :meth:`_coulomb_energy_forces` creates a position
        leaf and calls ``autograd.grad`` through
        ``molpot/derivation/force.py`` — dynamo breaks there, deliberately, and
        that is the *only* break a classical field may show. Owner: molpot
        derivation. A ``lj/cut`` + bonded field has no such term and traces as
        one graph with zero breaks.

        **What this does not cover: the virial (F34).** The classical core is
        the energy/force one. :meth:`calc_energy_forces_virial` — the entry an
        NPT step calls — evaluates a *strained* energy through
        :func:`~molpot.derivation.force.autograd_forces_and_virial`, whose two
        leaves and ``autograd.grad`` are the seam dynamo cannot trace, wrapped
        around a closure that rewrites five columns of the batch per call.
        Compiling it is a restructuring of that seam, not an installation, and
        until it happens **a barostat run is eager whatever is installed
        here**. That gap is not left to be discovered by a benchmark that looks
        fast on paper: :meth:`molix.md.MD.set_compiler` refuses a compiler while
        a barostat is attached, and :meth:`~molix.md.MD.set_barostat` refuses a
        barostat while one is installed, so no run can report itself compiled
        and step eager.

        ``compiler`` is structural — any ``callable -> callable``, such as
        :class:`molix.compile.Compiler`, which molpot must not import.

        Args:
            compiler: Applied to the mlip's nominated core, or to this
                container's :meth:`_classical_core`.

        Raises:
            RuntimeError: There is no compilable core to install — an mlip that
                does not satisfy ``CoreCompilable`` owns the whole energy
                (:meth:`calc_energy_forces` never reaches the classical terms
                for it), so there is nothing left to compile. Failing loudly
                beats compiling nothing and reporting success. An mlip that
                *does* satisfy it may raise this from its own ``compile_core``
                instead — molnex's MACE encoders do, when no eager forward has
                validated their element table and sized their k-space grids
                yet, neither of which a traced call can repair.

        References:
            ``.claude/notes/composable-potentials.md`` — finding **F28** (an
            ``MD`` rollout with no compile seam), closed by this method
            together with :meth:`molix.md.MD.set_compiler`; finding **F34**
            (the strained virial core this method does not compile), open and
            failed closed at both setters.
        """
        if self.mlip is not None:
            if not isinstance(self.mlip, CoreCompilable):
                raise RuntimeError(
                    "compile_core: this mlip has no compilable core "
                    f"({type(self.mlip).__name__} does not implement compile_core); "
                    "only an mlip satisfying molix.schema.CoreCompilable can be compiled"
                )
            self.mlip.compile_core(compiler)
            return
        self._compiled_classical = compiler(self._classical_core)

    def get_schema(self) -> ForceFieldSchemaView:
        """Metadata of the declared styles: parameter names, units, energy formulas."""
        return PotentialMeta.schema_for(self.forcefield.style_names(), units=self.forcefield.units)

    def calc_energy(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total energy: the sum of every term the ForceField declares.

        Overrides :meth:`~molpot.potentials.base.BasePotential.calc_energy`, which
        returns a Python ``float``: this returns a **tensor** still attached to the
        autograd graph, because :meth:`calc_forces` and the Hessian path
        differentiate it.

        **The mlip branch.** When :attr:`mlip` is not ``None`` (``def_pairstyle``
        was given an ``nn.Module`` instead of a classical style name) and the
        system carries an ``atoms`` namespace, this method never sums the
        classical term energies below — it hands the whole batch to
        :func:`molpot.derivation.protocol.call_energy`, which is the single
        place that decides whether the mlip's ``_write_energy`` protocol hook or
        a plain ``model(batch)`` forward runs. Whether the mlip has an installed
        compiled core (see :meth:`compile_core`) is invisible here too — the
        mlip decides, not this branch. A ``pos`` override is written onto
        ``atoms.pos`` *before* that call, since the mlip only ever reads
        positions off the batch — writing it after would be ignored. What the
        batch is *carried in* decides nothing here: a ``TensorDict``, its
        :class:`~molix.batch.TensorFrame` view and a nested mapping of the same
        namespaces are one system (:meth:`_frame`). With an mlip but a system
        carrying no ``atoms`` at all — the ``pos=``-only call — there is nothing
        to hand it, so this falls through to the classical path below instead
        (which, with no classical styles declared, contributes an empty sum).

        **Precondition — each molecule must arrive whole (F29).** A periodic
        frame's molecules must not be wrapped across a box face: the bonded
        terms price plain displacements while the pair terms price minimum-image
        ones, so a folded molecule returns a finite, differentiable, wrong
        energy. Enforced here, on the classical path, by
        :meth:`_check_geometry` — once per frame, whatever the field declares,
        so a bonded-only periodic field is covered too; the mechanism and its
        limits are on :meth:`_check_bonds_unwrapped`.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``), each molecule whole (see the precondition
                above). With any pair style declared it must also carry
                the ``edges`` neighbour list a molix owner placed
                (:meth:`_placed_pairs`). Must be a ``TensorDict`` to take the
                mlip branch above.
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

        Raises:
            ValueError: Raised through :meth:`term_energies` on the classical
                path — a declared style's parameter table is absent; a pair
                style is declared and the batch carries no usable ``edges``
                list; or ``pair:coulomb`` is evaluated under a ``graphs.cell``
                on a batch of more than one graph (one lattice gives one
                reciprocal sum, so each graph would otherwise be charged the
                whole box). See :meth:`term_energies` for each. Plus, on that
                path, the F29 refusal :meth:`_check_geometry` raises on a
                molecule wrapped across a box face — the one bind-time check
                that cannot wait for a bind, because a bonded-only field never
                performs one.
        """
        frame = self._frame(system)
        if self.mlip is not None and "atoms" in frame.keys():
            # The pos override lands on the batch first: the mlip reads its
            # positions off ``atoms.pos``, so writing after the call would be
            # ignored. The write goes through the view, which shares the
            # caller's storage, so it lands on the caller's own batch. Which
            # entry of the mlip evaluates the energy is ``call_energy``'s policy
            # alone — this branch must not reimplement it by probing the mlip.
            # Whether a compiled core is installed is likewise the mlip's own
            # business and invisible here.
            if pos is not None:
                frame["atoms", "pos"] = pos
            out = call_energy(self.mlip, frame)
            return out[ENERGY_KEY]
        p = self._pos(frame, pos)
        # The F29 gate, on the entry rather than on the bind: a bonded-only
        # field binds nothing, so a bind-only check would miss precisely the
        # composition whose bonded terms it protects (:meth:`_check_geometry`,
        # once per frame).
        self._check_geometry(frame, p)
        num_graphs = self._num_graphs(frame)
        total = torch.zeros((num_graphs,) if num_graphs > 1 else (), device=p.device, dtype=p.dtype)
        for e in self.term_energies(frame, pos=p).values():
            total = total + e
        return total

    def calc_forces(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Atomic forces ``F = -∂E/∂pos``.

        Delegates to the canonical autograd backend
        :func:`~molpot.derivation.force.autograd_forces_from_energy` — no
        second force path lives here. Per-graph energies ``(B,)`` are summed to
        one scalar before differentiating, so a multi-molecule batch works.

        Positions are re-attached as a fresh ``requires_grad`` leaf through
        :func:`~molpot.derivation.force.position_leaf`, which owns that seam for
        the whole package: any autograd history the caller's ``pos`` carried is
        not differentiated through, while the force stays connected to the model
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
            system: Batch / mapping carrying ``atoms.pos`` and the topology,
                each molecule whole — this differentiates :meth:`calc_energy`
                and inherits its F29 precondition verbatim.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Forces ``(N, 3)`` in the ForceField's unit convention
            (``units="real"`` default: kcal mol⁻¹ Å⁻¹).
        """
        frame = self._frame(system)
        p = position_leaf(self._pos(frame, pos))
        with torch.enable_grad():
            energy = self.calc_energy(frame, pos=p)
        return autograd_forces_from_energy(energy, p)

    def calc_energy_forces_virial(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
        *,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy, forces and virial from one strained autograd backward.

        MD NPT calls this once per step. ``create_graph`` follows the ambient
        grad mode — wrap the run in ``torch.no_grad()`` so the tape is not
        kept for parameter derivatives.

        **Precondition — each molecule must arrive whole (F29).** A periodic
        frame's molecules must not be wrapped across a box face: the bonded
        terms price plain displacements while the pair terms price minimum-image
        ones, so a folded molecule returns a finite, differentiable, wrong
        energy — and a wrong virial with it. Enforced here by
        :meth:`_check_geometry`, on the unstrained frame and before the strain
        closure runs, once per frame and whatever the field declares; the
        mechanism and its limits are on :meth:`_check_bonds_unwrapped`.

        Args:
            system: Batch carrying ``atoms.pos`` and ``graphs.cell``, each
                molecule whole (see the precondition above). Read — and written
                back to, for the duration of the strain closure — through its
                typed view (:meth:`_frame`), which shares the caller's storage.
            pos: Optional positions overriding ``atoms.pos``.

        Returns:
            ``(energy, forces, virial)`` with virial ``W = ∂E/∂ε`` shape
            ``(3, 3)`` in the ForceField energy unit.

        Raises:
            TypeError: ``system`` is not a batch, a view of one or a nested
                mapping of its namespaces (:func:`as_tensor_frame`).
            ValueError: ``graphs.cell`` is missing; or a classical pair term is
                declared and the batch carries a placed edge list without
                ``edges.shifts``, whose lattice remainder those terms recover
                as a constant that no strain reaches — the virial would be
                missing every pair that meets its partner through a lattice
                vector (F13). :meth:`calc_energy` and :meth:`calc_forces` stay
                exact on such a batch and keep working. Or, from
                :meth:`_check_geometry`, a molecule wrapped across a box face
                (F29) — a cell is a precondition of this entry, so the check
                always has a limit to measure against here.
        """
        frame = self._frame(system)
        nested = frame.keys(include_nested=True)
        if ("graphs", "cell") not in nested:
            raise ValueError("calc_energy_forces_virial requires graphs.cell")
        # Annotated because the view's ``__getitem__`` is typed as
        # "leaf or sub-namespace": what these keys hold is a tensor, and the
        # reads below (``.dim()``, ``.norm(dim=-1)``, indexing) are a tensor's.
        cell_buf: torch.Tensor = frame["graphs", "cell"]
        batched_cell = cell_buf.dim() == 3
        cell_3 = cell_buf[0] if batched_cell else cell_buf
        pos_buf = self._pos(frame, pos)
        # F29, on the frame as handed in: the strain closure below replaces
        # ``atoms.pos`` with a deformed copy, so the geometry a caller is
        # answerable for is this one, and it must be judged before it is
        # deformed (:meth:`_check_geometry`, once per frame).
        self._check_geometry(frame, pos_buf)
        edge_index: torch.Tensor | None = (
            frame["edges", "edge_index"] if ("edges", "edge_index") in nested else None
        )
        edge_diff_buf: torch.Tensor | None = (
            frame["edges", "edge_diff"] if ("edges", "edge_diff") in nested else None
        )
        # Strain the Cartesian periodic remainder, not a copy of edge_diff.
        # Passing frozen edge_diff as an independent cartesian makes local
        # graph energy (MACE / Polar) a function of strain only, so
        # ∂E/∂pos drops the short-range forces and NPT heats.
        stored_shifts = ("edges", "shifts") in nested
        if not stored_shifts and edge_index is not None and self._rebuilds_pairs():
            # A classical nonbonded term rebuilds its own pair vectors through
            # ``Potentials._placed_pairs``, which recovers a shift-less
            # list's lattice remainder as a **detached** constant — right in
            # value (energy and forces below match a stored-``shifts`` batch to
            # the bit) but not a function of the cell, so the strain applied
            # here never reaches a pair that meets its partner through a
            # lattice vector and its share of ``W = ∂E/∂ε`` is silently absent
            # (finding F13 in ``.claude/notes/composable-potentials.md``).
            # An mlip is not affected: it reads ``edges.edge_diff``, which
            # ``energy_fn`` rebuilds from the *strained* remainder below.
            raise ValueError(
                "calc_energy_forces_virial needs edges.shifts on a placed list under a cell "
                "when a classical pair term is declared: this list carries none, and the "
                "edge_diff fallback its pair vectors are rebuilt from is a detached constant "
                "that carries no lattice strain, so every pair reaching its partner through "
                "a lattice vector would be missing from the virial. Build the list with "
                "molix.md.NeighborList (which writes edges.shifts), or ask for energy and "
                "forces only — those are exact on this batch."
            )
        shifts_buf: torch.Tensor | None = None
        if stored_shifts:
            shifts_buf = frame["edges", "shifts"]
        elif edge_diff_buf is not None and edge_index is not None:
            src0 = edge_index[:, 0]
            tgt0 = edge_index[:, 1]
            shifts_buf = edge_diff_buf - (pos_buf[tgt0] - pos_buf[src0])
        cartesians: tuple[torch.Tensor, ...] = (shifts_buf,) if shifts_buf is not None else ()

        def energy_fn(
            pos_s: torch.Tensor, cell_s: torch.Tensor, *strained: torch.Tensor
        ) -> torch.Tensor:
            frame["atoms", "pos"] = pos_s
            frame["graphs", "cell"] = cell_s.unsqueeze(0) if batched_cell else cell_s
            if strained and stored_shifts:
                # The remainder a term rebuilds its own pair vectors from
                # (``Potentials._placed_pairs``) must be the *strained*
                # one, or every wrapped pair holds its lattice vector fixed
                # while the box deforms and its share of ``W = dE/deps`` is
                # missing.
                frame["edges", "shifts"] = strained[0]
            if edge_index is not None:
                src = edge_index[:, 0]
                tgt = edge_index[:, 1]
                diff = pos_s[tgt] - pos_s[src]
                if strained:
                    diff = diff + strained[0]
                frame["edges", "edge_diff"] = diff
                frame["edges", "edge_dist"] = diff.norm(dim=-1)
            energy = self.calc_energy(frame, pos=pos_s)
            return energy if energy.ndim == 0 else energy.sum()

        energy, forces, virial = autograd_forces_and_virial(energy_fn, pos_buf, cell_3, *cartesians)
        # Strain writes replace graphs.cell / pos / edge_diff with graph
        # tensors. Put the MD buffers back so the barostat copy_ and the
        # next neighbour update still see the same storage.
        frame["atoms", "pos"] = pos_buf
        frame["graphs", "cell"] = cell_buf
        if stored_shifts and shifts_buf is not None:
            frame["edges", "shifts"] = shifts_buf
        if edge_diff_buf is not None:
            frame["edges", "edge_diff"] = edge_diff_buf
            frame["edges", "edge_dist"] = edge_diff_buf.norm(dim=-1)
        graphs_bs = frame.graphs.batch_size
        stored = energy.detach().reshape(*graphs_bs) if graphs_bs else energy.detach().reshape(())
        write_energy(frame, stored)
        write_forces(frame, forces)
        write_virial(frame, virial)
        return energy, forces, virial

    def calc_virial(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
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
        system: TensorDict | TensorFrame | Mapping[str, Any],
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

        **What this method does around the evaluation itself.** Three things, in
        this order, and all of them are why the evaluation proper lives in
        :meth:`_classical_core` rather than here. First, when the field has a
        nonbonded term (:meth:`_rebuilds_pairs`) and no topology is bound yet, it
        calls :meth:`bind_topology` — the once-per-topology, host-side build of
        the exclusion table plus the bind-time checks (net charge, edge
        orientation), so every ``ValueError`` those raise can surface from
        *this* call and not only from an explicit bind. Doing it here rather
        than lazily inside the core is what lets the core be one graph from its
        first call: the same work traced inside a compiled region is six
        measured graph breaks. A bonded-only field binds nothing — it has no
        pair list to price and no exclusion table to consult.
        Second, and precisely because of that, the F29 geometry gate is *not*
        left to the bind: :meth:`_check_geometry` runs on every classical call,
        so the bonded-only field the bind skips is checked all the same. It
        carries its own once-per-frame record, so an MD rollout — one bound
        batch for the whole run — pays it once and an identity test per step
        thereafter, while a training stream is checked batch by batch.
        Third, it dispatches to the compiled build of the core when
        :meth:`compile_core` has installed one, and to the eager bound method
        otherwise. It is the same function either way — compiled and eager
        differ only in reduction order, never in physics — so nothing
        downstream branches on "is it compiled?". All three run *outside* the
        compiled region, which is what keeps their host syncs out of the trace.

        **Precondition — each molecule must arrive whole (F29).** A periodic
        frame's molecules must not be wrapped across a box face: the bonded
        terms price plain displacements while the pair terms price minimum-image
        ones, so a folded molecule returns a finite, differentiable, wrong
        energy. Enforced by the geometry gate above, which refuses such a frame
        whatever the field declares; the mechanism and its limits are on
        :meth:`_check_bonds_unwrapped`.

        Args:
            system: Batch / mapping carrying ``atoms.pos`` and the topology,
                each molecule whole (see the precondition above).
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Energy ``(B,)`` — ``()`` for a single molecule — and forces ``(N, 3)``,
            in the ForceField's unit convention (``units="real"`` default:
            kcal/mol and kcal mol⁻¹ Å⁻¹). mlip path: the mlip's own energy
            unit (eV for the MACE family), never a sum with classical terms.

        Raises:
            ValueError: A declared style's parameter table is absent — the same
                contract as :meth:`term_energies`, checked branch for branch so
                the two routes agree on which ForceFields evaluate at all — or
                a pair style is declared and the batch carries no usable
                ``edges`` list, or ``pair:coulomb`` meets a ``graphs.cell`` on
                a batch of more than one graph. All three are the
                :meth:`term_energies` contract. Plus, from the bind this method
                performs on a not-yet-bound classical field, the two
                :meth:`bind_topology` refusals: a net charge that misses the
                declared ``graphs.total_charge`` and an edge list carrying only
                ``source > target`` rows. And, from :meth:`_check_geometry` —
                which runs whether or not the field binds — a molecule wrapped
                across a box face.
        """
        frame = self._frame(system)
        if self.mlip is not None and "atoms" in frame.keys():
            # Same branch policy as ``calc_energy``: the mlip owns the whole
            # energy. Without this, an mlip ForceField — which declares no
            # classical style — falls through the term loop below and returns a
            # silent ``(0, zeros)`` instead of its energy and forces.
            leaf = position_leaf(self._pos(frame, pos))
            with torch.enable_grad():
                energy = self.calc_energy(frame, pos=leaf)
            return energy, autograd_forces_from_energy(energy, leaf)

        p = self._pos(frame, pos)
        # Bind the topology *before* the core, not lazily from inside the first
        # ``_scaled_pairs``. Identical work either way — this is the same
        # once-per-topology bind, just hoisted — but it is host-side work
        # (an ``.any()`` orientation check, an exclusion table of
        # data-dependent shape) and inside a compiled core it is six graph
        # breaks on the first trace, measured. Out here the core stays one
        # graph from its very first call, so a caller does not have to know to
        # evaluate once eagerly before ``compile_core``.
        if self._exclusions is None and self._rebuilds_pairs():
            self.bind_topology(frame, pos=p)
        # The F29 geometry gate, unconditionally — the bind above is skipped for
        # a bonded-only field, and that is the field whose bonded terms the
        # check protects. Idempotent per frame (:meth:`_check_geometry`): the
        # bind's own call and this one cost one pointer comparison between them,
        # and an MD rollout — one batch object for the whole run — pays the
        # check on its first step and that same pointer comparison on every
        # later one. It stays outside the core, so the host sync it costs when
        # it does run never reaches a trace.
        self._check_geometry(frame, p)
        # One function, two builds: the compiled one when ``compile_core`` has
        # installed it, otherwise the eager bound method. Never a second
        # evaluation path — see :meth:`compile_core`. The core is handed the
        # **view**, wrapped once out here where the wrap is eager: it reads the
        # same leaves the plain batch holds, so the trace is the one it always
        # was (:meth:`_frame`).
        core = self._compiled_classical
        if core is None:
            core = self._classical_core
        return core(frame, p)

    def _classical_core(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The classical ``(system, pos) -> (energy, forces)`` evaluation itself.

        The body of :meth:`calc_energy_forces` for a field with no mlip, split
        out as a flat bound method so it can be handed to a compiler:
        :meth:`compile_core` stores ``compiler(self._classical_core)`` and
        :meth:`calc_energy_forces` then calls that instead. Every contract
        stated there — the analytic bonded/LJ branch, the autograd Coulomb
        fallback, and the whole-molecule precondition (F29) — is this method's;
        it is documented once, on the public entry.

        The one break inside the compiled region is the Coulomb autograd seam
        in ``molpot/derivation/force.py`` (:meth:`compile_core`); a ``lj/cut``
        + bonded field traces as one graph with none. That holds for a core
        entered through :meth:`calc_energy_forces`, which binds the topology
        first. Tracing this method directly on a container that has never been
        evaluated traces the lazy bind with it — an ``.any()`` orientation
        check and an exclusion table of data-dependent shape, six breaks — so
        measure the seam the way the run reaches it.

        Args:
            frame: The batch's typed view, carrying the topology and the
                ``edges`` list — the container :meth:`calc_energy_forces`
                wrapped once, eagerly, before entering here. The view is the
                only spelling this core accepts, so the compiled region holds
                no wrap and no branch on the container's type; a caller
                measuring *this* function — the compile seam's own gate does —
                hands ``as_tensor_frame(batch)``, the same container production
                hands it, so the graph the gate certifies is the graph the run
                compiles.
            pos: Positions ``(N, 3)`` Å already resolved by :meth:`_pos` —
                detached here, so callers may pass the batch's own tensor.

        Returns:
            Energy ``(B,)`` — ``()`` for a single molecule — and forces
            ``(N, 3)`` in the ForceField's unit convention.
        """
        p = pos.detach()
        device = p.device
        dtype = p.dtype
        num_graphs = self._num_graphs(frame)
        batched = num_graphs > 1
        zero = torch.zeros((num_graphs,) if batched else (), device=device, dtype=dtype)
        n_atoms = p.shape[0]
        atom_batch = self._atom_batch(frame, n_atoms, device)

        energy = zero
        forces = p.new_zeros((n_atoms, 3))

        if "bond" in self._bonded_styles:
            self._require_bonded_params("bond", "bonds", self.bond)
            bond_index = self._bond_index(frame)
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

        if "angle" in self._bonded_styles:
            self._require_bonded_params("angle", "angles", self.angle)
            angle_index = self._angle_index(frame)
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

        if "dihedral" in self._bonded_styles:
            self._require_bonded_params("dihedral", "propers", self.proper)
            proper_index = self._proper_index(frame)
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

        if "improper" in self._bonded_styles:
            self._require_bonded_params("improper", "impropers", self.improper)
            improper_index = self._improper_index(frame)
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
            (rep, disp), f = self._lj_energy_forces(frame, p)
            energy = energy + rep + disp
            forces = forces + f

        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            e, f = self._coulomb_energy_forces(frame, p, n_atoms)
            energy = energy + e
            forces = forces + f

        return energy, forces

    def term_energies(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
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
                topology namespaces — plus, for any declared pair style, the
                ``edges`` neighbour list built upstream in molix.
            pos: Optional positions ``(N, 3)`` Å overriding ``atoms.pos``.

        Returns:
            Term name → energy ``(B,)`` (``()`` for a single molecule) in the
            ForceField's unit convention (``units="real"`` default: kcal/mol).

        Raises:
            ValueError: A declared style's parameter table is absent; a pair
                style is declared and the batch carries no usable placed
                neighbour list — no ``edges.edge_index``, or one with neither
                ``edges.shifts`` nor ``edges.edge_diff`` to price it by
                (:meth:`_placed_pairs`); or ``pair:coulomb`` is evaluated under
                a ``graphs.cell`` on a batch of more than one graph — one
                lattice gives one reciprocal sum, so each graph would be
                charged the whole box.
        """
        frame = self._frame(system)
        p = self._pos(frame, pos)
        device = p.device
        dtype = p.dtype
        num_graphs = self._num_graphs(frame)
        batched = num_graphs > 1
        zero = torch.zeros((num_graphs,) if batched else (), device=device, dtype=dtype)
        atom_batch = self._atom_batch(frame, p.shape[0], device)
        terms: dict[str, torch.Tensor] = {}

        if "bond" in self._bonded_styles:
            self._require_bonded_params("bond", "bonds", self.bond)
            bond_index = self._bond_index(frame)
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

        if "angle" in self._bonded_styles:
            self._require_bonded_params("angle", "angles", self.angle)
            angle_index = self._angle_index(frame)
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

        if "dihedral" in self._bonded_styles:
            self._require_bonded_params("dihedral", "propers", self.proper)
            proper_index = self._proper_index(frame)
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

        if "improper" in self._bonded_styles:
            self._require_bonded_params("improper", "impropers", self.improper)
            improper_index = self._improper_index(frame)
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
            terms["lj_repulsion"], terms["lj_dispersion"] = self._lj_energy(frame, p)
        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            terms["coulomb"] = self._coulomb_energy(frame, p)
        return terms

    def contribution_energies(
        self,
        system: TensorDict | TensorFrame | Mapping[str, Any],
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

    def _cell(self, frame: TensorFrame) -> torch.Tensor | None:
        if "graphs" not in frame.keys():
            return None
        cell = frame.graphs.get("cell", None)
        if cell is None:
            return None
        return cell[0] if cell.ndim == 3 else cell

    def _placed_pairs(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        *,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The batch's placed ``edges`` list, priced at the **live** positions.

        ``molpot`` consumes the cutoff graph; it never enumerates one. Both
        owners live upstream in ``molix`` (CLAUDE.md dependency graph;
        ``.claude/notes/architecture.md`` §Layer roles):

        * :class:`molix.batch.NeighborList` — the training path and the Edge
          Convention owner. Writes ``edge_index`` ``(E, 2)``, ``edge_diff``
          ``(E, 3)`` = ``pos[target] - pos[source]`` and ``edge_dist`` ``(E,)``,
          bidirectional at its ``symmetry=True`` default.
        * :class:`molix.md.NeighborList` — the MD path. Writes those three plus
          ``edges.shifts`` (each row's lattice remainder) at a fixed capacity
          whose dead rows are in-range self-loops.

        No in-tree producer writes ``edges.num_edges``. The MD list keeps its
        live count as a Python attribute and marks its dead tail with
        ``source == target`` rows, which the ``source < target`` half-list rule
        below already kills (finding F22 in
        ``.claude/notes/composable-potentials.md``; publishing the count from
        molix is a molix follow-up). The ``row < num_edges`` liveness leg is
        therefore **optional** — honoured when some producer does place the
        key, never required for a list to be priced.

        The list is returned at the capacity it arrived at and ``live`` is the
        mask of the rows that contribute, never a selection to apply:
        compacting it makes the output shape a function of the data, which is a
        graph break on the compiled force path, so the pair terms multiply by
        the mask instead. Liveness is ``source < target`` (one orientation of
        each bidirectional pair) ∧ — when the asking term declares a cutoff —
        ``0 < r < r_c`` ∧ — when the list declares one — ``row < num_edges``,
        compared as tensors because reading the live count out to Python is a
        host sync. Both legs travel with their condition: a ``cutoff`` of
        ``None`` drops the ``r`` test whole, ``0 < r`` included, which costs
        nothing because a zero-length row is a padding self-loop and
        ``source < target`` has already killed it.

        Each row's separation **vector** is recomputed from the live ``pos`` by
        the first branch the list supports, and the returned distance is its
        norm. Both come back, from the one tensor: a term that needs a
        direction (the analytic LJ force's ``r̂``) must divide *that* vector by
        *this* distance, not rebuild ``pos[t] - pos[s]`` for itself. Under a
        cell the two disagree for every pair whose nearest image is through a
        lattice vector — the unwrapped difference is longer than the distance
        priced and points the other way — so a term that re-derives its own
        direction reports a force that is not the gradient of the energy
        returned beside it. Owning both here is what keeps them one geometry.

        1. ``shifts`` — ``|pos[t] - pos[s] + shifts|``, clamped *before* the
           ``sqrt`` because a padding self-loop is a zero vector whose ``√``
           gradient is ``NaN``, which the later multiply-by-zero cannot undo.
           The only branch that is a function of both the live positions and
           the live lattice, hence the only one whose ``∂E/∂ε`` is complete
           (finding F13 in ``.claude/notes/composable-potentials.md``).
        2. ``edge_diff`` — the lattice remainder is recoverable only as
           ``edge_diff - (pos[t] - pos[s])`` on **detached** tensors, i.e. a
           constant. Equal to branch 1 to the last bit while ``pos`` is the
           geometry the list was written at (``edge_diff`` is defined as that
           same sum), and energy/forces stay exact under a rigid shift of the
           whole batch — but the constant is **stale** for any other positions,
           and it carries no strain, which is why
           :meth:`calc_energy_forces_virial` refuses a shift-less list.

        A stored ``edge_dist`` is **not** a third branch. It is a number, not
        geometry: forwarded as the answer it does not depend on the ``pos``
        leaf the energy is differentiated against, so ``dE/dpos`` is
        structurally zero and the batch prices a finite energy with silently
        zero forces — the worst failure an MD path can have. It also carries
        whatever imaging its owner applied, which a second convention here can
        only disagree with. A list with neither ``shifts`` nor ``edge_diff`` is
        refused instead (see ``Raises``).

        Args:
            frame: The batch's typed view carrying the ``edges`` namespace —
                the one every caller inside the evaluation already holds, so
                nothing is wrapped or branched on here, one hop inside the
                compiled region. A caller holding a plain batch wraps it with
                :func:`~molix.batch.as_tensor_frame` first.
            pos: Live positions ``(N, 3)`` in Å the distances are priced at —
                the strained / integrated leaf, not necessarily ``atoms.pos``.
            cutoff: Real-space cutoff ``r_c`` in Å of the asking term, or
                ``None`` to keep every row of the list.

        Returns:
            ``(edge_index, pair_vectors, dist, live)`` at the list's capacity —
            ``(E, 2)`` endpoints as placed, ``(E, 3)`` separation vectors in Å
            pointing source → target (``pos[t] - pos[s] + shift`` under a
            cell), ``(E,)`` separations in Å which are exactly
            ``|pair_vectors|``, and a boolean ``(E,)``. The vectors stay on the
            autograd graph with the distances. ``E == 0`` (a list with no rows)
            is an ordinary empty list — no pair in range — and prices at zero;
            only an *absent* list raises.

        Raises:
            TypeError: *system* is not a batch, a view of one or a nested
                mapping of its namespaces (:meth:`_frame`).
            ValueError: The batch carries no ``edges.edge_index``, or an
                ``edge_index`` with neither ``shifts`` nor ``edge_diff`` to
                price it by — a stored ``edge_dist`` alone is refused, not
                forwarded. Enumerating a replacement here would answer with a
                different neighbour set than the model was trained or
                integrated against, so the message names the geometry a list
                must carry and the two molix owners instead.
        """
        edges = frame.edges if "edges" in frame.keys() else None
        edge_index = None if edges is None else edges.get("edge_index", None)
        if edges is None or edge_index is None:
            raise ValueError(
                "this batch carries no edges.edge_index, and molpot does not enumerate "
                "neighbour pairs — the cutoff graph is built upstream by "
                "molix.batch.NeighborList (training batches: edge_index, edge_diff, "
                "edge_dist) or by molix.md.NeighborList (MD: the same at a fixed capacity, "
                "plus edges.shifts). Place a list on the batch before evaluating; a list "
                "with zero rows is accepted and prices at zero."
            )
        shifts = edges.get("shifts", None)
        edge_diff = edges.get("edge_diff", None)
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        if shifts is not None:
            vectors = pos[tgt] - pos[src] + shifts
        elif edge_diff is not None:
            implied = edge_diff.detach() - (pos.detach()[tgt] - pos.detach()[src])
            vectors = pos[tgt] - pos[src] + implied
        else:
            raise ValueError(
                "edges.edge_index is placed but carries no geometry to price it by: "
                "edges.shifts or edges.edge_diff is required. A stored edges.edge_dist is "
                "not geometry — it is not a function of the live positions, so forwarding it "
                "prices a finite energy with structurally zero forces. "
                "molix.batch.NeighborList writes edge_diff beside edge_dist; "
                "molix.md.NeighborList writes edges.shifts as well (the branch whose virial "
                "is complete). molpot never re-enumerates the pairs itself."
            )
        dist = (vectors * vectors).sum(-1).clamp_min(_SQ_DIST_CLAMP_MIN).sqrt()
        live = src < tgt
        num_edges = edges.get("num_edges", None)
        if num_edges is not None:
            rows = torch.arange(edge_index.shape[0], device=edge_index.device)
            live = live & (rows < num_edges)
        if cutoff is not None:
            live = live & (dist < cutoff) & (dist > 0)
        return edge_index, vectors, dist, live

    def _scaled_pairs(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        *,
        cutoff: float | None,
        scale_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Half list ``(edge_index, pair_vectors, dist, scales, live)`` for one pair term.

        The list keeps the capacity :meth:`_placed_pairs` read it at — ``live``
        is the boolean mask of the rows that contribute, not a selection to
        apply. ``scales`` is the exclusion factor of every row including the
        dead ones (which read ``1.0``), so the term multiplies instead of
        slicing. There is no "no list" answer: a batch with zero rows prices at
        zero and a batch with no ``edges`` namespace raises, because molpot
        does not enumerate its own pairs.

        The one seam every pair term reaches for, so it is where the bound
        table is checked against the batch. Binds the topology on first use
        (:meth:`bind_topology`) — eager convenience, not a rebind — and from
        then on compares the batch's shape-level fingerprint against the bound
        one, raising when they differ instead of scaling this batch's pairs by
        the previous topology's relations. A ``Potentials`` reused over a
        different topology must be rebound explicitly.

        Args:
            frame: The batch's typed view, carrying ``atoms.pos`` and the
                topology namespaces, as :meth:`bind_topology` reads them.
            pos: Live positions ``(N, 3)`` in Å the placed list is priced at.
            cutoff: Real-space cutoff ``r_c`` in Å of the asking term, or
                ``None`` to keep every row of the placed list.
            scale_kind: Interaction class of the asking term — one of the
                ForceField's :meth:`~molpot.ir.NonbondedScaling.kinds`
                (``"q"`` for Coulomb, ``"lj"`` for Lennard-Jones).

        Returns:
            ``(edge_index, pair_vectors, dist, scales, live)`` — pairs
            ``(E, 2)``, their source → target separation vectors ``(E, 3)`` in
            Å, the separations ``(E,)`` in Å those vectors' norms give,
            dimensionless scale factors ``(E,)`` and the boolean liveness mask
            ``(E,)``, all at the placed list's capacity. The vectors are
            :meth:`_placed_pairs`'s, forwarded unchanged: a term that wants a
            direction takes it from here rather than rebuilding one.

        Raises:
            ValueError: The batch's topology has a different shape than the
                bound one (atom count, or the ``bonds`` / ``angles`` /
                ``propers`` column counts) — the message names
                :meth:`bind_topology`, which is the fix; or the batch carries
                no usable placed list (:meth:`_placed_pairs`).
        """
        if self._exclusions is None:
            self.bind_topology(frame, pos=pos)
        else:
            current = self._topology_fingerprint(frame, pos.shape[0])
            if current != self._bound_topology:
                raise ValueError(
                    "this Potentials is bound to a topology with fingerprint "
                    f"{self._bound_topology} but the batch has {current} "
                    "(n_atoms, bonds, angles, propers; -1 = namespace absent); its "
                    "exclusion table is never rebuilt implicitly — call "
                    "bind_topology(system) for the new topology"
                )
        if self._exclusions is None:
            raise RuntimeError(
                "_scaled_pairs reached its exclusion lookup with no bound exclusion table; "
                "the branch above binds one on first use and the other rejects a topology "
                "change, so neither can leave it unset — call bind_topology(system)"
            )
        edge_index, vectors, dist, live = self._placed_pairs(frame, pos, cutoff=cutoff)
        scales = self._exclusions.lookup(edge_index, scale_kind, dtype=pos.dtype)
        return edge_index, vectors, dist, scales, live

    def _lj_energy(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(repulsion, dispersion)`` of the LJ style — zeros when no pair is in range."""
        if self.lj is None:
            raise RuntimeError(
                "_lj_energy called with no lj/cut style assembled; every caller reaches it "
                "under `if self.lj is not None`"
            )
        epsilon, sigma = self._lj_params()
        # Keep LJ on its own cutoff list. Reusing an unlimited Coulomb half-list
        # forced the exclusion lookup (and distance storage) onto all
        # intramolecular pairs even though LJ only needs ``r < cutoff``.
        edge_index, _vectors, dist, scales, live = self._scaled_pairs(
            frame, pos, cutoff=self.lj.cutoff, scale_kind="lj"
        )
        num_graphs = self._num_graphs(frame)
        atom_batch = self._atom_batch(frame, pos.shape[0], pos.device)
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
        frame: TensorFrame,
        pos: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """``((repulsion, dispersion), forces)`` of the LJ style."""
        if self.lj is None:
            raise RuntimeError(
                "_lj_energy_forces called with no lj/cut style assembled; every caller "
                "reaches it under `if self.lj is not None`"
            )
        epsilon, sigma = self._lj_params()
        edge_index, vectors, dist, scales, live = self._scaled_pairs(
            frame, pos, cutoff=self.lj.cutoff, scale_kind="lj"
        )
        num_graphs = self._num_graphs(frame)
        atom_batch = self._atom_batch(frame, pos.shape[0], pos.device)
        return self.lj.energy_forces(
            # The geometry's atom count, not the parameter table's length.
            # ``pos.shape[0]`` stays a shape (a ``SymInt`` under dynamic
            # shapes), which is what ``new_zeros`` wants — never ``int(...)``,
            # which would burn a host read into the traced graph.
            n_atoms=pos.shape[0],
            edge_index=edge_index,
            # The vector ``dist`` is the norm of — under a cell it is the
            # minimum image, which ``pos`` alone cannot reconstruct.
            pair_vectors=vectors,
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
        frame: TensorFrame,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        q = self._charges()
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")

        cell = self._cell(frame)
        if cell is None:
            if self.coulomb_open is None:
                raise RuntimeError(
                    "this batch carries no cell, so Coulomb takes the open-boundary path, "
                    "but no CoulombOpenPotential was assembled; _assemble_pair_potentials "
                    "builds the open and periodic terms together from one 'pair:coulomb' "
                    "style, so neither can exist without the other"
                )
            cut = self.coulomb_open.cutoff
            edge_index, _vectors, dist, scales, live = self._scaled_pairs(
                frame, pos, cutoff=cut, scale_kind="q"
            )
            num_graphs = self._num_graphs(frame)
            atom_batch = self._atom_batch(frame, pos.shape[0], pos.device)
            return self.coulomb_open.energy(
                edge_index=edge_index,
                distances=dist,
                charges=q,
                scales=scales,
                edge_batch=atom_batch[edge_index[:, 0]] if num_graphs > 1 else None,
                num_graphs=num_graphs,
                edge_mask=live,
            )

        if self.coulomb_periodic is None:
            raise RuntimeError(
                "this batch carries a cell, so Coulomb takes the periodic path, but no "
                "CoulombPeriodicPotential was assembled; _assemble_pair_potentials builds "
                "the open and periodic terms together from one 'pair:coulomb' style, so "
                "neither can exist without the other"
            )
        num_graphs = self._num_graphs(frame)
        if num_graphs > 1:
            # One lattice, one reciprocal sum, one number: ``_cell`` reads
            # ``graphs.cell[0]`` and the term below returns a scalar that
            # ``calc_energy`` would add into every entry of its ``(B,)``
            # accumulator — each graph charged the whole box's electrostatics,
            # evaluated in graph 0's cell, with no other cell ever read.
            # Per-graph periodic electrostatics needs a per-graph reciprocal
            # sum and a list that never bridges two graphs (finding F4 in
            # ``.claude/notes/composable-potentials.md``, which owns its own
            # spec); until it lands the batch is refused, not answered wrongly.
            raise ValueError(
                "periodic Coulomb evaluates one graph per cell, but this batch has "
                f"num_graphs={num_graphs}: the reciprocal sum would price the whole box "
                "in graph 0's cell and hand that same energy to every graph. Per-graph "
                "periodic batches are not supported yet — evaluate one periodic system "
                "per batch, or drop graphs.cell for the open-boundary Coulomb path."
            )
        # Neighbor list cutoff from the periodic Potential's real-space cutoff
        edge_index, _vectors, dist, _, live = self._scaled_pairs(
            frame,
            pos,
            cutoff=self.coulomb_periodic.real_space_cutoff,
            scale_kind="q",
        )
        # Per-row scales are *not* what the periodic term wants: under Ewald an
        # excluded pair also sits in the reciprocal sum, so it is corrected over
        # the topology's own pair table (whatever the neighbour list happened to
        # enumerate) rather than attenuated in the real-space sum.
        # ``_scaled_pairs`` has bound the table by now.
        if self._exclusions is None:
            raise RuntimeError(
                "periodic Coulomb reached its exclusion correction with no bound exclusion "
                "table; _scaled_pairs binds one on first use, so this means the topology was "
                "never bound — call bind_topology(system) before evaluating"
            )
        exclusions = self._exclusions
        excluded_pairs = exclusions.pairs
        return self.coulomb_periodic.energy(
            positions=pos,
            charges=q,
            cell=cell,
            edge_index=edge_index,
            distances=dist,
            edge_mask=live,
            excluded_pairs=excluded_pairs,
            excluded_scales=exclusions.lookup(excluded_pairs, "q", dtype=pos.dtype),
        )

    def _coulomb_energy_forces(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._charges()
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")

        # Open Coulomb and Ewald/PME/P3M all share the molpot Coulomb kernel
        # stack; neither exposes a cheap closed form for forces, so the single
        # force path is one autograd backward on a fresh position leaf. Both
        # halves of that seam — the leaf and the backward — belong to
        # ``molpot.derivation.force``: this method runs inside the compiled
        # classical core, and the two graph breaks they cost must be reported
        # against the module that owns them, not against the Coulomb energy.
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            p_nb = position_leaf(pos)
            nb_energy = self._coulomb_energy(frame, p_nb)
            create_graph = grad_enabled and q.requires_grad
            nb_forces = (
                autograd_forces_from_energy(nb_energy, p_nb, create_graph=create_graph)
                if nb_energy.requires_grad
                else pos.new_zeros((n_atoms, 3))
            )
        # ``create_graph=False`` freed ``nb_energy``'s graph inside the grad call;
        # detach it so a later ``loss.backward()`` never touches the freed graph.
        return nb_energy if create_graph else nb_energy.detach(), nb_forces

    def _rebuilds_pairs(self) -> bool:
        """Whether a declared term rebuilds its pair vectors from the placed list.

        The three nonbonded terms reach the neighbour list through
        :meth:`_scaled_pairs`; the bonded ones read ``pos`` and their own index
        columns, and an mlip reads ``edges.edge_diff`` as handed to it. Only the
        first group is served by :meth:`_placed_pairs` and therefore cares
        whether that list carries ``shifts``.
        """
        return (
            self.lj is not None
            or self.coulomb_open is not None
            or self.coulomb_periodic is not None
        )

    def _atom_batch(self, frame: TensorFrame, n: int, device: torch.device) -> torch.Tensor:
        if "atoms" in frame.keys():
            b = frame.atoms.get("batch", None)
            if b is not None:
                return b.long()
        return torch.zeros(n, dtype=torch.long, device=device)

    def _num_graphs(self, frame: TensorFrame) -> int:
        """How many graphs the energy is summed into — never a device sync.

        Every term needs this number to size its energy vector, so it is read
        on the hot path, inside :meth:`_classical_core` and again inside each
        pair term. Two of the three ways to get it are metadata (a sized
        ``graphs`` batch_size, or ``graphs.num_atoms``' length); the third —
        the largest entry of ``atoms.batch`` — is a **value**, and reading a
        value to Python is a host sync and a graph break under dynamo. So it is
        resolved once at :meth:`bind_topology` (:meth:`_count_graphs`) and
        served from there.

        A bind is per batch *shape* as well as per topology: rebind before
        evaluating a batch with a different graph count, exactly as for the
        exclusion table. Which batch the bound number may be served to is
        therefore the same question :meth:`_scaled_pairs` asks about the bound
        table, and it is asked the same way — the batch's whole
        :meth:`_topology_fingerprint` against ``_bound_topology``, not just its
        first entry. The atom count alone is not identity: two batches of the
        same ``N`` can hold a different number of molecules (fourteen atoms as
        one butane or as two heptane fragments), and answering the second with
        the first's count sizes every term's energy vector wrong — a shape
        error at best, a batch's energy summed into one graph at worst. The
        limit is the fingerprint's own: two topologies of equal shape are
        indistinguishable here, as everywhere the fingerprint is used.

        Args:
            frame: The typed view of the batch being evaluated; on the served
                path its ``atoms.pos`` length and the shapes of its columns are
                read — shapes only, so nothing here is a host sync, and a trace
                folds the reads to constants.

        Returns:
            The graph count ``B`` — ``1`` for an unbatched single molecule.
        """
        bound = self._bound_num_graphs
        fingerprint = self._bound_topology
        if bound is not None and fingerprint is not None:
            pos = frame.atoms.get("pos", None) if "atoms" in frame.keys() else None
            # No ``atoms.pos`` is no atom count to compare; the topology columns
            # still are, so the rest of the fingerprint still has to match.
            n_atoms = fingerprint[0] if pos is None else int(pos.shape[0])
            if self._topology_fingerprint(frame, n_atoms) == fingerprint:
                return bound
        return self._count_graphs(frame)

    def _count_graphs(self, frame: TensorFrame) -> int:
        """Resolve the graph count from the batch — the sync :meth:`_num_graphs` avoids.

        Three sources, cheapest first: the ``graphs`` namespace's own
        ``batch_size``, then the length of ``graphs.num_atoms``, then — only if
        neither exists — the largest entry of ``atoms.batch``, which is a tensor
        *value* and therefore the one host sync of the three. Called from
        :meth:`bind_topology`, where a sync is affordable, and from
        :meth:`_num_graphs` when the batch is not the bound one.

        Args:
            frame: The typed view whose ``graphs`` / ``atoms`` namespaces are read.

        Returns:
            The graph count ``B``; ``1`` when the batch names no graphs at all.
        """
        if "graphs" in frame.keys():
            graphs = frame.graphs
            if graphs.batch_size:
                return int(graphs.batch_size[0])
            num_atoms = graphs.get("num_atoms", None)
            if num_atoms is not None and num_atoms.ndim >= 1:
                return int(num_atoms.shape[0])
        atom_batch = frame.atoms.get("batch", None) if "atoms" in frame.keys() else None
        if atom_batch is not None and atom_batch.numel():
            # ``max``, not ``[-1]``: a collate that interleaves graphs still
            # answers with the count rather than with the last row's id.
            return int(atom_batch.detach().max().to("cpu")) + 1
        return 1

    # --- topology readers ---

    def _frame(self, system: TensorDict | TensorFrame | Mapping[str, Any] | None) -> TensorFrame:
        """This container's one accessor: the batch, as its typed view.

        Every public entry point calls this **once, eagerly**, and hands the
        view to everything below — the readers, the checks and
        :meth:`_classical_core` — so a compiled core sees the same leaves a
        plain batch would give it and the wrap never appears inside a trace.

        The wrap is memoized on the batch *object*, by weak reference, and this
        is a correctness requirement rather than a saving. The F29 geometry
        memo (:meth:`_check_geometry`) is keyed on the identity of the system it
        vouched for; a fresh wrapper per call would miss that memo on every
        step, which turns a once-per-frame host sync into a per-step one and
        puts it inside any region a caller compiles. A system that cannot be
        weakly referenced (a plain ``dict``) is simply re-wrapped, exactly as
        that memo re-checks it.

        Args:
            system: The batch — a ``TensorDict``, a :class:`TensorFrame` view
                of one, or a nested ``Mapping`` of its namespaces. ``None`` is
                the ``BasePotential`` "no system, positions passed as a keyword"
                call and reads as a system carrying no namespaces at all.

        Returns:
            The typed view (:func:`as_tensor_frame`).

        Raises:
            TypeError: *system* is not a batch, a view or a mapping.
        """
        if isinstance(system, TensorFrame):
            return system
        if system is None:
            # One empty view per container, not one per call: the ``pos=``-only
            # path is memoized by identity like any other frame, so a fresh
            # wrapper here would miss the F29 memo on every such call.
            empty = self._empty_frame
            if empty is None:
                empty = self._empty_frame = as_tensor_frame(TensorDict({}, batch_size=[]))
            return empty
        cached = self._frame_view
        if cached is not None and cached[0]() is system:
            return cached[1]
        view = as_tensor_frame(system)
        try:
            # Weak key, strong value — deliberately asymmetric. The key must not
            # keep the batch alive (a dead reference misses instead of matching
            # a reused address); the value is the view, which holds the batch's
            # leaves, so the previous batch stays alive until the next distinct
            # one arrives. One extra batch per step in training, none in MD.
            self._frame_view = (weakref.ref(system), view)
        except TypeError:
            # A mapping that cannot be weakly referenced is wrapped afresh each
            # call rather than keyed on an address the next system may reuse.
            self._frame_view = None
        return view

    def _pos(self, frame: TensorFrame, pos: torch.Tensor | None) -> torch.Tensor:
        if pos is not None:
            return pos
        if "atoms" in frame.keys():
            p = frame.atoms.get("pos", None)
            if p is not None:
                return p
        raise ValueError("calc_energy requires pos or system atoms.pos")

    def _topology_fingerprint(self, frame: TensorFrame, n_atoms: int) -> tuple[int, ...]:
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
            frame: The typed view whose ``bonds`` / ``angles`` / ``propers``
                namespaces are read, exactly as :meth:`bind_topology` reads them.
            n_atoms: Atom count ``N`` the table would be keyed against.

        Returns:
            ``(n_atoms, n_bonds, n_angles, n_propers)``, each ``-1`` when that
            namespace carries no usable index block.
        """
        counts = [int(n_atoms)]
        for reader in (self._bond_index, self._angle_index, self._proper_index):
            index = reader(frame)
            counts.append(-1 if index is None else int(index.shape[-1]))
        return tuple(counts)

    def _bond_index(self, frame: TensorFrame) -> torch.Tensor | None:
        """The covalent topology as the ``(2, N)`` COO index, or ``None``.

        Two spellings of the same block, because the batch has two producers:
        :func:`~molix.batch.collate.collate_molecules` writes
        ``bonds.bond_index`` (the schema's COO, whose shape is the anti-alias
        guard against ``edge_index``), while a caller assembling the namespace
        by hand — as the valence namespaces are — writes ``atomi`` / ``atomj``
        columns. Both are read; neither is preferred over the other beyond the
        order below.
        """
        if "bonds" not in frame.keys():
            return None
        bonds = frame.bonds
        packed = bonds.get("bond_index", None)
        if packed is not None:
            return packed
        atomi = bonds.get("atomi", None)
        atomj = bonds.get("atomj", None)
        if atomi is not None and atomj is not None:
            return torch.stack([atomi.long(), atomj.long()], dim=0)
        return None

    def _angle_index(self, frame: TensorFrame) -> torch.Tensor | None:
        if "angles" not in frame.keys():
            return None
        return self._term_index(frame.angles, ("atomi", "atomj", "atomk"))

    def _proper_index(self, frame: TensorFrame) -> torch.Tensor | None:
        if "propers" not in frame.keys():
            return None
        return self._term_index(frame.propers, ("atomi", "atomj", "atomk", "atoml"))

    def _improper_index(self, frame: TensorFrame) -> torch.Tensor | None:
        if "impropers" not in frame.keys():
            return None
        return self._term_index(frame.impropers, ("atomi", "atomj", "atomk", "atoml"))

    @staticmethod
    def _term_index(namespace: TensorDict, columns: tuple[str, ...]) -> torch.Tensor | None:
        """One valence namespace's index columns stacked into ``(arity, N)``.

        Args:
            namespace: The ``angles`` / ``propers`` / ``impropers`` namespace.
            columns: Its index column names, in index order.

        Returns:
            The stacked index, or ``None`` when the namespace is missing one of
            the columns its arity needs — a namespace that cannot be read is
            not a topology, and the term it belongs to contributes an explicit
            zero rather than a partial index.
        """
        cols = [namespace.get(name, None) for name in columns]
        if any(column is None for column in cols):
            return None
        return torch.stack([column.long() for column in cols], dim=0)

    @staticmethod
    def _identity_types(n: int, device: torch.device) -> torch.Tensor:
        return torch.arange(n, device=device, dtype=torch.long)

    def __repr__(self) -> str:
        return f"Potentials(forcefield={self.forcefield!r})"
