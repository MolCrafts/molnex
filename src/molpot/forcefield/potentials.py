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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from molix import logger as _logger_mod
from molix.autograd import Derivative
from molix.autograd.variables import get_column
from molix.frame import TensorFrame, as_tensor_frame
from molix.schema import STRAIN_KEY, CellBound
from molpot.contract import Contribution, ExclusionClass
from molpot.derivation.protocol import ENERGY_KEY, call_energy
from molpot.forcefield.exclusions import ExclusionTable
from molpot.forcefield.pair_terms import (
    CoulombOpenPotential,
    CoulombPeriodicPotential,
    LJCutPotential,
)
from molpot.forcefield.partial_hessian import TermHessian
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
logger = _logger_mod.getLogger(__name__)

PAIR_STYLE_DEFAULTS = PairStyleDefaults()

#: Floor on the per-graph deviation of ``Σ_g q`` from the declared
#: ``graphs.total_charge`` that passes without comment (e). The full tolerance
#: is this floor or :data:`_TOTAL_CHARGE_ROUNDOFF` times the graph's own
#: rounding scale, whichever is larger — see
#: :meth:`Potentials._warn_net_charge_drift`.
#:
#: The floor is what an fp32 **solve** returns, not what a sum costs. The
#: charges come out of an EEM system whose Lagrange row is the constraint, and
#: its residual follows the conditioning of that system, not ``Σ|q|`` of the
#: answer: measured drifts of 3.8e-6 e appear on small molecules where the
#: size-scaled term is only 3.6e-6 e, i.e. 21·u·Σ|q|. Below ~1e-5 e the two
#: causes cannot be told apart at this precision — and 1e-5 e is still three
#: orders below the 0.01 e of a head that has left its constraint.
#:
#: It is quoted **at float32** and scaled by the working precision, because
#: "what the solve can return" is a property of the arithmetic, not a constant:
#: in float64 the same 5e-6 e miss is a real error and must be reported.
#:
#: Exceeding it **warns**; it does not refuse the batch (operator ruling
#: 2026-09-11). The drift this catches has two very different causes and only
#: one of them is a modelling error: a charge head that has walked off the
#: constraint (0.01 e and up, a real wrong net charge) and fp32 accumulation
#: over a packed batch (arithmetic). Refusing both stopped production
#: training on the second, which is not a fault of the model. The magnitude is
#: reported so the first is still visible and still has to be explained.
_TOTAL_CHARGE_TOL = 1e-5

#: Unit roundoff the floor above is quoted at.
_TOTAL_CHARGE_TOL_EPS = float(torch.finfo(torch.float32).eps)

#: Multiple of the fp32 rounding scale ``u · √n · Σ|q|`` allowed before the
#: drift is called a modelling error. Summing *n* floats carries a rounding
#: error up to ``(n−1)·u·Σ|x|`` and about ``√n·u·Σ|x|`` in practice, so the
#: bound has to carry the atom count: a 350-atom molecule is not a 2-atom one.
#: Measured on a production-shaped OMol25 batch, the worst drift was ``14·u·Σ|q|``
#: — inside the deterministic bound, and the charges come out of an fp32 EEM
#: solve whose own residual adds to it. 4 leaves ~5× headroom over what is
#: observed while staying ~90× below the 0.01 e of a head off its constraint.
_TOTAL_CHARGE_ROUNDOFF = 4.0

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


class Potentials(nn.Module):
    """Classical Class-I energy on the batch schema (``nn.Module``).

    Embed in a larger torch model, or reach MD through the owning
    :class:`~molpot.forcefield.container.ForceField` and
    :meth:`molix.md.MD.set_forcefield`. :meth:`energy` writes
    ``graphs.energy`` onto a batch; :meth:`forward` is the
    :class:`molix.autograd.Derivative` built over it from the ``wrt`` keys
    given at construction, so ``Potentials(ff)(batch)`` writes the energy
    alone and ``Potentials(ff, wrt=("atoms.pos",))(batch)`` the energy and
    forces — the same op ``Derivative(pots, wrt=...)`` composes outside, which
    is how MD takes forces and virial. There is no analytic force branch:
    every derivative is the op's one ``torch.autograd.grad`` through the same
    energy, and the pair terms hand the minimum-image separation *vector* the
    energy is built from (:meth:`_placed_pairs`) onto that graph.

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

    Every pair term above reads the ``pairs`` neighbour list already on the
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
        wrt: TensorFrame keys :meth:`forward` differentiates the energy
            against (:class:`molix.autograd.Derivative`); ``()`` is energy
            only. ``graphs.strain`` needs a one-graph periodic batch whose
            placed list carries ``edges.shifts`` (F13, see :meth:`energy`).

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

    def __init__(self, forcefield: ForceField, *, wrt: Sequence[str | Sequence[str]] = ()) -> None:
        super().__init__()
        self.forcefield = forcefield
        self.lj: LJCutPotential | None = None
        self.coulomb_open: CoulombOpenPotential | None = None
        self.coulomb_periodic: CoulombPeriodicPotential | None = None
        self.mlip = getattr(forcefield, "_mlip", None)
        self._exclusions: ExclusionTable | None = None
        # Exclusion-table fingerprint, not an F29 cache: a mismatch against the
        # live batch raises and demands an explicit rebind (:meth:`_scaled_pairs`).
        self._bound_topology: tuple[int, ...] | None = None
        # How many graphs the bound batch holds, resolved once at
        # :meth:`bind_topology` because reading it off ``atoms.batch`` is a host
        # sync and therefore a graph break on a compiled core (:meth:`_num_graphs`).
        self._bound_num_graphs: int | None = None
        #: The field's own per-atom tables at the bound topology's ``N``; see
        #: :meth:`bind_topology`. ``None`` until the first bind.
        self._bound_atom_params: dict[str, torch.Tensor] | None = None
        # Wrap identity of the last TensorDict this container wrapped
        # (:meth:`_frame`). Compiled MD evaluates ``op(work)`` on the caller's
        # TensorDict; this memo keeps :func:`~molix.data.as_tensor_frame` out
        # of the CUDA-graph region. Not an F29 validity voucher.
        self._frame_view: tuple[weakref.ref[Any], TensorFrame] | None = None
        self._bonded_styles: frozenset[str] = frozenset()
        # The one derivative route: the op over :meth:`energy`, fixed at
        # construction (a bound method, so no submodule is registered).
        self._derivative = Derivative(self.energy, wrt=wrt)
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
            )
        else:
            self.improper = None

    @staticmethod
    def _opls_to_cosine(pp: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """OPLS ``k1..k4`` → the canonical cosine ``k`` / ``phase``.

        The OPLS quartet *is* the canonical series at ``n = 1,2,3,4`` with
        ``γ = 0, π, 0, π`` and ``k_m = K_m / 2`` — a dialect, not a form (spec
        ff-params-01). A live ``dihedral:opls`` head still speaks it, so this
        is where it is folded in.

        Live tables from a ``dihedral:opls`` head keep the OPLS spelling.
        The term looks up ``k``, not ``k1``; without this map a reused
        container prices the seed ``k`` (profile 2026-09-13: 295 vs 316).
        """
        k_opls = torch.stack([pp["k1"], pp["k2"], pp["k3"], pp["k4"]], dim=-1)
        k = 0.5 * k_opls
        phase = torch.zeros_like(k)
        phase[:, 1] = math.pi
        phase[:, 3] = math.pi
        # The dialect fixes all three: n = 1,2,3,4 for every type it names.
        # Emitting it here is what lets a caller hand in this batch's k1..k4
        # for a topology the seed never saw — the multiplicities come with it
        # instead of being gathered out of a seed table of another length.
        periodicity = (
            torch.tensor([1, 2, 3, 4], device=k.device, dtype=torch.long)
            .expand(k.shape[0], 4)
            .contiguous()
        )
        return {"k": k, "phase": phase, "periodicity": periodicity}

    def _make_proper(self, pp: dict[str, torch.Tensor]) -> ProperTorsionPeriodic:
        if "k1" in pp:
            pp = self._opls_to_cosine(pp)
            return ProperTorsionPeriodic(
                k=pp["k"], periodicity=pp["periodicity"], phase=pp["phase"]
            )
        pp = ProperTorsionBag._cosine_fields(pp)
        return ProperTorsionPeriodic(
            k=pp["k"],
            periodicity=pp["periodicity"],
            phase=pp["phase"],
        )

    @staticmethod
    def _term_tables(
        called: Mapping[str, torch.Tensor],
        namespace: TensorDict | None,
        term: BasePotential | None,
        names: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Caller's tables, else the term buffers, with dummy rows zeroed."""
        out = dict(called)
        if term is not None:
            for name in names:
                if name not in out:
                    value = getattr(term, name, None)
                    if value is not None:
                        out[name] = value
        return Potentials._mask_term_tables(out, namespace)

    @staticmethod
    def _mask_term_tables(
        called: Mapping[str, torch.Tensor],
        namespace: TensorDict | None,
    ) -> dict[str, torch.Tensor]:
        """Zero dummy-term rows (``mask`` False) so ghost valence prices nothing.

        Dummy geometry is finite (four offset ghosts) so ``‖r‖`` at 0 never
        enters autograd; ``k * 0`` is what drops the energy, not a slice.
        """
        out = dict(called)
        if namespace is None or "mask" not in namespace.keys():
            return out
        mask = namespace["mask"]
        for name, value in list(out.items()):
            if not torch.is_tensor(value) or value.shape[0] != mask.shape[0]:
                continue
            scale = mask.to(dtype=value.dtype)
            if value.ndim > 1:
                scale = scale.reshape(-1, *([1] * (value.ndim - 1)))
            out[name] = value * scale
        return out

    @staticmethod
    def _called_proper_tables(called: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predicted proper tables in the term's cosine spelling."""
        if called and "k1" in called:
            return Potentials._opls_to_cosine(called)
        if called and ("a" in called or "b" in called):
            return dict(ProperTorsionBag._cosine_fields(dict(called)))
        return dict(called)

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

    def forward(self, system: TensorDict | TensorFrame) -> TensorDict | TensorFrame:
        """The batch with ``graphs.energy`` — and the ``wrt`` derivatives — written.

        The :class:`molix.autograd.Derivative` of construction over
        :meth:`energy`: with ``wrt=()`` it is :meth:`energy` itself, with
        ``wrt=("atoms.pos",)`` it adds ``atoms.forces``, with
        ``("atoms.pos", "graphs.strain")`` ``graphs.virial`` too. Composes into
        the MD driver's force adapter and any training step.

        Args:
            system: Post-collate batch, or its :class:`~molix.data.TensorFrame`
                view — the view *is* the caller's leaves (:meth:`_frame`), so a
                write through it lands in the caller's own storage.

        Returns:
            The same ``system`` object, written in place.
        """
        return self._derivative(system)

    @property
    def derivative(self) -> Derivative:
        """The op :meth:`forward` runs: ``Derivative(self.energy, wrt=...)``, fixed in ``__init__``.

        What a training step marks its leaves through
        (:meth:`molix.autograd.Derivative.mark_leaves`).
        """
        return self._derivative

    def energy(
        self,
        system: TensorDict | TensorFrame,
        tables: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> TensorDict | TensorFrame:
        """Write ``graphs.energy`` onto the batch — the Derivative seam.

        The kernel compiled MD captures: :meth:`_sum_energy` plus this write.
        It does **not** call :meth:`calc_energy`, :meth:`_check_geometry` or
        :meth:`_check_bonds_unwrapped`, and it has no ``is_compiling()`` /
        identity skip to fake an F29 voucher. F29 is public
        :meth:`calc_energy` / :meth:`bind_topology` / :meth:`warmup` only.
        Bonded-only ``MD.run`` is in this compiled-energy residual: the
        integrator never wraps in place, and the caller unwraps before
        ``run``. That is not ``bind_work`` vouching F29.

        :meth:`_sum_energy` shaped to the ``graphs`` namespace's own batch
        size: the classical sum is a **scalar** ``()`` on a single-molecule
        batch, which is the right shape for the sum it is, while a collated
        batch of one graph carries ``graphs`` at ``batch_size=[1]`` and a
        TensorDict refuses to store a ``()`` leaf under it. Every collate in
        molnex produces exactly that batch, so without the reshape the most
        ordinary case is the one that raises. A batch with no ``graphs``
        namespace gets one at ``batch_size=[]``.

        **Padded batches.** A :class:`molix.data.MoleculePadder` batch
        carries ghost atoms, dead ``(ghost, ghost)`` padding edges and dead
        graph slots beside ``atoms.mask`` / ``graphs.mask``. This container
        reads no mask, and needs none: it has no per-atom energy term — every
        declared style is a bonded term over the topology columns, whose dummy
        rows are zeroed by their own ``mask`` (:meth:`_mask_term_tables`), or
        a pair term over the placed list, whose ``source < target`` liveness
        rule already kills a self-loop and whose ghosts sit on no live edge —
        so the padded energy is the unpadded one, the ghosts' forces are zero
        and the dead graph slots price zero. A *pair* style prices a padded
        batch when its per-atom ``epsilon`` / ``sigma`` / ``charge`` tables are
        handed in **per batch** (``tables=``) at the padded atom count, as a
        parameterizer's heads predict them (the Class-I training path); the
        seed tables a ForceField carries at the real atom count cannot index
        a ghost, so a fixed-table field is priced unpadded. An mlip field
        hands the whole batch, mask included, to the mlip's own ``energy``.

        **Virial precondition (F13).** Under a strain variable
        (``graphs.strain`` on the frame — the derivative op puts it there) a
        classical pair term prices a placed list *without* ``edges.shifts`` by
        a lattice remainder it recovers as a **detached** constant, so the
        strain never reaches a pair that meets its partner through a lattice
        vector and its share of ``W = ∂E/∂ε`` is silently absent. Such a
        batch is refused here; energy and forces alone stay exact on it.

        Args:
            system: Post-collate batch or its typed view, carrying
                ``atoms.pos`` ``(N, 3)`` Å and the topology; an mlip field
                hands it whole to the mlip's own ``energy``.

        Returns:
            The same ``system`` object, with ``graphs.energy`` written
            (kcal/mol under ``units="real"``; the mlip's native unit on that
            branch).

        Raises:
            ValueError: A classical pair term is declared, ``graphs.strain`` is
                on the frame and the placed list carries no ``edges.shifts``
                (F13); or any of :meth:`term_energies`' refusals. A first
                unbound pair-field call also reaches :meth:`bind_topology`
                through :meth:`_scaled_pairs` and may raise that bind's F29
                refusal — bind-once exclusion work, not a later-geometry
                voucher. Bonded-only fields never take that path.
        """
        frame = self._frame(system)
        if (
            self.mlip is None
            and get_column(frame, STRAIN_KEY) is not None
            and get_column(frame, ("pairs", "pair_index")) is not None
            and get_column(frame, ("pairs", "shifts")) is None
            and self._rebuilds_pairs()
        ):
            raise ValueError(
                "a virial needs edges.shifts on a placed list under a cell when a classical "
                "pair term is declared: this list carries none, and the pair_diff fallback "
                "its pair vectors are rebuilt from is a detached constant that carries no "
                "lattice strain, so every pair reaching its partner through a lattice vector "
                "would be missing from the virial. Build the list with molix.md.NeighborList "
                "(which writes edges.shifts), or ask for energy and forces only — those are "
                "exact on this batch."
            )
        total = self._sum_energy(frame, tables)
        graphs_ns = getattr(frame, "_tensordict", None)
        if not (isinstance(graphs_ns, dict) and graphs_ns.get("graphs") is not None):
            frame["graphs"] = TensorDict({}, batch_size=[])
        graphs_bs = frame.graphs.batch_size
        # A ``batch_size=[]`` graphs namespace constrains nothing, so the
        # energy is stored as produced; a sized one is the shape to answer in.
        frame[ENERGY_KEY] = total.reshape(*graphs_bs) if graphs_bs else total
        # The caller's own container back, not the view: the write above went
        # into its storage, so the one that was handed in composes onward.
        return system

    #: Bonded namespaces the partial-Hessian target covers, and how each is
    #: priced: the attribute holding its term, the index kwarg its forward
    #: reads, the ``*_types`` kwarg to null out, and the parameter names to
    #: backfill from the term's own buffers.
    _PARTIAL_HESSIAN_TERMS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
        ("bonds", "bond", "bond_index", "bond_types", ("k", "r0")),
        ("angles", "angle", "angle_index", "angle_types", ("k", "theta0")),
        ("propers", "proper", "proper_index", "proper_types", ()),
        ("impropers", "improper", "improper_index", "improper_types", ()),
    )

    def partial_hessian(
        self,
        system: TensorDict | TensorFrame,
        tables: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """The MM partial Hessian on the pairs ``hessian_pairs`` names.

        Stage 1 of the ByteFF curriculum fits harmonic force constants against
        the quantum Hessian, but not the whole ``(3N, 3N)`` of it: only the
        ``3x3`` mixed blocks ``d2E / dx_a dx_b`` of the atom pairs that share a
        bonded term. Those pairs are enumerated once by
        :class:`~molpot.forcefield.partial_hessian.PartialHessianPairs` and
        arrive as the ``hessian_pairs`` namespace, whose ``block`` column is the
        quantum label; each valence namespace carries ``hpair_<i>_<j>`` columns
        saying which row of it its own ``(i, j)`` block belongs to.

        **A pair's block is a sum over terms.** A 1-2 pair sits in a bond, in
        every angle and torsion through it, and the quantum block is the total
        curvature, so the contributions accumulate into the pair's row before
        anything is compared. That is why the pairs are deduplicated and why
        this returns one row per pair rather than one per term.

        Ghost valence rows contribute exactly zero without being masked here:
        :meth:`_mask_term_tables` has already zeroed their force constants, so
        their energy — and hence their curvature — is zero wherever their
        ``hpair`` columns point.

        Geometry is differentiated; the parameters keep their graph, so a loss
        on the returned blocks reaches the force constants that produced them.
        ``atoms.pos`` must already require grad — the caller marks it on the host,
        because creating a leaf inside a compiled region does not trace. Nothing
        here reads the gradient *with respect to* the positions; they are the
        variable the curvature is taken in, and stage 1 still fits only parameters.
        Grad is enabled around the gather, so an eval phase under ``no_grad`` still
        measures a curvature rather than the curvature of a constant.

        Args:
            system: Post-collate batch or its typed view, carrying
                ``atoms.pos``, the valence namespaces with their
                ``hpair_<i>_<j>`` columns, and ``hessian_pairs``.
            tables: Per-namespace predicted parameter tables, as for
                :meth:`energy`.

        Returns:
            ``(n_pairs, 9)`` — each row a row-major ``3x3`` block, in
            ``hessian_pairs`` order, in kcal mol⁻¹ Å⁻² under ``units="real"``.

        Raises:
            ValueError: *system* carries no ``hessian_pairs`` namespace, or a
                valence namespace carries ``hpair`` columns while no declared
                style can price it.
        """
        frame = self._frame(system)
        if "hessian_pairs" not in frame.keys():
            raise ValueError(
                "Potentials.partial_hessian: the batch carries no hessian_pairs namespace. "
                "It is the target's own pair list — build it with "
                "PartialHessianPairs and collate it beside the valence namespaces."
            )
        pos = self._pos(frame)
        pairs = frame["hessian_pairs"]
        n_pairs = int(pairs["atomi"].shape[0])
        blocks = pos.new_zeros((n_pairs, 9))
        called_all = tables or {}
        hessian = TermHessian()

        for label, attribute, index_kwarg, types_kwarg, names in self._PARTIAL_HESSIAN_TERMS:
            if label not in frame.keys():
                continue
            namespace = frame[label]
            choices = [key for key in sorted(namespace.keys()) if key.startswith("hpair_")]
            if not choices:
                continue
            term = getattr(self, attribute, None)
            index = getattr(self, f"_{attribute}_index")(frame)
            if term is None or index is None or index.shape[1] == 0:
                raise ValueError(
                    f"Potentials.partial_hessian: {label} carries {len(choices)} hpair "
                    f"columns but no {attribute} term could price it; declare the style "
                    f"and give it a parameter table, or drop the columns."
                )
            asked = dict(called_all.get(label, {}))
            if label == "propers":
                resolved = self._called_proper_tables(
                    self._term_tables(asked, namespace, term, self._proper_table_names(asked))
                )
            elif label == "impropers":
                resolved = self._term_tables(
                    asked, namespace, term, self._improper_table_names(term)
                )
            else:
                resolved = self._term_tables(asked, namespace, term, names)

            n_terms, width = int(index.shape[1]), int(index.shape[0])
            local_index = (
                torch.arange(n_terms * width, device=pos.device, dtype=torch.long)
                .reshape(n_terms, width)
                .T.contiguous()
            )
            call = {
                index_kwarg: local_index,
                types_kwarg: None,
                "term_batch": None,
                "num_graphs": None,
                **resolved,
            }

            def energy(local: torch.Tensor, _term=term, _call=call) -> torch.Tensor:
                return _term(pos=local.reshape(-1, 3), **_call)

            # `enable_grad` around the **gather**, not just around the double
            # backward inside `blocks`: under `torch.no_grad()` — which is every
            # eval phase — `pos[index]` produces a tensor that tracks nothing even
            # when `pos` itself is marked, and the curvature of a constant is not
            # what this measures. A derivative needs grad regardless of the mode it
            # is called in, which is why `Derivative` enables it too.
            with torch.enable_grad():
                per_term = hessian.blocks(energy, pos[index.T])
            # The namespace's own columns are the choice list: whatever
            # PartialHessianPairs enumerated is what it wrote, so the two can
            # never drift.
            for key in choices:
                i, j = (int(part) for part in key.removeprefix("hpair_").split("_"))
                blocks = blocks.index_add(
                    0,
                    namespace[key].long(),
                    per_term[:, i, :, j, :].reshape(n_terms, 9),
                )
        return blocks

    @staticmethod
    def _proper_table_names(called: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
        """Which proper-torsion dialect the caller is speaking.

        ``ProperTorsionPeriodic`` stores the polar cosine form, so backfilling
        its ``k`` / ``phase`` beside an OPLS ``k1..k4`` or a Cartesian
        ``(a, b)`` table would mix a seed's spelling *and* its term count into
        this batch — which is exactly what
        :meth:`~molpot.ir.bags.ProperTorsionBag._cosine_fields` refuses. Read
        the dialect off the caller's own keys instead.
        """
        if "k1" in called:
            return ("k1", "k2", "k3", "k4")
        if "a" in called or "b" in called:
            # Cartesian ``(a, b)`` is a complete dialect of its own — it is what
            # ``ProperTorsionParamHead`` predicts, because ``(k, gamma)`` has a
            # coordinate singularity at ``k = 0``.
            return ("a", "b", "periodicity")
        return ("k", "phase")

    @staticmethod
    def _improper_table_names(term: BasePotential) -> tuple[str, ...]:
        """The improper parameter names of whichever improper style is assembled."""
        if isinstance(term, ImproperHarmonic):
            return ("k", "chi0")
        return ("k", "phase", "periodicity")

    def warmup(self, system: TensorDict | TensorFrame) -> "Potentials":
        """Host setup for this batch: wrap, bind topology (F29 of *this* frame), bind cell.

        A build phase and a run phase, kept apart on purpose. Assembling a
        Class-I energy needs answers that only the host has: which styles the
        field declares and what their options resolved to, which pairs the
        exclusion table scales, whether **this** batch's molecules arrive
        unwrapped, how many mesh points this lattice needs. None of it is
        differentiable and none of it can be traced — a Python branch on a
        resolved option, an ``argmax`` read back to decide whether to raise, a
        table whose length is data-dependent.

        F29 here is :meth:`bind_topology`'s check of the coordinates and cell
        it is handed. Bind/F29 of the warmup batch is not bind/F29 of a
        replaced or mutated one: a later public :meth:`calc_energy` on a
        fresh wrapped TensorDict of the same shape, or on this object after
        an in-place wrap, still raises. This method does not set a ``_warm``
        flag and does not voucher later geometries.

        Call this once with a representative batch, then compile the
        Derivative seam (:meth:`energy`), not :meth:`calc_energy`::

            pots.warmup(batch)
            run = Compiler(fullgraph=True)(Derivative(pots, wrt=("atoms.pos",)))

        Idempotent: warming twice with the same batch does the work once.
        Warming with a *different* topology rebinds — which is what a new
        system means.

        Args:
            system: A batch shaped like the ones that follow: the same declared
                styles, the same namespaces, and a placed edge list when a pair
                style is declared. Positions ``atoms.pos`` ``(N, 3)`` Å and
                optional ``graphs.cell`` ``(3, 3)`` Å are the geometry F29
                reads on this call.

        Returns:
            ``self``, so the call chains.

        Raises:
            ValueError: Propagated from :meth:`bind_topology` — including F29
                when a bond is longer than half its cell's minimum
                perpendicular width (Å).
        """
        frame = self._frame(system)
        self.bind_topology(frame)
        cell = self._cell(frame)
        if cell is not None:
            self.bind_cell(cell[0] if cell.dim() == 3 else cell)
        return self

    def bind_topology(self, system: TensorDict | TensorFrame) -> None:
        """Build this topology's exclusion table, and check the declared net charge.

        Called **once per topology** — the connectivity analogue of
        :meth:`molix.schema.CellBound.bind_cell`, which is called once per
        cell. :meth:`_scaled_pairs` calls it lazily on the first nonbonded
        evaluation, so a single-batch score need no explicit call; a caller
        that reuses one ``Potentials`` over a **different** topology must
        call it again, because the bond / angle / torsion columns may not
        silently change under a bound table. Repeated calls are harmless:
        the table is rebuilt from scratch each time.

        **All or nothing.** Every check below runs before a single field of this
        container is written, so a bind that raises leaves the container exactly
        as it found it — still unbound if it was unbound, still on the previous
        topology if it was bound. That is not tidiness: the state a half-bind
        used to leave behind was *load-bearing*, because the lazy bind fires
        only when ``_exclusions`` is ``None``.
        A refused bind that had already installed the table therefore disarmed
        the very guard that would have refused the next call, and the batch the
        container had just rejected got a number back. It is
        :meth:`_check_composition`'s principle — a refused ForceField never
        leaves a half-built container behind — applied one stage later.

        The bind records a **shape-level** fingerprint of what it read — the
        atom count and the column counts of ``bonds`` / ``angles`` /
        ``propers``, Python ints off ``.shape``, no tensor content and no host
        sync — as ``_bound_topology``. Every later pair-field energy call
        compares it (:meth:`_scaled_pairs`) and raises rather than answering
        out of a stale table. That fingerprint is the **exclusion-table**
        identity, not an F29 cache: a later public :meth:`calc_energy` or
        another :meth:`bind_topology` still re-runs :meth:`_check_geometry`
        on the positions and cell it is handed. Its limit is the one that
        keeps two equal-shape topologies indistinguishable: a re-indexed
        molecule of the same column counts remains the caller's contract.
        Never silently rebinding is the point — a rebuilt table would be
        one more quietly wrong energy.

        F29 on this method is :meth:`_check_geometry` of **this** batch, with
        no identity / ``_warm`` / ``is_compiling()`` skip — including when
        first reached from :meth:`_scaled_pairs` on a compiled :meth:`energy`
        call. That lazy path is bind-once exclusion work, not a geometry
        voucher for later frames: after ``_exclusions`` is set, later
        :meth:`energy` calls compare the fingerprint by value and do not
        re-run F29. A bonded-only field never reaches :meth:`_scaled_pairs`,
        so :meth:`energy` never F29s on it. Bonded-only ``MD.run`` is in
        that compiled-energy residual (integrator-never-wraps plus the
        caller unwrapping before ``run``); this method is not ``bind_work``
        vouching the bound work batch.

        Compiling a **pair** field? Bind — or call public :meth:`calc_energy`
        — before entering the compiled :meth:`energy` region: the lazy call
        inside would otherwise trace this bind (and F29) as a graph break on
        the first step. That is host work moved out of the kernel, not an
        F29 skip inside it.

        The total-charge check lives here rather than in the force loop
        precisely because here it is affordable: eager, per graph and
        host-side, once per topology instead of once per step. Under a
        ``graphs.cell`` every bond is measured against half the cell's
        minimum perpendicular width (Å), and a molecule folded across a box
        face is refused instead of answered.

        Args:
            system: Batch carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers``).
                ``graphs.total_charge`` ``(B,)`` in e is checked when present,
                and so is the orientation of ``edges.pair_index``
                (:meth:`_check_pair_orientation`) — bind time is the one place
                that check is affordable, and a mis-oriented list is otherwise
                a silent zero rather than an error. ``graphs.cell`` ``(3, 3)``
                or ``(B, 3, 3)`` in Å, with ``bonds``, additionally puts the
                frame through :meth:`_check_bonds_unwrapped`.

        Warns:
            None. Net-charge vs ``graphs.total_charge`` is reported from
            :meth:`_coulomb_energy` on the q that electrostatics prices
            (EspalomaCharge / ByteFF: conserve ``sum q = Q_tot`` on the
            charges just assigned). The seed ForceField ``charge`` is a previous
            topology and is not summed here.

        Raises:
            ValueError: The batch's placed edge list holds
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
        # that can raise.
        frame = self._frame(system)
        p = self._pos(frame)
        n_atoms = int(p.shape[0])
        for label, index in (
            ("bonds", self._bond_index(frame)),
            ("angles", self._angle_index(frame)),
            ("propers", self._proper_index(frame)),
            ("impropers", self._improper_index(frame)),
        ):
            if index is not None and index.numel() > 0:
                hi = int(index.max().item())
                if hi >= n_atoms:
                    raise ValueError(
                        f"{label} index max {hi} >= n_atoms {n_atoms}; "
                        "connectivity was not rebased to this batch"
                    )
        if "pairs" in frame.keys():
            pair_index = frame.pairs.get("pair_index")
            if pair_index is not None and pair_index.numel() > 0:
                hi = int(pair_index.max().item())
                if hi >= n_atoms:
                    raise ValueError(f"edges.pair_index max {hi} >= n_atoms {n_atoms}")
        self._check_pair_orientation(frame)
        # Net-charge is checked on the q Coulomb prices (:meth:`_coulomb_energy`),
        # not the seed ``ForceField`` table. Class-I reuses one container;
        # pad made N static so a leftover seed of this N used to fire 0.1–1.6 e
        # false positives (GH200 2390967). EspalomaCharge / ByteFF conserve
        # ``sum q = Q_tot`` on the charges just assigned, not a previous
        # molecule's table (Wang et al., Digit. Discov. 2022; ByteFF BCC).
        # F29 of this geometry, every bind: no identity memo to clear. The
        # fingerprint is the exclusion-table identity stored below, not an
        # F29 voucher for a later rewrite of the same object.
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
        # The field's own per-atom tables, filtered to this topology's N once,
        # here in the eager bind. :meth:`_atom_params_for_n` serves the compiled
        # region from this instead of comparing ``int(value.shape[0])`` against
        # a traced ``n_atoms`` (one recompile per packed N) — and instead of
        # dropping them, which left a fixed-table field with no epsilon / sigma
        # / q at all inside the traced region.
        self._bound_atom_params = self._field_atom_params(p.shape[0])

    def _check_pair_orientation(self, frame: TensorFrame) -> None:
        """Refuse a placed list that :meth:`_placed_pairs` would read as all dead.

        :meth:`_placed_pairs` prices the ``source < target`` half of the
        bidirectional list :class:`molix.data.NeighborList` writes at its
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
                ``edges.pair_index`` ``(E, 2)``. No list, or a list of no rows,
                is nothing to check.

        Raises:
            ValueError: Some row runs ``source > target`` and none runs
                ``source < target``.
        """
        pairs = frame.pairs if "pairs" in frame.keys() else None
        pair_index = None if pairs is None else pairs.get("pair_index", None)
        if pair_index is None or pair_index.shape[0] == 0:
            return
        src, tgt = pair_index[:, 0], pair_index[:, 1]
        # One host sync, folded into a single boolean: "reversed rows exist and
        # forward ones do not".
        if bool((src > tgt).any() & ~(src < tgt).any()):
            raise ValueError(
                "this batch's edges.pair_index carries only source > target rows: "
                "Potentials prices the source < target half of a bidirectional list, so "
                "every pair term would silently cost zero on this one. "
                "molix.data.NeighborList(symmetry=False) produces exactly this "
                "upper-triangle-reversed half list — build it with the symmetry=True "
                "default (both orientations, one of them priced), or transpose the two "
                "columns before binding."
            )

    def _check_geometry(self, frame: TensorFrame, pos: torch.Tensor) -> tuple[int, ...]:
        """Run F29 on this batch's positions and cell, every call.

        Public :meth:`calc_energy`, :meth:`bind_topology` and :meth:`warmup`
        (via the bind) come through here. :meth:`energy` does not — it is the
        :class:`~molix.autograd.Derivative` seam and must not host-sync. There
        is no object-identity memo, no ``_warm`` flag, and no
        ``torch.compiler.is_compiling()`` hatch: a TensorDict whose coordinates
        were rewritten, a fresh batch of the same topology shape, and a
        TensorFrame compiled on first contact are all checked again. Object
        identity is not a physical invariant.

        Always computes :meth:`_topology_fingerprint` and always runs
        :meth:`_check_bonds_unwrapped`. The fingerprint is returned so
        :meth:`bind_topology` can store it as ``_bound_topology`` without
        building a second one. That comparison is the exclusion-table
        identity :meth:`_scaled_pairs` checks by value, not an F29 cache. A
        first unbound pair-field :meth:`energy` still reaches this method
        through :meth:`_scaled_pairs` → :meth:`bind_topology`; that is
        bind-once exclusion work, not a later-geometry voucher, and it has
        no skip hatch because the caller arrived through :meth:`energy`.

        Refusing a frame means reading a comparison back to Python, so this
        method cannot be *traced* into a graph. Skipping it under
        ``is_compiling()`` is the fail-open this seam forbids: a folded
        molecule priced through a compiled public :meth:`calc_energy` is the
        same finite, differentiable, wrong energy as through an eager one.
        Dynamo cannot trace the F29 ``argmax`` / Python raise; an unvouched
        compiled public :meth:`calc_energy` graph-breaks on that host sync
        and the eager re-run still runs the check (fail-closed). That is
        scoring, not the MD kernel. Compiled :meth:`energy` has no F29 host
        sync in the captured region unless a pair field's first unbound
        :meth:`_scaled_pairs` bind is still inside it (bind before compile).

        Args:
            frame: Typed view of the batch being scored, as
                :meth:`_check_bonds_unwrapped` reads it (``graphs.cell`` Å,
                the ``bonds`` namespace, ``atoms.batch``). Identity is **not**
                a key — :meth:`_frame`'s wrap memo is wrap identity for
                compiled MD ``op(work)``, not an F29 voucher.
            pos: Positions ``(N, 3)`` Å the energy would be priced at.

        Returns:
            The :meth:`_topology_fingerprint` of this frame
            (``(n_atoms, n_bonds, n_angles, n_propers)``).

        Raises:
            ValueError: Propagated from :meth:`_check_bonds_unwrapped` — some
                bond is longer than half the minimum perpendicular width of
                its graph's cell (F29, Å).
        """
        # No identity memo and no "are we tracing?" hatch — a miss that
        # skipped the check is a folded molecule priced as if it were whole.
        fingerprint = self._topology_fingerprint(frame, pos.shape[0])
        self._check_bonds_unwrapped(frame, pos)
        return fingerprint

    def _check_bonds_unwrapped(self, frame: TensorFrame, pos: torch.Tensor) -> None:
        """Refuse a periodic frame whose molecules are folded across a box face.

        This is the **one** place the F29 precondition of public
        :meth:`calc_energy`, :meth:`bind_topology` and :meth:`warmup` is
        spelled out and enforced. :meth:`energy` does not call it: that
        method is the :class:`~molix.autograd.Derivative` seam.

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
        always re-runs: the single ``argmax`` / ``float`` sync this costs is
        paid on every eager public scoring / bind, never inside compiled
        :meth:`energy` / the force path, which is why the check lives out
        here rather than in a term. A batch with no ``graphs.cell`` (an open
        boundary, nothing to wrap into), no bonds, or a degenerate lattice
        has no limit to exceed and is passed through untouched.

        Args:
            frame: The batch's typed view, optionally carrying
                ``graphs.cell`` ``(3, 3)`` or ``(B, 3, 3)`` in Å and the
                ``bonds`` namespace. ``atoms.batch`` picks each bond's cell out
                of a batched one.
            pos: Positions ``(N, 3)`` in Å the energy would be priced at. Read
                detached.

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

    def get_schema(self) -> ForceFieldSchemaView:
        """Metadata of the declared styles: parameter names, units, energy formulas."""
        return PotentialMeta.schema_for(self.forcefield.style_names(), units=self.forcefield.units)

    def _sum_energy(
        self,
        frame: TensorFrame,
        tables: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Classical term sum, or the mlip ``call_energy`` branch.

        Shared by public :meth:`calc_energy` (after F29 on the classical path)
        and :meth:`energy` (after the F13 guard). Not a new type, and not
        :meth:`molzoo.mace.potential.MacePotential._sum_energy` (that is a
        per-atom scatter × ``atoms.mask``). Does not run
        :meth:`_check_geometry`.

        When :attr:`mlip` is not ``None`` this returns
        :func:`~molpot.derivation.protocol.call_energy` of that module under
        ``ENERGY_KEY``. Otherwise it zeros a tensor of the batch's graph
        count and adds every :meth:`term_energies` value — starting from a
        tensor zero so an empty field does not collapse to Python ``0``.

        Args:
            frame: Typed view of the batch. Positions ``atoms.pos`` ``(N, 3)``
                Å size the classical zero; the mlip path hands the view
                through.
            tables: Optional predicted parameter tables keyed by namespace,
                forwarded to :meth:`term_energies`. ``None`` is the fixed
                ForceField.

        Returns:
            Energy ``(B,)`` over the ``B`` molecules, or ``()`` for a single
            molecule. Classical path: the ForceField's unit convention
            (``units="real"`` default: kcal/mol). mlip path: whatever
            :func:`~molpot.derivation.protocol.call_energy` wrote under
            ``ENERGY_KEY``.
        """
        if self.mlip is not None:
            # Which entry of the mlip evaluates the energy is ``call_energy``'s
            # policy alone — this branch must not reimplement it by probing the
            # mlip. The view shares the caller's storage, so whatever the mlip
            # writes lands on the caller's own batch.
            return call_energy(self.mlip, frame)[ENERGY_KEY]
        p = self._pos(frame)
        num_graphs = self._num_graphs(frame)
        total = torch.zeros((num_graphs,) if num_graphs > 1 else (), device=p.device, dtype=p.dtype)
        for e in self.term_energies(frame, tables).values():
            total = total + e
        return total

    def calc_energy(
        self,
        system: TensorDict | TensorFrame,
        tables: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Total energy: F29 on the classical path, then :meth:`_sum_energy`.

        The public scoring entry. A **tensor** still attached to the autograd
        graph; the Hessian path differentiates it directly. :meth:`energy` is
        the Derivative seam that writes this sum onto the batch — it does not
        call this method.

        **The mlip branch.** When :attr:`mlip` is not ``None`` (``def_pairstyle``
        was given an ``nn.Module`` instead of a classical style name) this
        method never runs F29 and never sums the classical term energies —
        :meth:`_sum_energy` hands the whole batch to
        :func:`molpot.derivation.protocol.call_energy`. What the batch is
        *carried in* decides nothing here: a ``TensorDict`` and its
        :class:`~molix.data.TensorFrame` view are one system (:meth:`_frame`).

        **Precondition — each molecule must arrive whole (F29).** A periodic
        frame's molecules must not be wrapped across a box face: the bonded
        terms price plain displacements while the pair terms price
        minimum-image ones, so a folded molecule returns a finite,
        differentiable, wrong energy. Enforced here, on the classical path
        (``mlip is None``), by :meth:`_check_geometry` **every call** —
        whatever the field declares, so a bonded-only periodic field is
        covered too. No identity memo, no ``_warm`` flag, no
        ``torch.compiler.is_compiling()`` hatch: warmup or bind of one batch
        is not a geometry voucher for the next, and compiling this method
        on an unvouched TensorFrame still raises. The mechanism and its
        limits are on :meth:`_check_bonds_unwrapped`.

        Args:
            system: Batch carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``), each molecule whole (see the precondition
                above). With any pair style declared it must also carry
                the ``pairs`` neighbour list a molix owner placed
                (:meth:`_placed_pairs`).

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
                style is declared and the batch carries no usable ``pairs``
                list; or ``pair:coulomb`` is evaluated under a ``graphs.cell``
                on a batch of more than one graph (one lattice gives one
                reciprocal sum, so each graph would otherwise be charged the
                whole box). See :meth:`term_energies` for each. Plus, on that
                path, the F29 refusal :meth:`_check_geometry` raises on a
                molecule wrapped across a box face — the one check that cannot
                wait for a bind, because a bonded-only field never performs
                one.
        """
        frame = self._frame(system)
        if self.mlip is None:
            self._check_geometry(frame, self._pos(frame))
        return self._sum_energy(frame, tables)

    def term_energies(
        self,
        system: TensorDict | TensorFrame,
        tables: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
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
            system: Batch carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces — plus, for any declared pair style, the
                ``pairs`` neighbour list built upstream in molix.

        Returns:
            Term name → energy ``(B,)`` (``()`` for a single molecule) in the
            ForceField's unit convention (``units="real"`` default: kcal/mol).

        Raises:
            ValueError: A declared style's parameter table is absent; a pair
                style is declared and the batch carries no usable placed
                neighbour list — no ``edges.pair_index``, or one with neither
                ``edges.shifts`` nor ``edges.pair_diff`` to price it by
                (:meth:`_placed_pairs`); or ``pair:coulomb`` is evaluated under
                a ``graphs.cell`` on a batch of more than one graph — one
                lattice gives one reciprocal sum, so each graph would be
                charged the whole box.
        """
        # This call's predicted tables, keyed by namespace. Empty is the fixed
        # force field: every term prices the table it was built with. A learned
        # field passes them here instead, because its tables are computed inside
        # the compiled region and cannot be owned by a module that outlives the
        # call (:meth:`~molpot.potentials.base.BasePotential.table`).
        called: Mapping[str, Mapping[str, torch.Tensor]] = tables or {}
        frame = self._frame(system)
        p = self._pos(frame)
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
                bond_called = self._term_tables(
                    called.get("bonds", {}), frame.get("bonds"), self.bond, ("k", "r0")
                )
                self._require_identity_table("k", n, bond_called, self.bond.k)
                self._require_identity_table("r0", n, bond_called, self.bond.r0)
                terms["bonds"] = self.bond(
                    pos=p,
                    bond_index=bond_index,
                    # ``None``, not ``arange(n)``: this container's tables carry one
                    # row per interaction (checked by ``_require_identity_table``
                    # above), so a type index would be the identity and the gather a
                    # copy whose backward is an atomic scatter — 77 % of device time
                    # on a production-shaped step (profile 2026-09-16).
                    bond_types=None,
                    term_batch=atom_batch[bond_index[0]] if batched else None,
                    num_graphs=num_graphs,
                    **bond_called,
                )
            else:
                terms["bonds"] = zero

        if "angle" in self._bonded_styles:
            self._require_bonded_params("angle", "angles", self.angle)
            angle_index = self._angle_index(frame)
            if self.angle is not None and angle_index is not None and angle_index.size(1) > 0:
                n = angle_index.size(1)
                angle_called = self._term_tables(
                    called.get("angles", {}),
                    frame.get("angles"),
                    self.angle,
                    ("k", "theta0"),
                )
                self._require_identity_table("k", n, angle_called, self.angle.k)
                self._require_identity_table("theta0", n, angle_called, self.angle.theta0)
                terms["angles"] = self.angle(
                    pos=p,
                    angle_index=angle_index,
                    # ``None``, not ``arange(n)``: this container's tables carry one
                    # row per interaction (checked by ``_require_identity_table``
                    # above), so a type index would be the identity and the gather a
                    # copy whose backward is an atomic scatter — 77 % of device time
                    # on a production-shaped step (profile 2026-09-16).
                    angle_types=None,
                    term_batch=atom_batch[angle_index[0]] if batched else None,
                    num_graphs=num_graphs,
                    **angle_called,
                )
            else:
                terms["angles"] = zero

        if "dihedral" in self._bonded_styles:
            self._require_bonded_params("dihedral", "propers", self.proper)
            proper_index = self._proper_index(frame)
            if self.proper is not None and proper_index is not None and proper_index.size(1) > 0:
                n = proper_index.size(1)
                # Backfill from the term only in the dialect the caller speaks.
                # ``ProperTorsionPeriodic`` stores the cosine form, so when the
                # caller hands in OPLS ``k1..k4`` for *this* topology, merging
                # the term's ``k`` / ``phase`` would mix the seed's spelling
                # *and* its term count into this batch's tables, and the
                # identity gather then refused a batch it had been given
                # correct parameters for.
                proper_tables = called.get("propers", {})
                proper_names = self._proper_table_names(proper_tables)
                proper_called = self._called_proper_tables(
                    self._term_tables(
                        proper_tables,
                        frame.get("propers"),
                        self.proper,
                        proper_names,
                    )
                )
                self._require_identity_table("k", n, proper_called, self.proper.k)
                self._require_identity_table("phase", n, proper_called, self.proper.phase)
                self._require_identity_table(
                    "periodicity", n, proper_called, self.proper.periodicity
                )
                terms["propers"] = self.proper(
                    pos=p,
                    proper_index=proper_index,
                    # ``None``, not ``arange(n)``: this container's tables carry one
                    # row per interaction (checked by ``_require_identity_table``
                    # above), so a type index would be the identity and the gather a
                    # copy whose backward is an atomic scatter — 77 % of device time
                    # on a production-shaped step (profile 2026-09-16).
                    proper_types=None,
                    term_batch=atom_batch[proper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                    **proper_called,
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
                improper_called = self._term_tables(
                    called.get("impropers", {}),
                    frame.get("impropers"),
                    self.improper,
                    ("k", "chi0", "phase"),
                )
                self._require_identity_table("k", n, improper_called, self.improper.k)
                if isinstance(self.improper, ImproperHarmonic):
                    self._require_identity_table("chi0", n, improper_called, self.improper.chi0)
                elif isinstance(self.improper, ImproperPeriodic):
                    self._require_identity_table("phase", n, improper_called, self.improper.phase)
                terms["impropers"] = self.improper(
                    pos=p,
                    improper_index=improper_index,
                    # ``None``, not ``arange(n)``: this container's tables carry one
                    # row per interaction (checked by ``_require_identity_table``
                    # above), so a type index would be the identity and the gather a
                    # copy whose backward is an atomic scatter — 77 % of device time
                    # on a production-shaped step (profile 2026-09-16).
                    improper_types=None,
                    term_batch=atom_batch[improper_index[0]] if batched else None,
                    num_graphs=num_graphs,
                    **improper_called,
                )
            else:
                terms["impropers"] = zero

        if self.lj is not None:
            terms["lj_repulsion"], terms["lj_dispersion"] = self._lj_energy(
                frame, p, called.get("atoms")
            )
        if self.coulomb_open is not None or self.coulomb_periodic is not None:
            terms["coulomb"] = self._coulomb_energy(frame, p, called.get("atoms"))
        return terms

    def contribution_energies(
        self, system: TensorDict | TensorFrame
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
            system: Batch carrying ``atoms.pos`` ``(N, 3)`` Å and the
                topology namespaces.

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
        for name, energy in self.term_energies(system).items():
            contribution = labels[name]
            previous = grouped.get(contribution)
            grouped[contribution] = energy if previous is None else previous + energy
        return grouped

    # ------------------------------------------------------------------
    # nonbonded
    # ------------------------------------------------------------------

    def _atom_params_for_n(
        self,
        called_atoms: Mapping[str, torch.Tensor] | None,
        n_atoms: int | None,
    ) -> dict[str, torch.Tensor]:
        """Per-atom tables for this call's ``N``, never a seed of a different size.

        ``{**field.params, **called}`` used to keep the seed ``epsilon`` /
        ``sigma`` when the predicted tables only overrode ``charge``. The next
        batch then gathered ``epsilon[pair_index]`` with ``N_new > N_seed``
        (CUDA vectorized gather OOB, profile 2026-09-13).
        """
        if torch.compiler.is_compiling():
            # Never a Python ``int(shape)`` against a traced ``n_atoms``: that
            # compares a tensor-derived length and recompiles once per packed N
            # (profile 2026-09-13). The bind already did the filtering for this
            # topology, so the traced region reads its result.
            merged = dict(self._bound_atom_params or {})
            merged.update(dict(called_atoms or {}))
            return merged
        merged = self._field_atom_params(n_atoms)
        merged.update(dict(called_atoms or {}))
        return merged

    def _field_atom_params(self, n_atoms: int | None) -> dict[str, torch.Tensor]:
        """The ForceField's own per-atom tables that fit ``n_atoms``.

        A table of another length belongs to another topology — gathering it at
        this batch's indices is the out-of-bounds read :meth:`_atom_params_for_n`
        exists to prevent — so it is dropped rather than handed on. Scalars fit
        every ``N``.
        """
        kept: dict[str, torch.Tensor] = {}
        for key, value in (self.forcefield.params("atoms") or {}).items():
            if n_atoms is None or value.ndim == 0 or int(value.shape[0]) == int(n_atoms):
                kept[key] = value
        return kept

    def _charges(
        self,
        called_atoms: Mapping[str, torch.Tensor] | None = None,
        *,
        n_atoms: int | None = None,
    ) -> torch.Tensor:
        """Partial charges ``(N,)`` of the declared ``pair:coulomb`` style.

        Raises:
            ValueError: ``atoms.charge`` is absent. A declared style with no table
                contributed a silent zero before (F2).
        """
        atoms = self._atom_params_for_n(called_atoms, n_atoms)
        if "charge" not in atoms:
            raise ValueError(
                "style 'pair:coulomb' is declared but its parameter table is absent: "
                "no charge (partial charges) under atoms"
            )
        return atoms["charge"]

    def _warn_net_charge_drift(self, frame: TensorFrame, q: torch.Tensor) -> None:
        """Report ``Σ_g q_live`` vs ``graphs.total_charge`` (eager only).

        EspalomaCharge solves EEM with a Lagrange row ``1ᵀq = Q_tot`` so the
        assigned charges sum to the declared total (~1e-8 e, choderalab
        espaloma-charge#32). ByteFF's BCC head likewise conserves molecular
        charge on the *predicted* q. This check is that post-assignment
        invariant, on the tensor Coulomb will use, with ghosts masked.
        """
        if torch.compiler.is_compiling():
            return
        declared = get_column(frame, ("graphs", "total_charge"))
        if declared is None or declared.numel() == 0:
            return
        atom_batch = self._atom_batch(frame, q.shape[0], q.device)
        if int(q.shape[0]) != int(atom_batch.shape[0]):
            return
        contrib = q.detach()
        ones_like_live = torch.ones_like(contrib)
        mask = get_column(frame, ("atoms", "mask"))
        if mask is not None and int(mask.shape[0]) == int(contrib.shape[0]):
            live_mask = mask.to(dtype=contrib.dtype)
            contrib = contrib * live_mask
            # Ghost rows are not atoms: they must not inflate the count the
            # rounding bound is built from any more than they enter ``Σ q``.
            ones_like_live = ones_like_live * live_mask
        summed = contrib.new_zeros(declared.shape[0]).index_add_(0, atom_batch, contrib)
        drift = (summed - declared.detach().to(summed)).abs()
        # The tolerance scales with the sum being checked. A fixed 1e-6 e was
        # sized on ≤32-atom molecules; OMol25 4m runs to 350 atoms, where fp32
        # accumulation alone costs a few 1e-6 e and the fixed threshold can no
        # longer tell arithmetic from a head that left its constraint. It fired
        # 33 times on one production-shaped profile batch, all arithmetic.
        unit_roundoff = float(torch.finfo(contrib.dtype).eps)
        abs_q = contrib.abs()
        abs_sum = abs_q.new_zeros(declared.shape[0]).index_add_(0, atom_batch, abs_q)
        live = abs_q.new_zeros(declared.shape[0]).index_add_(0, atom_batch, ones_like_live)
        floor = _TOTAL_CHARGE_TOL * unit_roundoff / _TOTAL_CHARGE_TOL_EPS
        tol = torch.clamp(
            _TOTAL_CHARGE_ROUNDOFF * unit_roundoff * live.sqrt() * abs_sum,
            min=floor,
        )
        graph = int(torch.argmax(drift - tol))
        if float(drift[graph]) > float(tol[graph]):
            logger.warning(
                f"graph {graph}: sum of live atoms.charge is {float(summed[graph]):.8g} e but "
                f"graphs.total_charge declares {float(declared[graph]):.8g} e "
                f"(drift {float(drift[graph]):.3g} e, threshold {float(tol[graph]):.3g} e). "
                "Under PBC an Ewald background term absorbs a non-neutral cell "
                "silently, so a drift that grows with training is a charge head "
                "leaving its constraint, not arithmetic."
            )

    def _lj_params(
        self,
        called_atoms: Mapping[str, torch.Tensor] | None = None,
        *,
        n_atoms: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom ``(epsilon, sigma)`` of the declared ``pair:lj/cut`` style.

        Raises:
            ValueError: ``atoms.epsilon`` or ``atoms.sigma`` is absent.
        """
        atoms = self._atom_params_for_n(called_atoms, n_atoms)
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
        """The batch's placed ``pairs`` list, priced at the **live** positions.

        ``molpot`` consumes the cutoff graph; it never enumerates one. Both
        owners live upstream in ``molix`` (CLAUDE.md dependency graph;
        ``.claude/notes/architecture.md`` §Layer roles):

        * :class:`molix.data.NeighborList` — the training path and the Edge
          Convention owner. Writes ``pair_index`` ``(E, 2)``, ``pair_diff``
          ``(E, 3)`` = ``pos[target] - pos[source]`` and ``pair_dist`` ``(E,)``,
          bidirectional at its ``symmetry=True`` default.
        * :class:`molix.md.NeighborList` — the MD path. Writes those three plus
          ``edges.shifts`` (each row's lattice remainder) at a fixed capacity
          whose dead rows are in-range self-loops.

        No in-tree producer writes ``edges.num_pairs``. The MD list keeps its
        live count as a Python attribute and marks its dead tail with
        ``source == target`` rows, which the ``source < target`` half-list rule
        below already kills (finding F22 in
        ``.claude/notes/composable-potentials.md``; publishing the count from
        molix is a molix follow-up). The ``row < num_pairs`` liveness leg is
        therefore **optional** — honoured when some producer does place the
        key, never required for a list to be priced.

        The list is returned at the capacity it arrived at and ``live`` is the
        mask of the rows that contribute, never a selection to apply:
        compacting it makes the output shape a function of the data, which is a
        graph break on the compiled force path, so the pair terms multiply by
        the mask instead. Liveness is ``source < target`` (one orientation of
        each bidirectional pair) ∧ — when the asking term declares a cutoff —
        ``0 < r < r_c`` ∧ — when the list declares one — ``row < num_pairs``,
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
        2. ``pair_diff`` — the lattice remainder is recoverable only as
           ``pair_diff - (pos[t] - pos[s])`` on **detached** tensors, i.e. a
           constant. Equal to branch 1 to the last bit while ``pos`` is the
           geometry the list was written at (``pair_diff`` is defined as that
           same sum), and energy/forces stay exact under a rigid shift of the
           whole batch — but the constant is **stale** for any other positions,
           and it carries no strain, which is why :meth:`energy` refuses a
           shift-less list under a strain variable (F13).

        A stored ``pair_dist`` is **not** a third branch. It is a number, not
        geometry: forwarded as the answer it does not depend on the ``pos``
        leaf the energy is differentiated against, so ``dE/dpos`` is
        structurally zero and the batch prices a finite energy with silently
        zero forces — the worst failure an MD path can have. It also carries
        whatever imaging its owner applied, which a second convention here can
        only disagree with. A list with neither ``shifts`` nor ``pair_diff`` is
        refused instead (see ``Raises``).

        Args:
            frame: The batch's typed view carrying the ``pairs`` namespace —
                the one every caller inside the evaluation already holds, so
                nothing is wrapped or branched on here, one hop inside the
                compiled region. A caller holding a plain batch wraps it with
                :func:`~molix.data.as_tensor_frame` first.
            pos: Live positions ``(N, 3)`` in Å the distances are priced at —
                the strained / integrated leaf, not necessarily ``atoms.pos``.
            cutoff: Real-space cutoff ``r_c`` in Å of the asking term, or
                ``None`` to keep every row of the list.

        Returns:
            ``(pair_index, pair_vectors, dist, live)`` at the list's capacity —
            ``(E, 2)`` endpoints as placed, ``(E, 3)`` separation vectors in Å
            pointing source → target (``pos[t] - pos[s] + shift`` under a
            cell), ``(E,)`` separations in Å which are exactly
            ``|pair_vectors|``, and a boolean ``(E,)``. The vectors stay on the
            autograd graph with the distances. ``E == 0`` (a list with no rows)
            is an ordinary empty list — no pair in range — and prices at zero;
            only an *absent* list raises.

        Raises:
            ValueError: The batch carries no ``edges.pair_index``, or an
                ``pair_index`` with neither ``shifts`` nor ``pair_diff`` to
                price it by — a stored ``pair_dist`` alone is refused, not
                forwarded. Enumerating a replacement here would answer with a
                different neighbour set than the model was trained or
                integrated against, so the message names the geometry a list
                must carry and the two molix owners instead.
        """
        pairs = frame.pairs if "pairs" in frame.keys() else None
        pair_index = None if pairs is None else pairs.get("pair_index", None)
        if pairs is None or pair_index is None:
            raise ValueError(
                "this batch carries no edges.pair_index, and molpot does not enumerate "
                "neighbour pairs — the cutoff graph is built upstream by "
                "molix.data.NeighborList (training batches: pair_index, pair_diff, "
                "pair_dist) or by molix.md.NeighborList (MD: the same at a fixed capacity, "
                "plus edges.shifts). Place a list on the batch before evaluating; a list "
                "with zero rows is accepted and prices at zero."
            )
        shifts = pairs.get("shifts", None)
        pair_diff = pairs.get("pair_diff", None)
        src, tgt = pair_index[:, 0], pair_index[:, 1]
        if shifts is not None:
            vectors = pos[tgt] - pos[src] + shifts
        elif pair_diff is not None:
            implied = pair_diff.detach() - (pos.detach()[tgt] - pos.detach()[src])
            vectors = pos[tgt] - pos[src] + implied
        else:
            raise ValueError(
                "edges.pair_index is placed but carries no geometry to price it by: "
                "edges.shifts or edges.pair_diff is required. A stored edges.pair_dist is "
                "not geometry — it is not a function of the live positions, so forwarding it "
                "prices a finite energy with structurally zero forces. "
                "molix.data.NeighborList writes pair_diff beside pair_dist; "
                "molix.md.NeighborList writes edges.shifts as well (the branch whose virial "
                "is complete). molpot never re-enumerates the pairs itself."
            )
        dist = (vectors * vectors).sum(-1).clamp_min(_SQ_DIST_CLAMP_MIN).sqrt()
        live = src < tgt
        num_pairs = pairs.get("num_pairs", None)
        if num_pairs is not None:
            rows = torch.arange(pair_index.shape[0], device=pair_index.device)
            live = live & (rows < num_pairs)
        if cutoff is not None:
            live = live & (dist < cutoff) & (dist > 0)
        return pair_index, vectors, dist, live

    def _scaled_pairs(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        *,
        cutoff: float | None,
        scale_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Half list ``(pair_index, pair_vectors, dist, scales, live)`` for one pair term.

        The list keeps the capacity :meth:`_placed_pairs` read it at — ``live``
        is the boolean mask of the rows that contribute, not a selection to
        apply. ``scales`` is the exclusion factor of every row including the
        dead ones (which read ``1.0``), so the term multiplies instead of
        slicing. There is no "no list" answer: a batch with zero rows prices at
        zero and a batch with no ``pairs`` namespace raises, because molpot
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
                (``"charge"`` for Coulomb, ``"lj"`` for Lennard-Jones).

        Returns:
            ``(pair_index, pair_vectors, dist, scales, live)`` — pairs
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
            self.bind_topology(frame)
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
        pair_index, vectors, dist, live = self._placed_pairs(frame, pos, cutoff=cutoff)
        scales = self._exclusions.lookup(pair_index, scale_kind, dtype=pos.dtype)
        return pair_index, vectors, dist, scales, live

    def _lj_energy(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        called_atoms: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(repulsion, dispersion)`` of the LJ style — zeros when no pair is in range."""
        if self.lj is None:
            raise RuntimeError(
                "_lj_energy called with no lj/cut style assembled; every caller reaches it "
                "under `if self.lj is not None`"
            )
        epsilon, sigma = self._lj_params(called_atoms, n_atoms=pos.shape[0])
        if not torch.compiler.is_compiling() and (
            int(epsilon.shape[0]) != int(pos.shape[0]) or int(sigma.shape[0]) != int(pos.shape[0])
        ):
            raise ValueError(
                f"LJ tables epsilon {tuple(epsilon.shape)} sigma {tuple(sigma.shape)} "
                f"!= n_atoms {int(pos.shape[0])}; a reused container must receive this "
                "batch's predicted epsilon/sigma, not the seed"
            )
        # Keep LJ on its own cutoff list. Reusing an unlimited Coulomb half-list
        # forced the exclusion lookup (and distance storage) onto all
        # intramolecular pairs even though LJ only needs ``r < cutoff``.
        pair_index, _vectors, dist, scales, live = self._scaled_pairs(
            frame, pos, cutoff=self.lj.cutoff, scale_kind="lj"
        )
        num_graphs = self._num_graphs(frame)
        atom_batch = self._atom_batch(frame, pos.shape[0], pos.device)
        return self.lj.energy(
            pair_index=pair_index,
            distances=dist,
            epsilon=epsilon,
            sigma=sigma,
            scales=scales,
            pair_batch=atom_batch[pair_index[:, 0]] if num_graphs > 1 else None,
            num_graphs=num_graphs,
            pair_mask=live,
        )

    def _coulomb_energy(
        self,
        frame: TensorFrame,
        pos: torch.Tensor,
        called_atoms: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q = self._charges(called_atoms, n_atoms=pos.shape[0])
        if q.shape[0] != pos.shape[0]:
            raise ValueError(f"charge length {q.shape[0]} != n_atoms {pos.shape[0]}")
        self._warn_net_charge_drift(frame, q)

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
            pair_index, _vectors, dist, scales, live = self._scaled_pairs(
                frame, pos, cutoff=cut, scale_kind="charge"
            )
            num_graphs = self._num_graphs(frame)
            atom_batch = self._atom_batch(frame, pos.shape[0], pos.device)
            return self.coulomb_open.energy(
                pair_index=pair_index,
                distances=dist,
                charges=q,
                scales=scales,
                pair_batch=atom_batch[pair_index[:, 0]] if num_graphs > 1 else None,
                num_graphs=num_graphs,
                pair_mask=live,
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
        pair_index, _vectors, dist, _, live = self._scaled_pairs(
            frame,
            pos,
            cutoff=self.coulomb_periodic.real_space_cutoff,
            scale_kind="charge",
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
            pair_index=pair_index,
            distances=dist,
            pair_mask=live,
            excluded_pairs=excluded_pairs,
            excluded_scales=exclusions.lookup(excluded_pairs, "charge", dtype=pos.dtype),
        )

    def _rebuilds_pairs(self) -> bool:
        """Whether a declared term rebuilds its pair vectors from the placed list.

        The three nonbonded terms reach the neighbour list through
        :meth:`_scaled_pairs`; the bonded ones read ``pos`` and their own index
        columns, and an mlip reads ``edges.pair_diff`` as handed to it. Only the
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
        on the hot path, inside :meth:`_sum_energy` and again inside each
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
                return graphs.batch_size[0]
            num_atoms = graphs.get("num_atoms", None)
            if num_atoms is not None and num_atoms.ndim >= 1:
                return num_atoms.shape[0]
        atom_batch = frame.atoms.get("batch", None) if "atoms" in frame.keys() else None
        if atom_batch is not None and atom_batch.numel():
            # ``max``, not ``[-1]``: a collate that interleaves graphs still
            # answers with the count rather than with the last row's id.
            return int(atom_batch.detach().max().to("cpu")) + 1
        return 1

    # --- topology readers ---

    def _frame(self, system: TensorDict | TensorFrame) -> TensorFrame:
        """This container's one accessor: the batch, as its typed view.

        Every public entry point calls this **once, eagerly**, and hands the
        view to everything below — the readers, the checks and the term
        evaluation — so a compiled region sees the same leaves a plain batch
        would give it and the wrap never appears inside a trace.

        A :class:`~molix.data.TensorFrame` is returned as-is (it already is
        the view). Any other :class:`~tensordict.TensorDictBase` hits
        ``_frame_view`` when the cached weakref is that same object, otherwise
        :func:`~molix.data.as_tensor_frame` (zero-copy leaves) and the pair
        is stored. That wrap identity is why compiled MD's ``op(work)`` does
        not break at ``TypedTensorDict.__setattr__`` on a reused work
        TensorDict. It is **not** an F29 validity voucher: handing a
        TensorFrame so this method does not wrap is not a substitute for
        :meth:`_check_geometry`. ``weakref`` is imported only to key this
        memo.

        Args:
            system: The batch — a ``TensorDict`` or a :class:`TensorFrame` view
                of one.

        Returns:
            The typed view (:func:`as_tensor_frame`).

        Raises:
            TypeError: *system* is not a batch or a view of one.
        """
        if isinstance(system, TensorFrame):
            return system
        if not isinstance(system, TensorDictBase):
            raise TypeError(
                "Potentials evaluates a post-collate batch (a TensorDict or its TensorFrame "
                f"view); got {type(system).__name__}"
            )
        cached = self._frame_view
        if cached is not None and cached[0]() is system:
            return cached[1]
        view = as_tensor_frame(system)
        # Weak key, strong value — deliberately asymmetric. The key must not
        # keep the batch alive (a dead reference misses instead of matching
        # a reused address); the value is the view, which holds the batch's
        # leaves, so the previous batch stays alive until the next distinct
        # one arrives. One extra batch per step in training, none in MD.
        self._frame_view = (weakref.ref(system), view)
        return view

    def _pos(self, frame: TensorFrame) -> torch.Tensor:
        if "atoms" in frame.keys():
            p = frame.atoms.get("pos", None)
            if p is not None:
                return p
        raise ValueError("calc_energy requires atoms.pos on the batch")

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
        :func:`~molix.data.collate.collate_molecules` writes
        ``bonds.bond_index`` (the schema's COO, whose shape is the anti-alias
        guard against ``pair_index``), while a caller assembling the namespace
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
    def _require_identity_table(
        name: str,
        n: int,
        called: Mapping[str, torch.Tensor],
        buffer: torch.Tensor,
    ) -> None:
        """Refuse ``arange(n)`` lookup into a seed table of a different length.

        Bonded terms index ``k`` with identity types. A reused container's
        seed ``k`` is the first batch's term count, so a longer topology
        gathers OOB (profile 2026-09-13 CUDA vectorized_gather).
        """
        if torch.compiler.is_compiling():
            return
        table = called.get(name, buffer)
        if table.shape[0] != n:
            raise ValueError(
                f"{name} length {table.shape[0]} != {n} terms; "
                "pass this batch's tables to energy(), do not reuse the seed"
            )

    @staticmethod
    def _identity_types(n: int, device: torch.device) -> torch.Tensor:
        """The identity type index.

        :meth:`term_energies` no longer builds one — it prices a
        per-interaction table directly (``*_types=None``) — but a caller
        holding a per-*type* table still needs to say which type each
        interaction is.
        """
        return torch.arange(n, device=device, dtype=torch.long)

    def __repr__(self) -> str:
        return f"Potentials(forcefield={self.forcefield!r})"
