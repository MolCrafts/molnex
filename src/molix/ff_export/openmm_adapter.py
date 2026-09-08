"""OpenMM force-spec adapter (pure Python; no live ``openmm`` import).

Emits JSON-friendly records matching OpenMM User Guide §19 force names:

* ``HarmonicBondForce`` — ``k`` kJ/mol/nm², ``r0`` nm
* ``HarmonicAngleForce`` — ``k`` kJ/mol/rad², ``theta0`` rad
* ``PeriodicTorsionForce`` — ``k`` kJ/mol, ``periodicity``, ``phase`` rad
* ``NonbondedForce`` — ``charge`` e, ``sigma`` nm, ``epsilon`` kJ/mol,
  plus the nonbonded *method* the model ran: ``nonbondedMethod``,
  ``cutoffDistance`` nm, ``ewaldErrorTolerance``, ``useSwitchingFunction``,
  ``switchingDistance`` nm and ``mixing`` — the Lennard-Jones combining rule,
  which is a name from a closed vocabulary rather than a number: either
  ``"lorentz-berthelot"`` or ``"geometric"``
  (:class:`~molpot.potentials.vdw.lj126.MixingRule`)

A single OpenMM ``NonbondedForce`` owns both nonbonded pair terms — the
Lennard-Jones 12-6 interaction and the Coulomb ``1/r`` one — together with how
they are truncated, which is why those six method fields ride on the same
record as the per-type parameters. ``nonbondedMethod`` is one of ``"PME"``
(particle-mesh Ewald: the periodic lattice sum, split into a real-space
remainder and an FFT-mesh reciprocal part), ``"CutoffNonPeriodic"`` (open
boundary, truncated at ``cutoffDistance``) or ``"NoCutoff"`` (open boundary,
every pair summed).

The combining rule is the one part of a Class-I model this backend cannot
always express. ``NonbondedForce`` combines per-atom ``ε`` / ``σ`` by
Lorentz–Berthelot and offers no alternative, so a ``"geometric"`` model — the
OPLS-AA rule — is emitted with ``case`` :attr:`TranslationCase.DECOMPOSE` and a
``mixing_note``: its parameters translate exactly, its *combination* does not,
and a consumer needs per-pair exceptions or a ``CustomNonbondedForce`` to build
the same model. See :meth:`OpenMMAdapter._nonbonded`.

Particle indices are **not** required for type-level IR bags; records carry
``type_index`` (and ``term_index`` for multi-term torsions). Topology binding
is a later consumer concern.

References:
    OpenMM User Guide §19 "Forces" — ``NonbondedForce`` for the method,
    cutoff and switching-function fields emitted here.
    Design note: ``.claude/notes/learnable-classical-ff.md`` (the
    ``learnable-classical-ff-08-ff-export`` spec it came from was completed and
    removed).
"""

from __future__ import annotations

from typing import Any

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.conventions import ANGSTROM_TO_NM, ConventionTable
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.force_spec import ForceSpec
from molpot import MixingRule
from molpot.ir import PotentialIR

__all__ = ["OpenMMAdapter"]

#: Why a Lennard-Jones bag with no declared combining rule cannot be exported.
#: Stated identically by :mod:`molix.ff_export.lammps_adapter`: it is one policy
#: — the rule is model identity, so no backend supplies one the model did not
#: declare — and the evaluation side refuses the same record.
_NO_MIXING_RULE = (
    "lj parameters without a declared mixing rule — a rule is model identity; "
    "declare it on pair:lj/cut"
)


class OpenMMAdapter(BackendAdapter):
    """Translate Class-I :class:`~molpot.ir.PotentialIR` into OpenMM force-spec records.

    Does **not** import OpenMM. Unit tests pin hard-coded numeric goldens: a
    bond force constant ``k`` of 100 kcal/mol/Å² comes out 41840 kJ/mol/nm²,
    and an AMBER torsion barrier ``Vn`` of 2 kcal/mol (which the IR carries as
    the half-barrier ``k`` = 1) comes out ``k`` = 4.184 kJ/mol.

    Args:
        conventions: Unit / form matrix to translate through. Defaults to
            :meth:`~molix.ff_export.conventions.ConventionTable.default_openmm`.
        ewald_error_tolerance: Dimensionless target error of OpenMM's
            particle-mesh Ewald (PME) sum — the relative accuracy from which
            OpenMM derives its own splitting parameter and mesh size. Reported
            on the ``NonbondedForce`` record whichever method the model ran,
            since the field is part of that record's schema. OpenMM's own
            default is ``5e-4``, tightened here so the exported system
            reproduces the molnex reciprocal-space term rather than merely a
            converged-looking one.
    """

    name = "openmm"

    def __init__(
        self,
        conventions: ConventionTable | None = None,
        ewald_error_tolerance: float = 1e-6,
    ) -> None:
        if conventions is None:
            conventions = ConventionTable.default_openmm()
        self.conventions = conventions
        self.ewald_error_tolerance = ewald_error_tolerance

    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Build an OpenMM-oriented :class:`ForceSpec` from ``ir``.

        Args:
            ir: Class-I bags in kcal/mol, Å, e, rad.
            meta: Optional metadata merged into the force-spec.

        Returns:
            Force-spec with OpenMM units (kJ/mol, nm, e, rad).

        Raises:
            UnsupportedTermError: For bags marked
                :attr:`TranslationCase.UNSUPPORTED` (e.g. harmonic improper).
        """
        forces: list[dict[str, Any]] = []

        if ir.bonds is not None:
            forces.append(self._bonds(ir))
        if ir.angles is not None:
            forces.append(self._angles(ir))
        if ir.propers is not None:
            forces.append(self._propers(ir))
        if ir.impropers_periodic is not None:
            forces.append(self._impropers_periodic(ir))
        if ir.impropers_harmonic is not None:
            row = self.conventions.row("improper_harmonic")
            raise UnsupportedTermError(
                "improper_harmonic",
                row.notes or "no OpenMM built-in harmonic improper force",
                case=row.case,
            )
        if ir.lj is not None or ir.charges is not None:
            forces.append(self._nonbonded(ir))

        metadata: dict[str, Any] = {
            "unit_system_source": ir.unit_system,
            "unit_system_target": "openmm_standard",
        }
        if meta:
            metadata.update(meta)

        return ForceSpec(
            backend=self.name,
            forces=forces,
            scaling=self.scaling_dict(ir.scaling),
            metadata=metadata,
        )

    def _bonds(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.bonds is not None
        row = self.conventions.row("bond_harmonic")
        k = self.as_floats(ir.bonds.k)
        r0 = self.as_floats(ir.bonds.r0)
        params = [
            {
                "type_index": i,
                "k": row.scale("k", ki),
                "r0": row.scale("r0", ri),
            }
            for i, (ki, ri) in enumerate(zip(k, r0, strict=True))
        ]
        return {
            "type": "HarmonicBondForce",
            "case": row.case.value,
            "parameters": params,
        }

    def _angles(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.angles is not None
        row = self.conventions.row("angle_harmonic")
        k = self.as_floats(ir.angles.k)
        theta0 = self.as_floats(ir.angles.theta0)
        params = [
            {
                "type_index": i,
                "k": row.scale("k", ki),
                "theta0": ti,
            }
            for i, (ki, ti) in enumerate(zip(k, theta0, strict=True))
        ]
        return {
            "type": "HarmonicAngleForce",
            "case": row.case.value,
            "parameters": params,
        }

    def _propers(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.propers is not None
        bag = ir.propers
        n_types, n_terms = bag.k.shape
        periodicity = self.as_ints(bag.periodicity)
        row = self.conventions.row("proper_periodic")
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"proper torsion idivf[{t}] must be non-zero")
            for m in range(n_terms):
                k_ir = float(bag.k[t, m].item()) / idivf
                params.append(
                    {
                        "type_index": t,
                        "term_index": m,
                        "periodicity": periodicity[m],
                        "phase": float(bag.phase[t, m].item()),
                        "k": row.scale("k", k_ir),
                    }
                )
        case = TranslationCase.DECOMPOSE if n_terms > 1 else TranslationCase.FORM_REPARAMETERIZE
        return {
            "type": "PeriodicTorsionForce",
            "case": case.value,
            "parameters": params,
        }

    def _impropers_periodic(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.impropers_periodic is not None
        bag = ir.impropers_periodic
        n_types, n_terms = bag.k.shape
        periodicity = self.as_ints(bag.periodicity)
        row = self.conventions.row("improper_periodic")
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"improper periodic idivf[{t}] must be non-zero")
            for m in range(n_terms):
                k_ir = float(bag.k[t, m].item()) / idivf
                params.append(
                    {
                        "type_index": t,
                        "term_index": m,
                        "periodicity": periodicity[m],
                        "phase": float(bag.phase[t, m].item()),
                        "k": row.scale("k", k_ir),
                    }
                )
        case = TranslationCase.DECOMPOSE if n_terms > 1 else TranslationCase.FORM_REPARAMETERIZE
        return {
            "type": "PeriodicTorsionForce",
            "role": "improper",
            "case": case.value,
            "parameters": params,
            "index_convention": "molrs_center_first",
        }

    def _nonbonded(self, ir: PotentialIR) -> dict[str, Any]:
        """Emit ``NonbondedForce`` — the LJ / charge parameters and the method.

        OpenMM ties the Lennard-Jones truncation and the real-space half of the
        Coulomb sum to a *single* ``cutoffDistance``, while the IR records the
        two radii separately. Splitting them is not physics: a converged Ewald
        sum is independent of where its real-space half is cut, because the
        reciprocal-space half makes up exactly what the real-space one drops.
        Exporting the LJ cutoff as ``cutoffDistance`` and letting OpenMM
        re-derive its own splitting parameter from
        :attr:`ewald_error_tolerance` is therefore exact, not a
        reparameterization — only when no LJ bag names a radius does the record
        fall back to a Coulomb one.

        The method word weighs **both** pair terms, in this order: ``"PME"``
        when the Coulomb half was a lattice sum
        (:attr:`~molpot.ir.NonbondedMethodRecord.is_lattice_sum`),
        ``"CutoffNonPeriodic"`` when *either* term declared a radius — an
        open-boundary ``coulomb_cutoff`` or a ``lj_cutoff`` — and ``"NoCutoff"``
        only when neither did. Reading the word off the Coulomb half alone
        would send a charge-free, Lennard-Jones-only model to ``"NoCutoff"``,
        under which OpenMM truncates nothing and ignores ``cutoffDistance``:
        the ``lj_cutoff`` the model declared would be exported and then
        silently not applied, which is a different model, not a rounding.

        ``cutoffDistance`` is then the radius the chosen word truncates at: the
        Lennard-Jones one whenever there is a Lennard-Jones bag carrying it,
        the Ewald real-space radius under ``"PME"`` without such a bag, and the
        open-boundary ``coulomb_cutoff`` otherwise.

        ``mixing`` states the model's Lennard-Jones **combining rule** — how one
        ``ε`` (kcal/mol) and one ``σ`` (Å) per atom type become one of each per
        *pair* — as a :class:`~molpot.potentials.vdw.lj126.MixingRule` value. It
        is present exactly when the model has a Lennard-Jones bag to mix: a
        charges-only model gets no key rather than a rule it never declared, and
        a bag whose record names no rule at all is refused (see Raises) rather
        than exported under a guessed one. A single
        ``NonbondedForce`` has no rule to choose: OpenMM mixes Lorentz–Berthelot
        and nothing else. A ``"geometric"`` model — OPLS-AA, which is geometric
        in ``σ`` too — therefore does not translate into one force object at
        all. It needs an explicit exception per pair, or a
        ``CustomNonbondedForce`` with the rule written into its energy
        expression, so the record's ``case`` becomes
        :attr:`TranslationCase.DECOMPOSE` and a ``mixing_note`` says which. The
        record still carries the per-type ``ε`` / ``σ`` unchanged; what changes
        is that a consumer may not build one plain ``NonbondedForce`` from them
        and call it the same model. A *declared* geometric rule does not raise,
        because the parameters *are* faithfully translated — only their
        combination is not.

        Args:
            ir: The IR being translated; at least one of ``lj`` / ``charges``
                is set.

        Returns:
            The ``NonbondedForce`` record. ``epsilon`` is kJ/mol, ``sigma`` nm
            and ``charges`` e. ``cutoffDistance`` and ``switchingDistance`` are
            in nm (the IR carries Å); the first is ``None`` under
            ``"NoCutoff"``, which truncates nothing.
            ``useSwitchingFunction`` is ``True`` exactly when the model declared
            a Lennard-Jones switching window (``lj_switch_start``), and
            ``switchingDistance`` is that window's inner radius — OpenMM then
            multiplies the Lennard-Jones interaction by a smooth taper running
            from ``1`` at ``switchingDistance`` to ``0`` at ``cutoffDistance``,
            instead of cutting it off abruptly. With no window declared the two
            come out ``False`` and ``None``. ``mixing`` is present exactly for a
            model with a Lennard-Jones bag; ``mixing_note`` only when that rule
            is one ``NonbondedForce`` cannot express.

        Raises:
            UnsupportedTermError: The model has Lennard-Jones parameters but its
                record names no combining rule. The rule is model identity, so
                no backend may supply one it was not given — the same policy
                :meth:`Potentials._assemble_pair_potentials
                <molpot.forcefield.potentials.Potentials._assemble_pair_potentials>`
                applies on the evaluation side, where ``MixingRule(None)``
                raises.
        """
        row_lj = self.conventions.row("lj")
        row_q = self.conventions.row("charge")
        params: list[dict[str, Any]] = []
        charges: list[float] = []

        if ir.lj is not None:
            eps = self.as_floats(ir.lj.epsilon)
            sig = self.as_floats(ir.lj.sigma)
            for i, (e, s) in enumerate(zip(eps, sig, strict=True)):
                params.append(
                    {
                        "type_index": i,
                        "epsilon": row_lj.scale("epsilon", e),
                        "sigma": row_lj.scale("sigma", s),
                    }
                )
        if ir.charges is not None:
            charges = self.as_floats(ir.charges.q)

        method = ir.nonbonded_method
        if method.is_lattice_sum:
            nonbonded_method = "PME"
        elif method.coulomb_cutoff is not None or method.lj_cutoff is not None:
            nonbonded_method = "CutoffNonPeriodic"
        else:
            nonbonded_method = "NoCutoff"

        if ir.lj is not None and method.lj_cutoff is not None:
            cutoff_A = method.lj_cutoff
        elif method.is_lattice_sum:
            cutoff_A = method.real_space_cutoff
        else:
            cutoff_A = method.coulomb_cutoff
        if nonbonded_method == "NoCutoff" or cutoff_A is None:
            cutoff_nm = None
        else:
            cutoff_nm = cutoff_A * ANGSTROM_TO_NM
        switch_A = method.lj_switch_start

        if ir.lj is None:
            # No Lennard-Jones bag, nothing to combine: the record states no
            # rule at all rather than one the model never used. A charges-only
            # model may still carry ``lj_mixing`` on its method record (the
            # ForceField declared the style, the IR has no bag), and reporting
            # that here would describe mixing that never happens.
            mixing = None
        elif method.lj_mixing is None:
            raise UnsupportedTermError("lj", _NO_MIXING_RULE)
        else:
            mixing = method.lj_mixing
        record: dict[str, Any] = {
            "type": "NonbondedForce",
            "case": row_lj.case.value,
            "parameters": params,
            "charges": charges,
            "charge_case": row_q.case.value,
            "nonbondedMethod": nonbonded_method,
            "cutoffDistance": cutoff_nm,
            "ewaldErrorTolerance": self.ewald_error_tolerance,
            "useSwitchingFunction": switch_A is not None,
            "switchingDistance": None if switch_A is None else switch_A * ANGSTROM_TO_NM,
            **({} if mixing is None else {"mixing": mixing}),
        }
        if mixing == MixingRule.GEOMETRIC:
            record["case"] = TranslationCase.DECOMPOSE.value
            record["mixing_note"] = (
                "OpenMM NonbondedForce mixes Lorentz-Berthelot; geometric sigma "
                "needs per-pair exceptions or a CustomNonbondedForce"
            )
        return record
