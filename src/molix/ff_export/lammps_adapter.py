"""LAMMPS force-spec adapter (pure Python; no live ``lmp`` import).

Translates Class-I :class:`~molpot.ir.PotentialIR` into LAMMPS ``units real``
coeff records. Energy/length are already kcal/mol and Å in Class-I canonical
units; the load-bearing reparameterizations are:

* harmonic bond/angle: molnex ``E = ½ k x²`` → LAMMPS ``E = K x²`` with
  ``K = k/2``; angle ``theta0`` converted to degrees.
* proper torsions: multi-term cosine bags; when ``n∈{1..4}`` with OPLS phases
  they are also emitted as a compact ``dihedral_style opls`` ``K1..K4`` group.
* nonbonded terms: the pair style is chosen from the *method* the model ran
  (:class:`~molpot.ir.NonbondedMethodRecord`), not from the bags alone — a
  switching window selects ``lj/mdf`` over ``lj/cut`` and a lattice solver
  selects ``coul/long`` plus a ``kspace_style`` over ``coul/cut``. molnex's
  quintic taper ``1 − x³(10 − 15x + 6x²)`` and the ``lj/mdf`` taper
  ``(1 − x)³(1 + 3x + 6x²)`` with ``x = (r − r_m)/(r_cut − r_m)`` expand to the
  same polynomial ``1 − 10x³ + 15x⁴ − 6x⁵``, so the window transfers exactly.
* the Lennard-Jones **combining rule** — how one ``ε`` (kcal/mol) and one ``σ``
  (Å) per atom type become one of each per *pair* — travels on the same ``pair``
  record, as ``mixing``. It is not a reparameterization but model identity:
  LAMMPS would otherwise apply its own default rule, which is geometric in both
  ``ε`` and ``σ`` and so is not molnex's Lorentz–Berthelot default. Carrying it
  is what lets :class:`~molix.ff_export.LammpsWriter` render explicit
  ``pair_coeff i j`` lines mixed by the model's own rule instead of leaving the
  cross terms to the deck.

Every number is rendered with ``{:.7g}``: a LAMMPS input line is parsed as
text, so the record carries the same digits the deck will.

References:
    LAMMPS docs: bond_style harmonic, angle_style harmonic, dihedral_style
    opls, pair_style lj/mdf, pair_style lj/cut/coul/long, kspace_style pppm,
    special_bonds
    Spec: ``.claude/notes/learnable-classical-ff.md`` (peer of OpenMMAdapter);
    composable-a0-05-export (periodic pair styles)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.force_spec import ForceSpec
from molpot.ir import (
    NonbondedMethodRecord,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
)

__all__ = ["LammpsAdapter"]

#: Why a Lennard-Jones bag with no declared combining rule cannot be exported.
#: The evaluation side refuses the same record (``MixingRule(None)`` raises in
#: ``Potentials._assemble_pair_potentials``), so neither half of molnex ever
#: supplies a rule the model did not state.
_NO_MIXING_RULE = (
    "lj parameters without a declared mixing rule — a rule is model identity; "
    "declare it on pair:lj/cut"
)


#: How close a phase must be to 0 or π to read as cvff's sign, rad — the same
#: tolerance the OPLS-manifold check uses for its phases.
_PHASE_TOL = 1e-6


def _resolved_smearing(smearing: float | None) -> float:
    """The Ewald smearing σ of a record whose Coulomb term ran a lattice sum.

    :meth:`~molpot.forcefield.ForceField._nonbonded_method_record` fills
    ``smearing`` together with ``coulomb_calculator`` — both are ``None``
    exactly when no ``pair:coulomb`` style is declared — so a lattice solver
    with no width means the resolver and the export disagree about which model
    this is. Emitting ``kspace_modify gewald`` from a guessed width would hand
    LAMMPS a different decomposition under a line that claims to state molnex's.

    Args:
        smearing: The record's ``smearing`` field, in Å.

    Returns:
        ``smearing``.

    Raises:
        ValueError: ``smearing`` is ``None`` or not positive.
    """
    if smearing is None or smearing <= 0.0:
        raise ValueError(
            "a lattice Coulomb solver was exported with smearing="
            f"{smearing!r}; the Ewald splitting width is model identity and "
            "kspace_modify gewald cannot be written without it"
        )
    return smearing


class LammpsAdapter(BackendAdapter):
    """Translate Class-I IR into LAMMPS ``units real`` force-spec records.

    Does **not** invoke LAMMPS. Unit tests pin hard-coded numeric goldens
    (bond ``k = 100 kcal/mol/Å²`` → LAMMPS ``K = 50 kcal/mol/Å²``, the ``½k``
    → ``K`` halving; OPLS phases in rad → ``dihedral_coeff`` in kcal/mol).

    Args:
        kspace_accuracy: Relative RMS force accuracy passed to
            ``kspace_style pppm`` when the model ran a lattice Coulomb sum.
            Dimensionless; LAMMPS's own default is ``1e-4``, tightened here so
            the exported deck reproduces the molnex reciprocal-space term
            rather than merely a converged-looking one.
    """

    name = "lammps"

    def __init__(self, kspace_accuracy: float = 1e-6) -> None:
        self.kspace_accuracy = kspace_accuracy

    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Build a LAMMPS-oriented :class:`ForceSpec` from ``ir``.

        Args:
            ir: Class-I bags in kcal/mol, Å, e, rad.
            meta: Optional metadata merged into the force-spec.

        Returns:
            Force-spec with LAMMPS ``units real`` coeff conventions.

        Raises:
            UnsupportedTermError: When the nonbonded method the model ran has
                no LAMMPS pair style (see :meth:`_pair`).
        """
        forces: list[dict[str, Any]] = []

        if ir.bonds is not None:
            forces.append(self._bonds(ir))
        if ir.angles is not None:
            forces.append(self._angles(ir))
        if ir.propers is not None:
            forces.extend(self._propers(ir))
        if ir.impropers_harmonic is not None:
            forces.append(self._impropers_harmonic(ir))
        if ir.impropers_periodic is not None:
            forces.append(self._impropers_periodic(ir))
        if ir.lj is not None or ir.charges is not None:
            forces.extend(self._pair(ir))
        if ir.scaling is not None:
            forces.append(self._exclusions(ir.scaling))
        if ir.charges is not None:
            forces.append(self._charges(ir))

        metadata: dict[str, Any] = {
            "unit_system_source": ir.unit_system,
            "unit_system_target": "lammps_real",
            "units": "real",
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
        k = self.as_floats(ir.bonds.k)
        r0 = self.as_floats(ir.bonds.r0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "r0": ri,
            }
            for i, (ki, ri) in enumerate(zip(k, r0, strict=True))
        ]
        return {
            "type": "bond_style harmonic",
            "style": "harmonic",
            "category": "bond",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _angles(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.angles is not None
        k = self.as_floats(ir.angles.k)
        theta0 = self.as_floats(ir.angles.theta0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "theta0_deg": math.degrees(ti),
            }
            for i, (ki, ti) in enumerate(zip(k, theta0, strict=True))
        ]
        return {
            "type": "angle_style harmonic",
            "style": "harmonic",
            "category": "angle",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _propers(self, ir: PotentialIR) -> list[dict[str, Any]]:
        assert ir.propers is not None
        bag = ir.propers
        if self._is_opls_manifold(bag):
            return [self._propers_opls(bag)]
        return [self._propers_fourier(bag)]

    def _propers_opls(self, bag: ProperTorsionBag) -> dict[str, Any]:
        n_types = bag.k.shape[0]
        periods = self.as_int_rows(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            K = [0.0, 0.0, 0.0, 0.0]
            for m, n_per in enumerate(periods[t]):
                k_tm = float(bag.k[t, m].item())
                if n_per in (1, 2, 3, 4):
                    K[n_per - 1] += 2.0 * k_tm
            params.append(
                {
                    "type_index": t + 1,
                    "K1": K[0],
                    "K2": K[1],
                    "K3": K[2],
                    "K4": K[3],
                }
            )
        return {
            "type": "dihedral_style opls",
            "style": "opls",
            "category": "dihedral",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _propers_fourier(self, bag: ProperTorsionBag) -> dict[str, Any]:
        n_types, n_terms = bag.k.shape
        periods = self.as_int_rows(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            terms = [
                {
                    "periodicity": periods[t][m],
                    "k": float(bag.k[t, m].item()),
                    "phase": float(bag.phase[t, m].item()),
                }
                for m in range(n_terms)
            ]
            params.append({"type_index": t + 1, "terms": terms})
        return {
            "type": "dihedral_style fourier",
            "style": "fourier",
            "category": "dihedral",
            "case": TranslationCase.DECOMPOSE.value,
            "parameters": params,
        }

    @staticmethod
    def _is_opls_manifold(bag: ProperTorsionBag) -> bool:
        """True when propers are exactly n=1..4 with OPLS phases (0, π, 0, π)."""
        periods = bag.periodicity.detach().long().tolist()
        # Every type must carry the same 1..4 quartet; one row that does not is
        # a general torsion and the whole bag exports as `fourier`.
        if any(row != [1, 2, 3, 4] for row in periods):
            return False
        expected = torch.tensor(
            [0.0, math.pi, 0.0, math.pi],
            device=bag.phase.device,
            dtype=bag.phase.dtype,
        )
        delta = torch.remainder(bag.phase - expected.unsqueeze(0) + math.pi, 2 * math.pi) - math.pi
        return bool(torch.all(torch.abs(delta) < 1e-6))

    def _impropers_harmonic(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.impropers_harmonic is not None
        bag = ir.impropers_harmonic
        k = self.as_floats(bag.k)
        chi0 = self.as_floats(bag.chi0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "chi0_deg": math.degrees(ci),
            }
            for i, (ki, ci) in enumerate(zip(k, chi0, strict=True))
        ]
        return {
            "type": "improper_style harmonic",
            "style": "harmonic",
            "category": "improper",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _impropers_periodic(self, ir: PotentialIR) -> dict[str, Any]:
        """Periodic impropers as LAMMPS ``improper_style cvff``.

        cvff is ``K[1 + d cos(nχ)]``: one term, with a sign ``d = ±1`` where the IR
        has a phase. So ``k[1 + cos(nχ − γ)]`` maps exactly when a type has one
        non-zero term and ``γ ∈ {0, π}`` (``d = +1`` / ``−1``); any other row has no
        cvff spelling and is refused — it used to be recorded as cvff regardless.

        Raises:
            UnsupportedTermError: A type with more than one non-zero term, or a
                phase other than 0 or π.
        """
        assert ir.impropers_periodic is not None
        bag = ir.impropers_periodic
        n_types, n_terms = bag.k.shape
        periods = self.as_int_rows(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            live = [m for m in range(n_terms) if float(bag.k[t, m].item()) != 0.0]
            if len(live) > 1:
                raise UnsupportedTermError(
                    "improper_periodic",
                    f"type {t + 1} has {len(live)} non-zero terms; cvff holds one",
                )
            if not live:
                params.append({"type_index": t + 1, "K": 0.0, "d": 1, "n": periods[t][0]})
                continue
            m = live[0]
            phase = math.remainder(float(bag.phase[t, m].item()), 2.0 * math.pi)
            if abs(phase) < _PHASE_TOL:
                sign = 1
            elif abs(abs(phase) - math.pi) < _PHASE_TOL:
                sign = -1
            else:
                raise UnsupportedTermError(
                    "improper_periodic",
                    f"type {t + 1} has phase {phase:.6g} rad; cvff takes only 0 or π",
                )
            params.append(
                {
                    "type_index": t + 1,
                    "K": float(bag.k[t, m].item()),
                    "d": sign,
                    "n": periods[t][m],
                }
            )
        return {
            "type": "improper_style cvff",
            "style": "cvff",
            "category": "improper",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _pair(self, ir: PotentialIR) -> list[dict[str, Any]]:
        """Emit the nonbonded records the model's method spells in LAMMPS.

        Four steps, each its own method because each answers a different
        question: which Lennard-Jones sub-style the window and cutoff ask for
        (:meth:`_lj_sub_style`), which Coulomb sub-style the solver asks for
        (:meth:`_coulomb_sub_style`), how the two combine into the single
        ``pair_style`` line LAMMPS takes (:meth:`_fuse_pair_style`), and what
        reciprocal-space solver has to follow when the Coulomb half was summed
        over the lattice (:meth:`_kspace_records`). The style is a function of
        the *method* (:class:`~molpot.ir.NonbondedMethodRecord`) throughout, not
        of the bags — the bags only fill the coefficient table, which is
        :meth:`_lj_parameters`.

        Args:
            ir: The IR being translated; at least one of ``lj`` / ``charges``
                is set.

        Returns:
            The ``pair`` record — ``sub_styles`` carries the style words and
            their radii in Å as floats, so the writer never re-parses the
            rendered ``type`` line, and ``mixing`` the model's combining rule
            (a :class:`~molpot.potentials.vdw.lj126.MixingRule` value), which
            :meth:`~molix.ff_export.LammpsWriter.write_pair_block` mixes the
            explicit ``pair_coeff i j`` lines with. It is present exactly when
            the model has a Lennard-Jones bag: a charges-only model has nothing
            to mix and gets no key, rather than a default rule it never
            declared. Followed by the ``kspace`` and ``kspace_modify`` records
            when a lattice solver ran.

        Raises:
            UnsupportedTermError: Propagated from the two sub-style steps — a
                Lennard-Jones bag with no cutoff radius, or charges with
                neither a lattice solver nor a Coulomb cutoff — or raised here
                for a Lennard-Jones bag whose record names no combining rule.
                The rule is model identity, so a deck cannot be written by
                guessing one; this is the same policy
                :meth:`Potentials._assemble_pair_potentials
                <molpot.forcefield.potentials.Potentials._assemble_pair_potentials>`
                applies on the evaluation side, where ``MixingRule(None)``
                raises.
            ValueError: Propagated from :meth:`_kspace_records` when a lattice
                solver ran with no ``smearing`` on the record.
        """
        method = ir.nonbonded_method
        lj = self._lj_sub_style(ir, method)
        coulomb = self._coulomb_sub_style(ir, method)
        line, style, sub_styles = self._fuse_pair_style(lj, coulomb)
        case = (
            TranslationCase.DECOMPOSE
            if style == "hybrid/overlay"
            else TranslationCase.DIRECT_UNIT_SCALE
        )
        if ir.lj is not None and method.lj_mixing is None:
            raise UnsupportedTermError("lj", _NO_MIXING_RULE)
        pair = {
            "type": f"pair_style {line}",
            "style": style,
            "category": "pair",
            "case": case.value,
            "sub_styles": sub_styles,
            **({} if ir.lj is None else {"mixing": method.lj_mixing}),
            "parameters": self._lj_parameters(ir),
        }
        lattice = coulomb is not None and coulomb["style"] == "coul/long"
        return [pair, *(self._kspace_records(method) if lattice else ())]

    def _lj_parameters(self, ir: PotentialIR) -> list[dict[str, Any]]:
        """The per-atom Lennard-Jones coefficient table of the ``pair`` record.

        Kept apart from the style steps because it answers a different
        question: they decide which ``pair_style`` line is written, this decides
        what goes on the ``pair_coeff`` lines under it.

        The rows are **per atom**, in atom order, because that is what a Class-I
        :class:`~molpot.ir.PotentialIR` carries: its
        :class:`~molpot.ir.LJBag` holds one ``ε`` / ``σ`` pair for each atom of
        the system, learned per atom rather than per type. Collapsing them onto
        the deck's atom types is
        :meth:`~molix.ff_export.LammpsWriter.write_pair_block`'s job — it has
        the :class:`~molpot.forcefield.TypeAssignment` that says which atoms are
        one type, and this adapter deliberately does not (it translates an IR,
        which knows nothing of a system's typing). ``type_index`` is therefore
        an **atom ordinal** (1-based), not a LAMMPS type id; the writer refuses
        the pair outright if its length disagrees with the assignment it is
        given.

        Args:
            ir: The IR being translated.

        Returns:
            One ``{"type_index", "epsilon", "sigma"}`` entry per atom, ``ε`` in
            kcal/mol and ``σ`` in Å, or an empty list for a charges-only model.
            These are the parameters as the model stores them, **unmixed**: the
            per-pair ``ε_ij`` / ``σ_ij`` that reach the deck are made from them
            by :meth:`~molix.ff_export.LammpsWriter.write_pair_block`, under the
            ``mixing`` rule carried on the same record.
        """
        if ir.lj is None:
            return []
        eps = self.as_floats(ir.lj.epsilon)
        sig = self.as_floats(ir.lj.sigma)
        return [
            {"type_index": i + 1, "epsilon": e, "sigma": s}
            for i, (e, s) in enumerate(zip(eps, sig, strict=True))
        ]

    def _lj_sub_style(
        self,
        ir: PotentialIR,
        method: NonbondedMethodRecord,
    ) -> dict[str, Any] | None:
        """The Lennard-Jones sub-style the model's window and cutoff spell.

        A switching window is what selects ``lj/mdf <r_switch> <r_cut>`` over
        the hard ``lj/cut <r_cut>``: molnex's quintic taper and ``lj/mdf``'s
        expand to the same polynomial, so the window transfers exactly.

        Args:
            ir: The IR being translated.
            method: Its nonbonded method.

        Returns:
            ``{"style", "args"}`` with radii in Å, or ``None`` when the model
            carries no Lennard-Jones bag.

        Raises:
            UnsupportedTermError: A bag with no cutoff radius. Every LAMMPS
                pair style truncates, so no style is an untruncated 12-6 sum.
        """
        if ir.lj is None:
            return None
        if method.lj_cutoff is None:
            raise UnsupportedTermError(
                "lj",
                "Lennard-Jones bag exported without a cutoff radius; every "
                "LAMMPS pair style truncates, so no style is an untruncated "
                "12-6 sum",
            )
        if method.lj_switch_start is None:
            return {"style": "lj/cut", "args": [method.lj_cutoff]}
        return {"style": "lj/mdf", "args": [method.lj_switch_start, method.lj_cutoff]}

    def _coulomb_sub_style(
        self,
        ir: PotentialIR,
        method: NonbondedMethodRecord,
    ) -> dict[str, Any] | None:
        """The Coulomb sub-style the model's solver spells.

        A lattice solver is summed in reciprocal space, so LAMMPS takes the
        real-space half at ``coul/long <r_real>`` and the rest through a
        ``kspace_style``; an open-boundary model is truncated outright at
        ``coul/cut <r_coul>``. The two are not interchangeable, which is why a
        lattice solver with no split radius raises instead of borrowing the
        open-boundary style.

        Args:
            ir: The IR being translated.
            method: Its nonbonded method.

        Returns:
            ``{"style", "args"}`` with radii in Å, or ``None`` when the model
            carries no charges.

        Raises:
            UnsupportedTermError: A lattice solver ran but the record names no
                ``real_space_cutoff`` to split at, or charges were exported with
                neither a lattice solver nor a Coulomb cutoff. The first fails
                closed rather than falling through to ``coul/cut``: that style
                is a *truncated* open-boundary sum with no reciprocal-space
                half, so the deck would silently drop the lattice contribution
                the model was solved with — the same reason
                :func:`_resolved_smearing` refuses a split it cannot state. The
                second has no spelling at all, since every LAMMPS pair style
                truncates and none is an untruncated open-boundary ``1/r`` sum.
        """
        if ir.charges is None:
            return None
        if method.is_lattice_sum:
            if method.real_space_cutoff is None:
                raise UnsupportedTermError(
                    "coulomb",
                    "a lattice Coulomb solver "
                    f"({method.coulomb_calculator!r}) was exported with no "
                    "real_space_cutoff; coul/long has nowhere to split, and "
                    "falling back to coul/cut would export a truncated "
                    "open-boundary sum in place of the lattice one",
                )
            return {"style": "coul/long", "args": [method.real_space_cutoff]}
        if method.coulomb_cutoff is not None:
            return {"style": "coul/cut", "args": [method.coulomb_cutoff]}
        raise UnsupportedTermError(
            "coulomb",
            "charges exported with neither a lattice solver nor a Coulomb "
            "cutoff; LAMMPS has no untruncated open-boundary 1/r sum",
        )

    def _fuse_pair_style(
        self,
        lj: dict[str, Any] | None,
        coulomb: dict[str, Any] | None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Combine the two sub-styles into the one ``pair_style`` LAMMPS takes.

        An **unswitched** LJ term has a fused spelling with Coulomb —
        ``lj/cut/coul/long`` / ``lj/cut/coul/cut`` — and fusing is preferred
        because one style is cheaper than an overlay of two. A **switched** one
        has no fused form, so the two are summed by ``hybrid/overlay``; plain
        ``hybrid`` would hand each type pair to exactly one sub-style and drop
        Coulomb from every LJ pair.

        Args:
            lj: The Lennard-Jones sub-style, or ``None``.
            coulomb: The Coulomb sub-style, or ``None``.

        Returns:
            ``(line, style, sub_styles)`` — the ``pair_style`` line without its
            keyword, the style word the writer branches on, and the sub-style
            records after any fusion. Numbers are rendered by :meth:`number`,
            so the record carries the digits the deck will.
        """
        sub_styles: list[dict[str, Any]] = [sub for sub in (lj, coulomb) if sub is not None]
        if len(sub_styles) == 2 and sub_styles[0]["style"] == "lj/cut":
            first, second = sub_styles
            fused: dict[str, Any] = {
                "style": f"lj/cut/{second['style']}",
                "args": [*first["args"], *second["args"]],
            }
            sub_styles = [fused]
        body = " ".join(
            " ".join([sub["style"], *(self.number(a) for a in sub["args"])]) for sub in sub_styles
        )
        if len(sub_styles) > 1:
            return f"hybrid/overlay {body}", "hybrid/overlay", sub_styles
        return body, sub_styles[0]["style"], sub_styles

    def _kspace_records(self, method: NonbondedMethodRecord) -> list[dict[str, Any]]:
        """The reciprocal-space solver and its splitting parameter.

        ``kspace_style pppm`` is emitted for every lattice calculator,
        including ``"ewald"``: a converged lattice sum is the same number
        whichever solver produced it, so the recorded choice is an accuracy/cost
        trade rather than a different energy, and ``pppm`` at
        :attr:`kspace_accuracy` reproduces it.

        The splitting parameter then travels with it. LAMMPS otherwise derives
        its own from the requested accuracy, which decomposes the *same*
        electrostatics differently: at a finite real-space cutoff neither half
        is converged better than ``erfc(alpha * r_c)``, so two alphas leave two
        different residues and the deck reproduces a neighbouring model rather
        than this one. ``smearing`` is model identity for exactly that reason
        (``molpot/ir/nonbonded.py``), and molnex's Coulomb kernel screens with
        ``erf(r / (sigma * sqrt(2)))``, so ``alpha = 1 / (sigma * sqrt(2))``.

        Args:
            method: The nonbonded method, whose Coulomb term ran a lattice sum.

        Returns:
            The ``kspace`` record followed by the ``kspace_modify`` one.

        Raises:
            ValueError: The record names no ``smearing`` (see
                :func:`_resolved_smearing`).
        """
        alpha = 1.0 / (_resolved_smearing(method.smearing) * math.sqrt(2.0))
        return [
            {
                "type": f"kspace_style pppm {self.number(self.kspace_accuracy)}",
                "style": "pppm",
                "category": "kspace",
                "accuracy": self.kspace_accuracy,
            },
            {
                "type": f"kspace_modify gewald {self.number(alpha)}",
                "style": "gewald",
                "category": "kspace_modify",
                "gewald": alpha,
            },
        ]

    def _exclusions(self, scaling: NonbondedScaling) -> dict[str, Any]:
        """Emit ``special_bonds`` from the force field's 1-2 / 1-3 / 1-4 scales.

        LAMMPS spells the two Class-I interaction classes ``lj`` and ``coul``;
        the IR spells the latter ``charge``. The weights are dimensionless, ``0``
        removing the bonded-path pair and ``1`` keeping it in full.

        Args:
            scaling: The force field's per-interaction-class scales.

        Returns:
            The ``special_bonds`` record, with the LAMMPS weights also carried
            as ``lj`` / ``coul`` float triples.
        """
        lj = list(scaling.triple("lj"))
        coul = list(scaling.triple("charge"))
        weights = " ".join(self.number(x) for x in lj)
        charges = " ".join(self.number(x) for x in coul)
        return {
            "type": f"special_bonds lj {weights} coul {charges}",
            "category": "special_bonds",
            "case": TranslationCase.DIRECT_UNIT_SCALE.value,
            "lj": lj,
            "coul": coul,
        }

    @staticmethod
    def number(value: float) -> str:
        """Render ``value`` the way it will read in the LAMMPS input deck.

        Seven significant digits, and this is the **one** place that decides so.
        A LAMMPS deck is parsed as text: whatever is written here is exactly
        what LAMMPS evaluates, so the rendering is part of the exported model
        and not a display choice. Seven digits keep the round-trip within
        ~1e-7 relative — thirty times under the ``1e-6`` energy gate of the
        classical parity (``interface/tests/parity_classical.py``), where it is
        the measured floor of the Lennard-Jones term — while staying short
        enough that a deck is still something a human reads.

        :meth:`~molix.ff_export.LammpsWriter.write_pair_block` calls this rather
        than restating the format, so the ``pair_style`` line the adapter
        renders and the ``pair_coeff`` lines the writer renders can never
        disagree about precision (ledger debt **F25** records the cost of the
        coefficient table those lines make up).

        Args:
            value: The number, in the record's own unit.

        Returns:
            Its LAMMPS spelling.
        """
        return f"{value:.7g}"

    def _charges(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.charges is not None
        q = self.as_floats(ir.charges.charge)
        return {
            "type": "atom charges",
            "category": "charge",
            "case": TranslationCase.DIRECT_UNIT_SCALE.value,
            "parameters": [{"atom_index": i + 1, "charge": qi} for i, qi in enumerate(q)],
        }
