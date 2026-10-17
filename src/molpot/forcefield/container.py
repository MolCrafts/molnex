"""ForceField — molpy ForceField with torch param tables and a torch override.

This **is** a :class:`molpy.ff.forcefield.ForceField`. Styles and native types go through
molpy; learnable continuous tables live on the instance. ``to_potentials()``
overrides the molrs CPU kernels with :class:`Potentials` (torch).
IO is :mod:`molpy.io` — pass this object to the writers after :meth:`realize`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
from molpy.ff.forcefield import ForceField as MolpyForceField

from molix.frame import TensorFrame, as_tensor_frame
from molix.neighbor import NeighborList
from molpot.forcefield.potentials import PAIR_STYLE_DEFAULTS, Potentials
from molpot.forcefield.schema import PotentialMeta
from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    NonbondedMethodRecord,
    PotentialIR,
    ProperTorsionBag,
)
from molpot.ir.scaling import ExclusionScales, NonbondedScaling
from molpot.manifest import (
    MANIFEST_SCHEMA,
    ElectrostaticsRecord,
    InteractionRecord,
    ModelManifest,
    NumericsRecord,
    ParamTableRecord,
)
from molpot.potentials.vdw.lj126 import MixingRule

__all__ = ["ClassIForceField", "ForceField", "OPLSAAForceField", "TypeAssignment"]


#: Model profiles :meth:`ForceField.manifest` can describe. ``"A0"`` is the
#: fixed-charge Class-I form; ``"A1"`` joins it when the induced-response
#: records exist to fill (``composable-a1``). Module-private: a profile is
#: chosen at the one call site that writes a manifest, not looked up.
_MANIFEST_PROFILES = ("A0",)


def _param_key(params: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """A ``def_type`` parameter mapping as a hashable, order-independent key.

    Floats are compared bit-exactly, which is what "these two atoms carry the
    same Lennard-Jones parameters" has to mean here: the values are about to be
    rendered into a LAMMPS deck as text, so two numbers that differ in the last
    ulp are two coefficient lines and therefore two types.
    """
    return tuple(sorted(params.items()))


def _condense(keys: Iterable[Hashable]) -> tuple[tuple[int, ...], int]:
    """Group ``keys`` by equality, numbering the classes in first-occurrence order.

    Args:
        keys: One key per atom or interaction.

    Returns:
        ``(assignment, n_classes)`` — the 0-based class of every item, and how
        many distinct classes there are.
    """
    classes: dict[Hashable, int] = {}
    assignment: list[int] = []
    for key in keys:
        assignment.append(classes.setdefault(key, len(classes)))
    return tuple(assignment), len(classes)


def _representatives(assignment: Sequence[int], n_classes: int) -> tuple[int, ...]:
    """Index of the first member of every class of ``assignment``.

    :meth:`ForceField.realize` registers one molpy type per class and needs one
    concrete member to read the coefficients off; because
    :func:`_condense` numbers classes in first-occurrence order, member ``0`` of
    class ``t`` is also the item that created it.
    """
    first = [-1] * n_classes
    for index, cls in enumerate(assignment):
        if first[cls] < 0:
            first[cls] = index
    return tuple(first)


def _type_labels(prefix: str, count: int) -> tuple[str, ...]:
    """``count`` LAMMPS type labels ``<prefix>1``…, zero-padded to a common width.

    Two constraints shape these strings, and both are measured rather than
    assumed:

    **A label may not begin with a digit.** They leave molnex as LAMMPS *type
    labels* — molpy writes them into the ``Atom Type Labels`` section of the data
    file and onto the ``*_coeff`` lines of the include — and a bare ``"1"`` /
    ``"1-2"`` stops ``read_data`` at ``ERROR: Invalid type label 1-2``, so no
    exported system carrying bonds can be read at all.

    **They must all be the same length.** molrs numbers a data file's types by
    sorting their labels *as text*, and decimal strings sort in numeric order
    only when equally long: unpadded ``t1 … t26`` sorts ``t1, t10, …, t19, t2,
    t20, …``, so LAMMPS type ``2`` is the tenth type and every numeric
    ``pair_coeff i j``
    (:meth:`molix.ff_export.LammpsWriter.write_pair_block`) reads another type's
    Lennard-Jones well depth ``ε`` (kcal/mol) and diameter ``σ`` (Å). Measured on
    a 26-atom water/ion box exported that way, ``E_vdwl`` — LAMMPS' name for the
    summed Lennard-Jones energy — came out ``-0.2251 kcal/mol`` against molnex's
    ``-0.8163 kcal/mol``, no warning.

    Args:
        prefix: Class letter — ``t`` atoms, ``b`` bonds, ``a`` angles,
            ``d`` propers (dihedrals), ``i`` impropers.
        count: Number of types to name.

    Returns:
        The labels, in class order, so ``labels[t]`` names class ``t`` and text
        order is class order.
    """
    width = max(1, len(str(count)))
    return tuple(f"{prefix}{n:0{width}d}" for n in range(1, count + 1))


def _bonded_keys(
    atom_type: Sequence[int],
    members: Sequence[tuple[int, ...]],
    params: Sequence[Mapping[str, float]],
) -> list[tuple[tuple[int, ...], tuple[tuple[str, float], ...]]]:
    """Type key of every bonded interaction: its atom types plus its parameters.

    Both halves are needed. Without the atom types, an O–H and a C–H bond that
    happened to share a force constant would be one LAMMPS type, and the data
    file would then name a type whose ``bond_coeff`` says nothing about which
    atoms it joins; without the parameters, two O–H bonds with different ``k``
    would collapse onto one coefficient line and one of the two force constants
    would silently disappear from the deck.

    The atom types are taken in the interaction's own order — molnex does not
    canonicalise ``(i, j)`` against ``(j, i)``, so a topology that lists a bond
    both ways gets two types carrying identical coefficients. That is a deck one
    line longer than necessary, never a wrong one.
    """
    return [
        (tuple(atom_type[a] for a in atoms), _param_key(row))
        for atoms, row in zip(members, params, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class TypeAssignment:
    """Which LAMMPS type every atom and interaction of one system belongs to.

    The output of :meth:`ForceField.assign_types` and the **one** numbering the
    exporter uses: :meth:`ForceField.realize` registers a molpy type per class
    from it and :class:`~molix.ff_export.LammpsWriter` labels the data file's
    atoms and interactions from the same object, so the coefficient lines of the
    include and the type labels of the data file cannot disagree.

    Pure data — indices, labels and the one parameter a type is *named* by, no
    behaviour. The coefficients themselves stay on the ForceField, which is
    where they can be trained; this type only says which of them are the same.

    ``atom_charges`` is the exception that proves the rule, and it is here
    because it is read on both sides of the molpy seam: it is part of the atom
    key (two atoms are one type only if their charge matches), *and* it is the
    fourth column of the data file's ``Atoms # full`` section. Read a second
    time on the writer's side those two disagree the moment the charges live on
    the field rather than on the system — a deck whose types exist *because* the
    atoms differ in charge, and whose atoms are then all neutral, which LAMMPS
    runs without a word (routed from ``composable-a0-debt-04``).

    Attributes:
        atom_type: Class of every atom, 0-based, indexing ``atom_labels``.
        atom_labels: Label of every atom type (``t1``…), in class order.
        atom_charges: Charge of every atom in ``e``, exactly the values
            :meth:`ForceField.assign_types` keyed the atom types on — the
            field's own ``atoms`` table where it carries ``charge``, else the
            system's ``atoms.charge``, else zeros (``atom_style full`` has a charge
            column either way, so "no charge source" is 0, not absent).
        bond_type: Class of every bond of ``system["bonds"]``.
        bond_labels: Label of every bond type (``b1``…).
        angle_type: Class of every angle of ``system["angles"]``.
        angle_labels: Label of every angle type (``a1``…).
        proper_type: Class of every proper torsion of ``system["propers"]``.
        proper_labels: Label of every proper type (``d1``…, for *dihedral*).

    Impropers have no fields here. They were typed and labelled (``i1``…) until
    2026-09-08 with nothing on either side of the seam reading either one —
    :meth:`ForceField.realize` registers no improper type and
    :class:`~molix.ff_export.LammpsWriter` writes no improper section, both
    because molpy's LAMMPS include writer takes the harmonic form only. A
    numbering no file states is not a smaller version of the gap, it is a
    second thing to keep in step with the first; the day the form is
    registered, the fields come back beside their writer (see
    :meth:`ForceField.assign_types` for the limitation itself).
    """

    atom_type: tuple[int, ...]
    atom_labels: tuple[str, ...]
    atom_charges: tuple[float, ...]
    bond_type: tuple[int, ...] = ()
    bond_labels: tuple[str, ...] = ()
    angle_type: tuple[int, ...] = ()
    angle_labels: tuple[str, ...] = ()
    proper_type: tuple[int, ...] = ()
    proper_labels: tuple[str, ...] = ()


def _optional_float(value: float | str | None) -> float | None:
    """One style option as a float, or ``None`` when it was never declared.

    ``None`` is not ``0.0``: an undeclared switching window means "no window"
    and an undeclared cutoff means "no truncation", and a zero would be a
    different model.
    """
    return None if value is None else float(value)


class ForceField(MolpyForceField):
    """Generic molpy ForceField plus torch param tables.

    Concrete forms such as :class:`OPLSAAForceField` inherit this type and
    register their styles in ``__init__``. Evaluation::

        pots = ff.to_potentials()   # overrides molpy native → torch Potentials
        E = pots.calc_energy(system)

    Args:
        name: Force-field name carried into molpy.
        units: Unit set passed to molpy. ``"real"`` (the default) is the
            LAMMPS convention: energy kcal/mol, length Å, charge ``e``.
        nonbonded_scaling: Dimensionless 1–2 / 1–3 / 1–4 factors the nonbonded
            terms apply to pairs one, two or three covalent bonds apart (see
            :class:`~molpot.ir.NonbondedScaling`). Defaults to
            ``NonbondedScaling()`` — the AMBER-style ``charge`` 0 / 0 / 5⁄6 and
            ``lj`` 0 / 0 / 0.5 — **not** to "no scaling"; a product form with
            different conventions passes its own, as
            :class:`OPLSAAForceField` does.

    Attributes:
        nonbonded_scaling: The scales above — a property, not a plain field:
            assigning it also writes molpy's own ``special_bonds`` arrays, so
            the policy that runs and the policy an exported deck states cannot
            drift apart (see the property and :meth:`set_special_bonds`).
            :meth:`~molpot.forcefield.Potentials.bind_topology` reads them
            once per topology to build its
            :class:`~molpot.forcefield.ExclusionTable`, so replacing this
            attribute after a ``Potentials`` was bound does not reach the
            table already in use.
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
        # The molrs base is built entirely in ``__new__`` (``name``, ``units``) and
        # has no ``__init__`` of its own, so there is nothing to forward here — a
        # ``super().__init__(name, units)`` reaches ``object.__init__`` and raises.
        self.nonbonded_scaling = (
            nonbonded_scaling if nonbonded_scaling is not None else NonbondedScaling()
        )
        self._style_options: dict[str, dict[str, Any]] = {}
        self._params: dict[str, dict[str, torch.Tensor]] = {}
        self._mlip: Any = None

    @property
    def nonbonded_scaling(self) -> NonbondedScaling:
        """The 1–2 / 1–3 / 1–4 exclusion policy, mirrored into molpy on write.

        Assigning here also calls molpy's
        :meth:`~molpy.ff.forcefield.ForceField.set_special_bonds`, because molpy — not molnex
        — is what formats the ``special_bonds`` line of an exported deck, and it
        formats it from *its own* arrays. A policy that lived only on the Python
        side would leave those at molpy's unscaled default and export
        ``special_bonds lj 0 0 1 coul 0 0 1``: every 1–4 pair at full strength,
        which is a different force field and the most expensive error a Class-I
        deck can carry silently. Mirroring on assignment rather than at write
        time means the two can never be out of step, including after
        :meth:`clone`.

        Only the two kinds molpy has a slot for — ``lj`` and ``charge`` (its
        ``coul``) — are mirrored; a kind a later model adds keeps living in
        :class:`~molpot.forcefield.ExclusionTable` alone, since LAMMPS has
        nowhere to put it either.
        """
        return self._nonbonded_scaling

    @nonbonded_scaling.setter
    def nonbonded_scaling(self, scaling: NonbondedScaling) -> None:
        self.set_special_bonds(list(scaling.triple("lj")), list(scaling.triple("charge")))

    def set_special_bonds(self, lj: Any, coul: Any) -> None:
        """Set molpy's exclusion triples **and** this container's scaling, together.

        Overrides :meth:`molpy.ff.forcefield.ForceField.set_special_bonds` so the two
        representations of one policy cannot diverge. molpy owns the arrays an
        exported deck's ``special_bonds`` line is formatted from; molnex owns
        :class:`~molpot.ir.NonbondedScaling`, which
        :meth:`~molpot.forcefield.Potentials.bind_topology` reads to build its
        :class:`~molpot.forcefield.ExclusionTable` and which the IR and the
        manifest report. Writing either one alone leaves the model that runs and
        the model that is described disagreeing about which pairs are excluded —
        the most expensive silent error a Class-I deck can carry, since an
        unscaled 1–4 runs every third-neighbour pair at full strength.

        Both directions land here: :attr:`nonbonded_scaling`'s setter calls this
        method, and a caller reaching for molpy's own spelling gets the molnex
        side updated too.

        Kinds other than ``lj`` and ``charge`` are preserved as they were. molpy has
        exactly two slots and LAMMPS exactly two classes, so a kind a later
        model declares (``induced`` for a polarizable field) has nowhere to go
        on either side and keeps living in the exclusion table alone — it must
        not be dropped merely because this method was called.

        Args:
            lj: Length-3 ``(1-2, 1-3, 1-4)`` Lennard-Jones scales.
            coul: Length-3 ``(1-2, 1-3, 1-4)`` Coulomb scales — molnex's ``charge``
                kind under molpy's ``coul`` name.

        Raises:
            ValueError: Propagated from molpy when either sequence is not
                length 3.
        """
        super().set_special_bonds(lj, coul)
        # __init__ assigns nonbonded_scaling, whose setter calls this method, so
        # the first call happens before there is a previous policy to carry over.
        previous = getattr(self, "_nonbonded_scaling", None)
        scales = (
            {kind: getattr(previous, kind) for kind in previous.kinds()}
            if previous is not None
            else {}
        )
        scales["lj"] = ExclusionScales(*(float(x) for x in lj))
        scales["charge"] = ExclusionScales(*(float(x) for x in coul))
        self._nonbonded_scaling = NonbondedScaling(**scales)

    def def_atomstyle(self, name: str = "full") -> Self:
        self.def_style("atom", name)
        return self

    def def_bondstyle(self, name: str = "harmonic") -> Self:
        self.def_style("bond", name)
        return self

    def def_anglestyle(self, name: str = "harmonic") -> Self:
        self.def_style("angle", name)
        return self

    def def_dihedralstyle(self, name: str = "periodic") -> Self:
        self.def_style("dihedral", name)
        return self

    def def_improperstyle(self, name: str = "harmonic") -> Self:
        self.def_style("improper", name)
        return self

    def def_pairstyle(self, name: str | Any = "lj/cut", **options: Any) -> Self:
        """Register a pair style.

        A string is a classical style (``lj/cut``, ``coulomb``, …). An
        ``nn.Module`` (MACE, PiNet, …) is the MLIP pair term — same seam,
        same ForceField. Periodicity is the neighbour list's job, not a
        second ForceField class.

        **Re-declaring a style updates the options you name and keeps the
        rest.** A second declaration is how a caller states one option on a
        template that already declared the others —
        ``OPLSAAForceField().def_pairstyle("lj/cut", cutoff=12.0)`` gives the
        template's Lennard-Jones a cutoff and leaves its ``mixing="geometric"``
        standing. Replacing the whole option dict instead, as this did until
        2026-09-08, dropped that rule in one line and returned the model to
        Lorentz–Berthelot ``σ``: a different force field on every pair of
        unequal sizes, stated nowhere (ledger finding **F24**). Naming an
        option again overrides it, so ``mixing=`` is still how the rule is
        changed.

        Only the *options* merge. molpy's side of the declaration re-registers
        the style itself, so any types :meth:`realize` put on it do not survive
        a second declaration: declare a field, then realize it, never the other
        way round (which is why the writers realize a :meth:`clone`).

        Option values are kept as given and resolved once, in
        :meth:`_nonbonded_method_record`. Option *keys* are checked here,
        against the style's :attr:`~molpot.forcefield.schema.StyleSchema.options`
        — the only place that knows which keywords are ever read. An unknown key
        used to be stored and then silently never looked at, so
        ``def_pairstyle("lj/cut", mixng="geometric")`` ran Lorentz–Berthelot and
        every description of the model agreed with the run rather than with the
        intent. The one option with a *closed value* vocabulary — ``lj/cut``'s
        ``mixing`` — has its value checked here too, for the same reason: a rule
        is model identity, and an unrecognised one would otherwise reach a
        manifest describing a model nothing can evaluate. Declaring is where
        both fail.

        An unregistered style name is left alone: it has no schema to check
        against, and :class:`~molpot.forcefield.Potentials` refuses it by name
        when the ForceField is assembled. The MLIP branch likewise keeps its
        options unchecked — an ``nn.Module`` pair term is not a registered
        style and has no declared keyword set.

        Args:
            name: Style name, or the ``nn.Module`` of an MLIP pair term.
            **options: Style options, e.g. ``cutoff`` / ``switch_start`` (Å) and
                ``mixing`` (:class:`~molpot.potentials.vdw.lj126.MixingRule`
                value) for ``lj/cut``.

        Returns:
            ``self``, so declarations chain.

        Raises:
            ValueError: An option key the style does not declare, or a
                ``mixing`` value naming no
                :class:`~molpot.potentials.vdw.lj126.MixingRule`.
        """
        if not isinstance(name, str):
            self._mlip = name
            if options:
                self._style_options.setdefault("pair:mlip", {}).update(options)
            return self
        key = f"pair:{name}"
        self._check_style_options(key, options)
        self.def_style("pair", name)
        if options:
            self._style_options.setdefault(key, {}).update(options)
        return self

    @staticmethod
    def _check_style_options(key: str, options: Mapping[str, Any]) -> None:
        """Refuse a declaration a registered style could not read (fail closed).

        Keys come from the registry
        (:meth:`~molpot.forcefield.schema.PotentialMeta.options_for`); the
        ``mixing`` *value* is checked against
        :class:`~molpot.potentials.vdw.lj126.MixingRule` and only for
        ``pair:lj/cut``, whose option it is — on any other style ``mixing`` is
        an unknown key, which the first check has already rejected.

        Args:
            key: Style key being declared, ``"<category>:<name>"``.
            options: The keyword options passed with the declaration.

        Raises:
            ValueError: An option key the style does not declare, or a
                ``mixing`` value naming no :class:`MixingRule`.
        """
        try:
            accepted = PotentialMeta.options_for(key)
        except KeyError:
            return
        for name in options:
            if name not in accepted:
                allowed = ", ".join(accepted) or "(none)"
                raise ValueError(f"{key} has no option {name!r}; options: {allowed}")
        if key == "pair:lj/cut" and "mixing" in options:
            try:
                MixingRule(options["mixing"])
            except ValueError:
                allowed = ", ".join(repr(rule.value) for rule in MixingRule)
                raise ValueError(
                    f"unknown LJ mixing rule {options['mixing']!r}; allowed: {allowed}"
                ) from None

    def style_options(self, key: str) -> dict[str, Any]:
        """Options registered for a style key (e.g. ``pair:lj/cut``)."""
        return dict(self._style_options.get(key, {}))

    def style_names(self) -> list[str]:
        """Declared styles as ``"<category>:<name>"`` keys, in declaration order.

        molrs 0.15 exposes styles as handles (:attr:`styles`); this is the key
        form the registry (:class:`~molpot.forcefield.schema.PotentialMeta`) and
        the option table here are indexed by.
        """
        return [f"{style.category}:{style.name}" for style in self.styles]

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
        """A second container describing the same model — the base of every copy-on-write.

        **By value** — the clone owns these and may be rewritten without the
        receiver noticing: the parameter tables (:meth:`params`, one fresh dict
        per category; the tensors themselves are shared, which is what makes
        :meth:`with_params` / :meth:`to_dtype` copy-on-write rather than a deep
        copy), the per-style options (:meth:`style_options`), the exclusion
        policy (:attr:`nonbonded_scaling`, re-assigned through the setter so
        molpy's ``special_bonds`` arrays are mirrored on the clone too), and the
        molpy style registry, which is rebuilt by re-declaring each name.

        **By reference** — the mlip pair term (``def_pairstyle(module)``). This
        is a clone of the *container*, not of the network: the clone evaluates
        the same ``nn.Module``, sharing its parameters, its buffers, and any
        compiled core installed on it. That is the intended meaning — restating
        a field at another dtype or with fitted tables must not fork the model
        being trained — and it is why the carry lives here rather than in the
        callers: rebuilding styles by name cannot reach the mlip, because the
        mlip *has* no name (it is the module), so a clone that stopped at the
        registry handed :meth:`to_potentials` ``mlip=None`` and evaluated a
        silently zero-energy field.

        The mlip is deliberately not a molpy style — the term loop of
        :class:`~molpot.forcefield.Potentials` is empty for an mlip field — so
        there is nothing for :meth:`has_style` / :meth:`style_names` to carry
        beyond its ``"pair:mlip"`` options entry, which travels with the rest by
        value; both containers report the same style set either way.

        Returns:
            A new container of this class describing the same model.
        """
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
        out._mlip = self._mlip
        return out

    def to_dtype(self, dtype: torch.dtype) -> Self:
        """Copy-on-write: the same field with every float parameter table at ``dtype``.

        The precision of a *field* is the precision of its numbers, and they do
        not all live in one place: the assembled :class:`Potentials` registers
        the bonded tables as buffers (so ``.to(dtype)`` reaches those), while
        the per-atom nonbonded ones — ``atoms.epsilon`` / ``atoms.sigma`` /
        ``atoms.charge`` — are read live off this container on every evaluation and
        no module cast can see them. This method is the container's half:
        restate every table, then assemble. Both halves together are what
        ``MD(dtype, potential_dtype=…)`` means by evaluating a potential at a
        second precision (:class:`molix.md.MD`).

        The reason molnex has that seam at all is the
        quantization-as-thermal-noise study in :mod:`molix.quant`, whose
        operating point is an fp64 trajectory over an fp32 potential: lowering
        the *potential's* precision perturbs the forces, and the effect of that
        perturbation on the trajectory can only be read against an integrator
        that is not itself a second noise source. Running the whole loop at
        fp32 confounds the two.

        Copy-on-write like :meth:`with_params`: the receiver keeps its own
        tensors, and only floating tables are touched — type-id and index
        columns keep their integer dtype, which a blanket ``.to(dtype)`` would
        destroy.

        Args:
            dtype: Floating dtype for every float parameter table.

        Returns:
            A new container of this class at ``dtype``.
        """
        out = self.clone()
        out._params = {
            category: {
                name: value.to(dtype) if value.is_floating_point() else value
                for name, value in table.items()
            }
            for category, table in self._params.items()
        }
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

    def neighbor_list(self, *, pbc: bool = False, max_num_pairs: int = 512) -> NeighborList:
        """The :class:`~molix.data.NeighborList` this field is priced against (F21).

        Since ``composable-a0-03b`` molpot consumes a placed ``pairs`` namespace
        and never enumerates one: the live pair set of every nonbonded term *is*
        whatever list the batch carries. Which radius that list has to reach is
        therefore not the caller's taste but this field's identity — the largest
        radius among **the terms that will run**, each already resolved once by
        :meth:`_nonbonded_method_record`. A list built shorter than that
        silently drops every pair between the two radii, with nothing raised:
        the private ``LJ_CUT_CUTOFF = 10.0`` the four fitting scripts each kept
        a copy of would take a periodic OPLS-AA model, whose real-space Coulomb
        reaches 15 Å, down to under a third of its electrostatic pairs (ledger
        finding **F21**).

        **``pbc`` is the switch between the two Coulomb terms.**
        :meth:`Potentials._assemble_pair_potentials` builds *both* — the
        truncated open sum
        (:class:`~molpot.forcefield.pair_terms.CoulombOpenPotential`) and the
        lattice sum (:class:`~molpot.forcefield.pair_terms.CoulombPeriodicPotential`,
        PME by default) — and picks between them by whether the batch carries a
        cell. The record therefore describes both possibilities at once, and
        its :attr:`~molpot.ir.nonbonded.NonbondedMethodRecord.is_lattice_sum`
        cannot be the discriminator: an undeclared ``calculator`` resolves to
        the ``pme`` default, so it is true for *every* declared ``pair:coulomb``,
        including a model that will only ever run open. The data settles it, so
        ``pbc`` does: periodic → the Ewald split's ``real_space_cutoff``; open →
        the declared ``coulomb_cutoff``.

        An open ``pair:coulomb`` with no cutoff has no radius at all — it is the
        full ``N(N-1)/2`` sum, which no finite list reproduces — so it is
        refused rather than given a number nobody chose. That is why
        ``OPLSAAForceField().neighbor_list()`` raises by design: the template
        declares ``pair:coulomb`` bare, which under open boundaries is
        untruncated electrostatics. The same field is well defined the moment
        the boundary is stated — ``neighbor_list(pbc=True)`` is 15.0 Å, the
        ``3 × smearing`` real-space split — or the moment the truncation is,
        ``def_pairstyle("coulomb", cutoff=…)``.

        ``symmetry`` is not an argument, because only one value of it is alive:
        :meth:`Potentials._placed_pairs` prices the ``source < target`` half of
        a bidirectional list, and ``NeighborList(symmetry=False)`` emits the
        *other* half, which every pair term multiplies by an all-false mask —
        an energy of exactly zero, refused at
        :meth:`Potentials.bind_topology` since ``composable-a0-03b`` rather
        than run (ledger finding **F22**). So the list handed out here is
        always full bidirectional.

        ``pbc`` and ``max_num_pairs`` are properties of the *data*, not of the
        model: the same parameters price an isolated molecule and a periodic
        box, and the pair buffer is sized by the systems being batched. They are
        forwarded so a caller never has to rebuild the list by hand to change
        one — the path by which a script ends up enumerating a graph at a radius
        its force field does not run at.

        **What this list does not carry.**
        :meth:`molix.data.NeighborList.execute` writes ``edges.pair_index``,
        ``edges.pair_diff`` and ``edges.pair_dist`` — no ``edges.shifts``, the
        per-pair lattice remainder in Å. Energies and forces are exact without
        it, but :meth:`Potentials.energy` refuses a batch under a strain
        variable (a virial) that has none, because the remainder it would
        otherwise rebuild is a detached constant that carries no lattice
        strain (ledger finding **F13**). A run that needs the virial builds its list with
        :class:`molix.md.NeighborList`, which writes ``shifts``.

        Args:
            pbc: Whether the batched systems are periodic — both the boundary
                the list itself applies (minimum image against ``graphs.cell``
                ``(B, 3, 3)`` in Å) and the statement of which Coulomb term the
                model will run.
            max_num_pairs: Half-pair buffer the C++ kernel is given, a count
                rather than a length; the list grows it to ``N(N-1)/2`` when a
                system needs more.

        Returns:
            The list, at ``cutoff`` in Å and ``symmetry=True``, ready to mount
            as a :class:`~molix.data.task.BatchTask` (``DataModule(...,
            batch_tasks=[ff.neighbor_list(), TotalCharge()])``) or to
            ``execute`` on a batch at a collate site that bypasses one.

        Raises:
            ValueError: An open-boundary ``pair:coulomb`` (``pbc=False``) with
                no declared cutoff, which is the untruncated ``N(N-1)/2`` sum;
                an ``nn.Module`` (MLIP) pair term, whose radius is the
                encoder's ``r_max`` and not a style option this record can
                read — a field that prices its nonbonded energy through an
                MLIP is asked for its list at the encoder's cutoff, not here;
                or no pair style declared at all.

        References:
            ``.claude/notes/composable-potentials.md`` — findings **F21** (the
            scripts that built lists at a radius their field did not run at),
            **F22** (why ``symmetry=False`` reads as dead) and **F13** (the
            missing ``shifts`` and the virial).
        """
        method = self._nonbonded_method_record()
        radii: list[float] = []
        if method.lj_cutoff is not None:
            radii.append(float(method.lj_cutoff))
        if self.has_coulomb():
            if pbc:
                if method.real_space_cutoff is None:
                    # Unreachable through ``def_pairstyle``: the resolver derives
                    # this radius from the smearing for every declared coulomb
                    # style. A ``None`` here is that resolver disagreeing with
                    # itself, not a user error — and a zero-radius list would be
                    # an empty electrostatics nobody asked for.
                    raise ValueError(
                        "the nonbonded record declares pair:coulomb but leaves "
                        "'real_space_cutoff' unset; there is no periodic radius to cover"
                    )
                radii.append(float(method.real_space_cutoff))
            elif method.coulomb_cutoff is not None:
                radii.append(float(method.coulomb_cutoff))
            else:
                raise ValueError(
                    "open-boundary Coulomb without a cutoff needs every pair; "
                    "declare cutoff=... on pair:coulomb or build the list with "
                    "pbc=True for a periodic system"
                )
        if not radii:
            if self._mlip is not None:
                # The MLIP is a pair term with no style options: its radius is
                # a property of the encoder (``r_max`` on its config), which
                # this record never reads. Naming a radius here would be this
                # field guessing at another object's cutoff.
                raise ValueError(
                    "the pair term is an MLIP module: its cutoff is the encoder's "
                    "own (r_max on its config), not this force field's record; "
                    "build the list at that radius"
                )
            raise ValueError("no pair style declared: nothing to price")
        return NeighborList(
            cutoff=max(radii),
            max_num_pairs=max_num_pairs,
            pbc=pbc,
            symmetry=True,
        )

    def assign_types(
        self,
        system: Any,
        *,
        masses: torch.Tensor | list[float] | None = None,
    ) -> TypeAssignment:
        """Condense ``system`` onto LAMMPS types by parameter equality (F25).

        A LAMMPS type is an equivalence class of *parameters*, not an atom: two
        atoms carrying the same mass, charge and Lennard-Jones pair
        (``ε`` kcal/mol, ``σ`` Å) are one atom type, and two interactions between
        the same atom types with the same force constants are one bonded type.
        molnex exported one type per atom and one per interaction until
        2026-09-08 — *identity typing*, the ``composable-a0`` shortcut. That is
        correct and quadratic: with the explicit ``pair_coeff i j`` lines
        :meth:`molix.ff_export.LammpsWriter.write_pair_block` has to write, 26
        atoms cost 351 coefficient lines and 1000 atoms half a million, plus a
        type label apiece in the data file (ledger debt **F25**). Condensing
        costs one pass over the system and leaves that block ``T(T+1)/2`` lines
        in the number of *distinct* types — ten lines for the 26-atom water/ion
        box of ``interface/tests/parity_classical.py``, which has four.

        This is the **single owner** of the mapping. :meth:`realize` registers
        one molpy type per class from it, and
        :class:`~molix.ff_export.LammpsWriter` labels the data file's atoms and
        interactions from the same object. Two independent condensations are two
        numberings that can differ once, and the deck states its coefficients by
        number — a numbering that slips is a ``pair_coeff`` line reading another
        type's ``ε`` / ``σ``, which LAMMPS accepts without a word.

        **What enters a key.** Exactly the parameters :meth:`realize` would hand
        to molpy's ``def_type``, compared as bit-exact fp64: mass, charge and the
        Lennard-Jones pair for an atom, dropping any of them the field carries no
        table for (a field with no charges types by mass alone); the ordered
        tuple of atom types plus the interaction's own parameters for a bonded
        type, so two O–H bonds with different ``(k, r0)`` stay two types. An atom
        type is therefore "the atoms molpy would be given identical parameters
        for", which is the only grouping that cannot change what the deck means.

        **Labels.** ``t``/``b``/``a``/``d`` for atoms / bonds / angles /
        propers, numbered from 1 in first-occurrence order and
        zero-padded to the type count, so text order — the order molrs numbers a
        data file's types in — is class order (:func:`_type_labels`).

        The per-atom charges that entered the atom key travel out on the
        assignment (:attr:`TypeAssignment.atom_charges`), because the data file
        has to state the same ones it typed the atoms by — see that attribute.

        **Impropers are not typed at all.** :meth:`realize` registers no
        improper type — molpy's LAMMPS include writer takes the harmonic form
        only, and ``improper:periodic``, the form :class:`OPLSAAForceField`
        declares, has no spelling to be registered under — and
        :class:`~molix.ff_export.LammpsWriter` writes no improper section, so a
        field whose impropers carry energy exports a deck that omits that term
        (ledger debt **F25**). They *were* keyed here, by their atom types
        alone, and labelled ``i1``… for a writer that has never read either;
        that numbering was deleted on 2026-09-08 rather than kept warm, since
        an unread class map is one more thing to keep in step with the file it
        does not appear in. It returns when the form does — with parameters in
        :meth:`_proper_params`' shape, so the types split by force constant
        like every other bonded class.

        Args:
            system: Post-collate nested batch, read through its typed view
                (:func:`~molix.data.as_tensor_frame`), so a
                ``TensorDict``, a :class:`~molix.data.TensorFrame` and a
                nested mapping of the namespaces are one system. Read here:
                ``atoms.pos`` ``(N, 3)`` in Å — only its length — ``atoms.charge``
                in units of the elementary charge ``e`` when no ``charge`` parameter
                table is filled, and the bonded namespaces ``bonds`` /
                ``angles`` / ``propers`` (:meth:`_arity`, which takes the
                collate's ``bonds.bond_index`` COO as readily as ``atomi`` …
                ``atoml`` columns). An ``impropers`` namespace is not read
                (above).
            masses: Per-atom mass in amu ``(N,)``. Default: ``12.011`` amu — the
                mass of carbon — for every atom. Mass is part of the atom key, so
                the default collapses a mixed system onto fewer types than it
                has species; pass the real masses to any export that will move.

        Returns:
            The assignment.

        Raises:
            TypeError: ``system`` is not a batch, a view of one or a nested
                mapping of its namespaces, or ``system["atoms"]["pos"]`` is not
                a tensor.
            ValueError: ``masses`` or a parameter table has a different length
                than ``atoms.pos``.
            KeyError: A bonded namespace is present but missing one of the index
                columns its arity needs.

        References:
            ``.claude/notes/composable-potentials.md`` — finding **F25**
            (identity typing and the quadratic pair block it forced), closed by
            this method.
        """
        frame = as_tensor_frame(system)
        pos = self._system_tensor(frame, "atoms", "pos")
        n_atoms = int(pos.shape[0])
        atom_type, n_atom_types = _condense(
            _param_key(row) for row in self._atom_params(frame, n_atoms, masses)
        )
        # The same resolver ``_atom_params`` keyed the atoms with, so the charge
        # the deck writes is the charge its typing was computed from.
        charges = self._charge_list(frame, n_atoms)
        bonds = self._arity(frame, "bonds", 2)
        angles = self._arity(frame, "angles", 3)
        propers = self._arity(frame, "propers", 4)
        bond_type, n_bond_types = _condense(
            _bonded_keys(atom_type, bonds, self._bond_params(len(bonds)))
        )
        angle_type, n_angle_types = _condense(
            _bonded_keys(atom_type, angles, self._angle_params(len(angles)))
        )
        proper_type, n_proper_types = _condense(
            _bonded_keys(atom_type, propers, self._proper_params(len(propers)))
        )
        return TypeAssignment(
            atom_type=atom_type,
            atom_labels=_type_labels("t", n_atom_types),
            atom_charges=tuple(charges) if charges is not None else (0.0,) * n_atoms,
            bond_type=bond_type,
            bond_labels=_type_labels("b", n_bond_types),
            angle_type=angle_type,
            angle_labels=_type_labels("a", n_angle_types),
            proper_type=proper_type,
            proper_labels=_type_labels("d", n_proper_types),
        )

    def realize(
        self,
        system: Any,
        *,
        masses: torch.Tensor | list[float] | None = None,
        types: TypeAssignment | None = None,
        pairs: bool = True,
    ) -> Self:
        """Fill native molpy type tables from torch params + topology.

        ``pairs=False`` registers no pair types: a **bonded-only** realization,
        for a deck whose nonbonded section is written elsewhere (the molrs
        writer prints a ``pair_coeff`` line for every pair type it holds).

        One molpy type per class of :meth:`assign_types` — the condensed typing
        that replaced one-type-per-atom on 2026-09-08 (ledger debt **F25**) —
        registered under that method's labels. After this, :mod:`molpy.io` can
        write ``self`` directly. Bonded types are given their label explicitly
        (``def_type("b1", …)``) rather than letting molpy compose one from
        the atom types: two O–H bonds with different force constants share their
        atom types, so a composed name would collide and one of the two
        ``bond_coeff`` lines would silently disappear.

        Mutates this container in place, which is why the writers clone first
        (:meth:`molix.ff_export.LammpsWriter.write_settings` realizes
        ``forcefield.clone()``): a realized field carries the types of *that*
        system and is no longer a template for any other.

        **One condensation per deck.** A caller that already holds the
        assignment — every exporter does, because the data file has to label
        the same classes this registers — passes it as ``types`` and this
        computes none. It used to condense unconditionally, so writing one deck
        ran :meth:`assign_types` three times over the same system (once for the
        caller, once for the writer's own equality guard, once here) to reach
        the answer it was handed.

        Args:
            system: Post-collate nested batch; see :meth:`assign_types` for the
                columns read.
            masses: Per-atom mass in amu ``(N,)``. Default: ``12.011`` amu — the
                mass of carbon — for every atom, which suffices for a
                single-point energy and is wrong for anything that integrates.
                Read whether or not ``types`` is given — the mass a type is
                *registered* with comes from here — so it must be the same
                ``masses`` the given assignment was condensed at, mass being
                part of an atom key.
            types: This system's typing under this field. Passing one **asserts
                it is this field's own** — ``self.assign_types(system,
                masses=masses)`` — and nothing re-derives it to check: an
                assignment from another field, another system or other masses
                would register molpy types under labels its data file never
                uses, which LAMMPS reads as a shifted numbering without
                complaint. Default: condense it here.

        Returns:
            ``self``, so the call chains straight into a molpy writer.

        Raises:
            TypeError: ``system["atoms"]["pos"]`` is not a tensor.
            ValueError: ``masses`` — or one of the parameter tables the types
                are keyed on — has a different length than ``atoms.pos``
                (raised through :meth:`assign_types`).
            KeyError: A bonded namespace is present but missing one of the
                index columns its arity needs.
        """
        frame = as_tensor_frame(system)
        if types is None:
            types = self.assign_types(frame, masses=masses)
        pos = self._system_tensor(frame, "atoms", "pos")
        n_atoms = int(pos.shape[0])
        astyle = self.get_style("atom", "full")
        if astyle is None:
            self.def_atomstyle("full")
            astyle = self.get_style("atom", "full")
        params = self._atom_params(frame, n_atoms, masses)
        atom_reps = _representatives(types.atom_type, len(types.atom_labels))
        atom_types = [
            astyle.def_type(label, **params[rep])
            for label, rep in zip(types.atom_labels, atom_reps, strict=True)
        ]
        self._realize_bonds(frame, atom_types, types)
        self._realize_angles(frame, atom_types, types)
        self._realize_propers(frame, atom_types, types)
        if pairs:
            self._realize_pairs(atom_types, atom_reps)
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
                )

        lj_bag = None
        charge_bag = None
        atoms = self.params("atoms")
        if atoms is not None:
            if "epsilon" in atoms and "sigma" in atoms:
                lj_bag = LJBag(epsilon=atoms["epsilon"], sigma=atoms["sigma"])
            q = atoms.get("charge")
            if q is not None:
                charge_bag = ChargeBag(charge=q)

        return PotentialIR(
            bonds=bonds_bag,
            angles=angles_bag,
            propers=propers_bag,
            impropers_periodic=impropers_periodic,
            impropers_harmonic=impropers_harmonic,
            lj=lj_bag,
            charges=charge_bag,
            scaling=self.nonbonded_scaling,
            nonbonded_method=self._nonbonded_method_record(),
            unit_system="class_i_canonical",
        )

    def _nonbonded_method_record(self) -> NonbondedMethodRecord:
        """The nonbonded method this ForceField will actually run (F7).

        The **single** resolver of the pair-style options: each field is the
        declared option where one was given and otherwise the
        :data:`~molpot.forcefield.potentials.PAIR_STYLE_DEFAULTS` fallback.
        :meth:`~molpot.forcefield.potentials.Potentials._assemble_pair_potentials`
        builds its terms from this record and :meth:`to_ir` / :meth:`manifest`
        report it. Each of those three calls this method for itself and gets a
        fresh record — nothing caches one — but because they all resolve the
        options here, the model that runs and the model that is described come
        out equal and cannot disagree.

        The real-space split radius is the one derived fallback: unset, it is
        ``real_space_cutoff_factor × smearing``, and that effective radius is
        what the record states — a ``None`` there would name no radius at all
        while the term ran at one. Only the two options with no fallback of any
        kind — the Lennard-Jones switching window and the open-boundary Coulomb
        cutoff — stay ``None`` when unset: they mean "no window" and "no
        truncation", which are absences rather than numbers.

        The Lennard-Jones combining rule is a plain fallback like the cutoffs
        (:attr:`~molpot.forcefield.potentials.PairStyleDefaults.lj_mixing`), and
        the only one whose value set is closed — :meth:`def_pairstyle` already
        refused anything outside
        :class:`~molpot.potentials.vdw.lj126.MixingRule`, so what is read back
        here is a rule that exists.

        Returns:
            The record. A ForceField with no pair style gets one with every
            field ``None`` rather than no record at all, so a consumer branches
            on a field and never on the record's presence.
        """
        defaults = PAIR_STYLE_DEFAULTS
        lj_cutoff: float | None = None
        lj_switch_start: float | None = None
        lj_mixing: str | None = None
        coulomb_cutoff: float | None = None
        coulomb_calculator: str | None = None
        real_space_cutoff: float | None = None
        smearing: float | None = None
        lr_wavelength: float | None = None
        mesh_spacing: float | None = None
        capacity_factor: float | None = None
        if "lj/cut" in self.pair_styles():
            lj = self.style_options("pair:lj/cut")
            lj_cutoff = float(lj.get("cutoff", defaults.lj_cutoff))
            lj_switch_start = _optional_float(lj.get("switch_start"))
            lj_mixing = str(lj.get("mixing", defaults.lj_mixing))
        if self.has_coulomb():
            coulomb = self.style_options("pair:coulomb")
            coulomb_cutoff = _optional_float(coulomb.get("cutoff"))
            coulomb_calculator = str(coulomb.get("calculator", defaults.coulomb_calculator))
            smearing = float(coulomb.get("smearing", defaults.smearing))
            declared_split = _optional_float(coulomb.get("real_space_cutoff"))
            real_space_cutoff = (
                defaults.real_space_cutoff_factor * smearing
                if declared_split is None
                else declared_split
            )
            lr_wavelength = float(coulomb.get("lr_wavelength", defaults.lr_wavelength))
            mesh_spacing = float(coulomb.get("mesh_spacing", defaults.mesh_spacing))
            capacity_factor = float(coulomb.get("capacity_factor", defaults.capacity_factor))
        return NonbondedMethodRecord(
            lj_cutoff=lj_cutoff,
            lj_switch_start=lj_switch_start,
            lj_mixing=lj_mixing,
            coulomb_cutoff=coulomb_cutoff,
            coulomb_calculator=coulomb_calculator,
            real_space_cutoff=real_space_cutoff,
            smearing=smearing,
            lr_wavelength=lr_wavelength,
            mesh_spacing=mesh_spacing,
            capacity_factor=capacity_factor,
        )

    def manifest(self, *, profile: str) -> ModelManifest:
        """Describe this force field as a versioned :class:`ModelManifest` (S2).

        The ForceField is where the styles, their options, the parameter tables
        and the exclusion scaling already live, so it is what fills the
        manifest; :class:`~molpot.forcefield.Potentials` is only its torch view
        and owns no second copy of any of them.

        :attr:`~molpot.manifest.ModelManifest.representation` and
        :attr:`~molpot.manifest.ModelManifest.provenance` are left ``None`` on
        purpose — composition is the caller's job, with ``record`` the
        :class:`~molpot.manifest.RepresentationRecord` of whatever filled this
        ForceField's tables (the fuller example, electrostatics included, is on
        :class:`~molpot.manifest.ModelManifest`)::

            import dataclasses

            from molpot.manifest import ProvenanceRecord

            manifest = dataclasses.replace(
                ff.manifest(profile="A0"),
                representation=record,
                provenance=ProvenanceRecord(
                    reference_method="wB97X-D/def2-TZVPD",
                    data_version="v0",
                    run_id="exp-0001",
                ),
            )

        What the manifest says about electrostatics is limited to what a
        container can know: the multipole order (``0`` — point charges) and the
        permanent exclusion scaling. A ``pair:coulomb`` style is not a charge
        model, so it constrains nothing and runs no solve; ``atoms.charge`` may
        equally have been read from a file. The constraint and the solver come
        from :meth:`AtomicChargeModel.electrostatics_record
        <molpot.charge.AtomicChargeModel.electrostatics_record>` and are
        composed in the same way as the representation.

        Args:
            profile: Model profile this force field belongs to — ``"A0"`` for
                the fixed-charge Class-I form.

        Returns:
            The manifest: one :class:`~molpot.manifest.InteractionRecord` per
            declared energy-bearing style (``atom:*`` is a per-atom record
            layout, not a term, and is skipped), the shared
            :class:`~molpot.ir.nonbonded.NonbondedMethodRecord`, the A0
            electrostatics with this ForceField's permanent exclusion scaling,
            a content hash per filled parameter category, and the global
            numerical precision. Units are this ForceField's own
            (``"real"``: kcal/mol, Å, ``e``).

        Raises:
            ValueError: ``profile`` is not one this molnex describes
                (``"A0"`` today). Written through verbatim it would label the
                manifest as a profile whose records — induced scaling, a
                response solver — this schema has no fields for, and the
                mislabelling would surface only in whatever read it back.
            KeyError: A declared non-``atom:*`` style is not registered in
                :class:`~molpot.forcefield.PotentialMeta` — an unregistered
                style has no contract to record.
        """
        if profile not in _MANIFEST_PROFILES:
            raise ValueError(
                f"unknown manifest profile {profile!r}; this molnex describes "
                f"{', '.join(_MANIFEST_PROFILES)}"
            )
        view = PotentialMeta.schema_for(self.style_names(), units=self.units)
        return ModelManifest(
            schema_version=MANIFEST_SCHEMA,
            profile=profile,
            units=self.units,
            interactions=tuple(
                InteractionRecord(
                    style=key,
                    contract=schema.contract,
                    params=schema.params,
                    options=self.style_options(key),
                )
                for key, schema in view.styles.items()
            ),
            nonbonded=self._nonbonded_method_record(),
            electrostatics=ElectrostaticsRecord(
                multipole_order=0,
                polarization_scheme=None,
                damping=None,
                # A container holds charges; it does not obtain them. The
                # charge model names the constraint it imposed and the solve it
                # ran (``AtomicChargeModel.electrostatics_record``), and the
                # caller composes the two records.
                charge_constraint="none",
                total_charge=None,
                permanent_scaling=self.nonbonded_scaling,
                induced_scaling=None,
                solver=None,
            ),
            tables=tuple(
                ParamTableRecord.from_tables(category, tables)
                for category, tables in sorted(self._params.items())
                if tables
            ),
            numerics=NumericsRecord.from_runtime(),
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
            periodicity = (
                torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
                .expand(n, 4)
                .contiguous()
            )
            return ProperTorsionBag(k=k, periodicity=periodicity, phase=phase)

        pp = ProperTorsionBag._cosine_fields(pp)
        if "k" not in pp:
            raise ValueError("propers params need 'k', cartesian 'a'/'b', or OPLS 'k1..k4'")
        return ProperTorsionBag(
            k=pp["k"],
            periodicity=pp["periodicity"],
            phase=pp["phase"],
        )

    def _atom_params(
        self,
        frame: TensorFrame,
        n_atoms: int,
        masses: torch.Tensor | list[float] | None,
    ) -> list[dict[str, float]]:
        """The molpy ``AtomStyle.def_type`` parameters of every atom.

        One mapping per atom: ``mass`` in amu always, ``charge`` in ``e`` when
        the field's ``atoms`` table or the system carries charges, and
        ``epsilon`` (kcal/mol) / ``sigma`` (Å) when the ``atoms`` table carries
        both. These are both the atom key :meth:`assign_types` groups on and the
        coefficients :meth:`realize` registers — one function, so a type can
        never be a class of atoms that would have been registered differently.

        Raises:
            ValueError: A table does not have one row per atom. Silently reading
                the first ``n_atoms`` rows of a longer table would type the
                system against another system's parameters.
        """
        rows: list[dict[str, float]] = [{"mass": mass} for mass in self._mass_list(masses, n_atoms)]
        charges = self._charge_list(frame, n_atoms)
        if charges is not None:
            for row, charge in zip(rows, charges, strict=True):
                row["charge"] = charge
        atoms = self.params("atoms")
        if atoms is not None and "epsilon" in atoms and "sigma" in atoms:
            epsilon = [float(x) for x in atoms["epsilon"].detach().cpu().reshape(-1).tolist()]
            sigma = [float(x) for x in atoms["sigma"].detach().cpu().reshape(-1).tolist()]
            for row, eps, sig in zip(rows, epsilon, sigma, strict=True):
                row["epsilon"] = eps
                row["sigma"] = sig
        return rows

    def _bond_params(self, count: int) -> list[dict[str, float]]:
        """``BondStyle.def_type`` parameters of every bond — ``k``, ``r0``.

        In molrs's convention, which is LAMMPS's (molrs ≥ 0.16): ``k`` is
        LAMMPS's ``K`` in ``K (r − r0)²``, i.e. half of this container's
        ``½ k (r − r0)²`` force constant. Handing molrs the container's ``k``
        unconverted doubled every realized bond energy.

        Empty mappings when the field declares no harmonic bond style or carries
        no ``bonds`` table: :meth:`realize` registers nothing for those bonds, so
        nothing distinguishes them beyond the atom types they join.
        """
        bp = self.params("bonds")
        if self.bond_style() != "harmonic" or bp is None:
            return [{} for _ in range(count)]
        return [
            {"k": 0.5 * float(bp["k"][n].item()), "r0": float(bp["r0"][n].item())}
            for n in range(count)
        ]

    def _angle_params(self, count: int) -> list[dict[str, float]]:
        """``AngleStyle.def_type`` parameters of every angle — ``k``, ``theta0``.

        In molrs's (LAMMPS's) convention: ``k`` is ``K`` in ``K (θ − θ0)²``,
        half of this container's ``½ k`` force constant, and ``theta0`` is in
        **degrees**, where the container stores radians.
        """
        ap = self.params("angles")
        if self.angle_style() != "harmonic" or ap is None:
            return [{} for _ in range(count)]
        return [
            {
                "k": 0.5 * float(ap["k"][n].item()),
                "theta0": math.degrees(float(ap["theta0"][n].item())),
            }
            for n in range(count)
        ]

    def _proper_params(self, count: int) -> list[dict[str, float]]:
        """``DihedralStyle.def_type`` parameters of every proper torsion.

        The canonical spelling on both sides: ``k1``…``k4`` for an OPLS table,
        and the per-term ``k{m}`` / ``periodicity{m}`` / ``phase{m}`` triples of
        spec ff-params-01 for a general periodic table — the same spellings
        :meth:`_realize_propers` registers, since the type key is defined as
        what would be registered. This used to translate into molpy's ``c1``…
        ``c4``, which the molrs OPLS kernel does not read: every torsion of a
        realized field then evaluated to zero, in silence (spec ff-params-01).

        A periodic table used to emit the first amplitude alone, under the bare
        name ``k``. The molrs kernel requires the multiplicity with it, so a
        realized ``ClassIForceField`` could not be built into potentials at all,
        and every term past the first was dropped on the floor.
        """
        pp = self.params("propers")
        if self.dihedral_style() is None or pp is None:
            return [{} for _ in range(count)]
        if "k1" in pp:
            return [
                {
                    "k1": float(pp["k1"][n].item()),
                    "k2": float(pp["k2"][n].item()),
                    "k3": float(pp["k3"][n].item()),
                    "k4": float(pp["k4"][n].item()),
                }
                for n in range(count)
            ]
        # Cartesian ``(a, b)`` is the fit's interior form and a shared
        # periodicity row is legal: :meth:`_propers_to_ir` is the one place that
        # converts and broadcasts, so a realized field and its IR cannot disagree.
        bag = self._propers_to_ir(pp)
        n_terms = bag.k.shape[1]
        out: list[dict[str, float]] = []
        for n in range(count):
            terms: dict[str, float] = {}
            for m in range(n_terms):
                terms[f"k{m + 1}"] = float(bag.k[n, m].item())
                terms[f"periodicity{m + 1}"] = float(bag.periodicity[n, m].item())
                # molrs stores every phase in degrees; the bag holds radians.
                terms[f"phase{m + 1}"] = math.degrees(float(bag.phase[n, m].item()))
            out.append(terms)
        return out

    def _realize_bonds(
        self, frame: TensorFrame, atom_types: list[Any], types: TypeAssignment
    ) -> None:
        if self.bond_style() != "harmonic":
            return
        bp = self.params("bonds")
        bonds = self._arity(frame, "bonds", 2)
        if bp is None or not bonds:
            return
        bstyle = self.get_style("bond", "harmonic")
        params = self._bond_params(len(bonds))
        for label, rep in zip(
            types.bond_labels,
            _representatives(types.bond_type, len(types.bond_labels)),
            strict=True,
        ):
            i, j = bonds[rep]
            bstyle.def_type(
                label,
                atom_types[types.atom_type[i]],
                atom_types[types.atom_type[j]],
                **params[rep],
            )

    def _realize_angles(
        self, frame: TensorFrame, atom_types: list[Any], types: TypeAssignment
    ) -> None:
        if self.angle_style() != "harmonic":
            return
        ap = self.params("angles")
        angles = self._arity(frame, "angles", 3)
        if ap is None or not angles:
            return
        astyle = self.get_style("angle", "harmonic")
        params = self._angle_params(len(angles))
        for label, rep in zip(
            types.angle_labels,
            _representatives(types.angle_type, len(types.angle_labels)),
            strict=True,
        ):
            i, j, k = angles[rep]
            astyle.def_type(
                label,
                atom_types[types.atom_type[i]],
                atom_types[types.atom_type[j]],
                atom_types[types.atom_type[k]],
                **params[rep],
            )

    def _realize_propers(
        self, frame: TensorFrame, atom_types: list[Any], types: TypeAssignment
    ) -> None:
        style = self.dihedral_style()
        if style is None:
            return
        pp = self.params("propers")
        propers = self._arity(frame, "propers", 4)
        if pp is None or not propers:
            return
        dstyle = self.get_style("dihedral", style)
        params = self._proper_params(len(propers))
        for label, rep in zip(
            types.proper_labels,
            _representatives(types.proper_type, len(types.proper_labels)),
            strict=True,
        ):
            i, j, k, l = propers[rep]
            dstyle.def_type(
                label,
                atom_types[types.atom_type[i]],
                atom_types[types.atom_type[j]],
                atom_types[types.atom_type[k]],
                atom_types[types.atom_type[l]],
                **params[rep],
            )

    def _realize_pairs(self, atom_types: list[Any], atom_reps: Sequence[int]) -> None:
        if "lj/cut" not in self.pair_styles():
            return
        atoms = self.params("atoms")
        if atoms is None or "epsilon" not in atoms:
            return
        pstyle = self.get_style("pair", "lj/cut")
        for at, rep in zip(atom_types, atom_reps, strict=True):
            # A self pair is named by its atom type, as molrs keys it.
            pstyle.def_type(
                at.name,
                at,
                epsilon=float(atoms["epsilon"][rep].item()),
                sigma=float(atoms["sigma"][rep].item()),
            )

    @staticmethod
    def _system_tensor(frame: TensorFrame, *keys: str) -> torch.Tensor:
        """One leaf of the typed view, refused unless it is a tensor.

        Args:
            frame: The batch's typed view (:func:`~molix.data.as_tensor_frame`).
            keys: The leaf's path, e.g. ``("atoms", "pos")``.

        Returns:
            The leaf.

        Raises:
            TypeError: The leaf is not a tensor.
        """
        value = frame[keys] if len(keys) > 1 else frame[keys[0]]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"system{list(keys)} must be a tensor, got {type(value)!r}")
        return value

    @staticmethod
    def _arity(frame: TensorFrame, ns: str, arity: int) -> list[tuple[int, ...]]:
        """One connectivity namespace as a list of per-interaction index tuples.

        ``bonds`` has **two** spellings on a batch, and both are read here for
        the same reason :meth:`~molpot.forcefield.potentials.Potentials._bond_index`
        reads both: :func:`~molix.data.collate.collate_molecules` writes the
        schema's ``(2, N)`` COO ``bond_index`` — whose shape is the anti-alias
        guard against ``pair_index`` — while a caller assembling the namespace
        by hand writes ``atomi`` / ``atomj`` columns, as every valence namespace
        does. Reading only the columns made every Frame-collated system look
        like a system with no bonds at all, which types and exports a deck whose
        molecules have come apart. ``angles`` / ``propers`` / ``impropers`` have
        the column spelling only.

        Args:
            frame: The batch's typed view.
            ns: The namespace — ``"bonds"``, ``"angles"`` or ``"propers"``.
            arity: How many atoms one interaction of that namespace spans.

        Returns:
            One ``(i, j, …)`` tuple of global atom indices per interaction, in
            the namespace's own row order; empty when the namespace is absent.

        Raises:
            KeyError: The namespace is present but carries neither a usable
                index block nor the columns its arity needs.
            ValueError: ``bonds.bond_index`` is not the schema's ``(arity, N)``
                COO. A cutoff neighbour list's ``(E, 2)`` ``pair_index`` is the
                shape this catches, and the whole reason the two differ
                (CLAUDE.md § pair_index vs bond_index): read as bonds it would
                type and export a molecule that is bonded by proximity.
        """
        if ns not in frame.keys():
            return []
        block = frame[ns]
        packed = block.get("bond_index", None) if ns == "bonds" else None
        if packed is not None:
            if int(packed.shape[0]) != arity:
                raise ValueError(
                    f"system[{ns!r}].bond_index is {tuple(packed.shape)}, not the schema's "
                    f"({arity}, N) COO — a cutoff neighbour graph is not a covalent topology, "
                    "and the differing shape is what says so"
                )
            arrays = [row.detach().cpu().long().reshape(-1) for row in packed]
        else:
            cols = ("atomi", "atomj", "atomk", "atoml")[:arity]
            missing = [c for c in cols if c not in block.keys()]
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

    def _charge_list(self, frame: TensorFrame, n_atoms: int) -> list[float] | None:
        """Per-atom charge in ``e``, in one fixed order of precedence, or ``None``.

        The field's own ``atoms`` table wins over the system's ``atoms.charge``
        column, because a trained or freshly assigned charge lives on the field
        and the batch may still carry the charges it was read from disk with.
        ``None`` means neither source exists — which :meth:`assign_types` turns
        into zeros for the data file's charge column, since ``atom_style full``
        has that column either way, while :meth:`_atom_params` instead omits
        ``charge`` from the type key so a field with no charges types by mass
        (and Lennard-Jones parameters) alone.

        The **single** reader of both sources: :meth:`assign_types` keys the
        atom types on what this returns and hands the same list out on
        :attr:`TypeAssignment.atom_charges`, so the charges a deck states are
        the charges its types were condensed by.
        """
        atoms = self.params("atoms")
        if atoms is not None and "charge" in atoms:
            return [float(x) for x in atoms["charge"].detach().cpu().reshape(-1).tolist()]
        if "atoms" in frame.keys() and "charge" in frame.atoms.keys():
            q = frame.atoms["charge"]
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
    """A general Class-I container: the diagonal forms, no vendor dialect.

    Harmonic bonds and angles, the **general periodic torsion** for propers and
    impropers, Lennard-Jones 12-6 and Coulomb. This is the shape a *fitted*
    field takes: the torsion is the full cosine series ``Σ_m k_m [1 + cos(n_m φ
    − γ_m)]`` with its multiplicities and phases free per type, so an AMBER or
    CHARMM torsion fits as naturally as an OPLS one. OPLS-AA is the special
    case where every type carries ``n = 1,2,3,4`` and ``γ = 0, π, 0, π``; it is
    :class:`OPLSAAForceField`, a named product, and the exporters detect that
    manifold and write ``dihedral_style opls`` when it holds (spec
    ff-params-01 § 5).

    Choosing this over the named product is what keeps a learned field from
    being born inside one vendor's functional form.

    Args:
        name: Force-field name.
        units: Unit set passed to molpy (default ``"real"``).
        nonbonded_scaling: Dimensionless 1-2 / 1-3 / 1-4 scales. No default is
            invented here — the scales are part of the parameterisation being
            fitted, so the caller states them.
        mixing: Lennard-Jones combination rule for ``pair:lj/cut``.
    """

    def __init__(
        self,
        name: str = "class-i",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
        mixing: str = "lorentz-berthelot",
    ) -> None:
        super().__init__(name, units, nonbonded_scaling=nonbonded_scaling)
        self.def_atomstyle("full")
        self.def_bondstyle("harmonic")
        self.def_anglestyle("harmonic")
        self.def_dihedralstyle("periodic")
        self.def_improperstyle("periodic")
        self.def_pairstyle("lj/cut", mixing=mixing)
        self.def_pairstyle("coulomb")


class OPLSAAForceField(ForceField):
    """OPLS-AA Class-I force-field container (LAMMPS ``units real``).

    Registers the OPLS-AA diagonal form on construction: harmonic bond and
    angle, ``dihedral:opls`` proper, ``improper:periodic``, ``pair:lj/cut``,
    and ``pair:coulomb``. When ``nonbonded_scaling`` is omitted, binds the
    OPLS-AA 1–4 scales ``f14 = 1/2`` for both Coulomb and LJ
    (``q.s14 == lj.s14 == 0.5``); the 1–2 and 1–3 scales remain ``0``, i.e.
    directly bonded pairs and the two ends of a bond angle are excluded from
    the nonbonded terms entirely. This differs from the
    :class:`~molpot.ir.NonbondedScaling` default, whose 1–4 Coulomb factor is
    the AMBER ``5⁄6``. An explicit ``nonbonded_scaling`` is kept as given (not
    silently replaced).

    The ``pair:lj/cut`` style is declared with ``mixing="geometric"``: OPLS-AA
    combines **both** ``ε`` and ``σ`` geometrically (Jorgensen 1996), unlike the
    Lorentz–Berthelot arithmetic ``σ`` of AMBER / CHARMM, which is molnex's
    default and which molnex applied to this template too until 2026-09-08
    (ledger finding ``F24``). The two differ on every pair of unequal sizes —
    ``σ = 3.5 Å`` with ``2.5 Å`` gives ``2.958 Å`` rather than ``3.0 Å`` — so
    the rule the parameters were fitted with has to be the rule they are
    evaluated with, and it now rides on the model's
    :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` into every manifest and
    exported deck.

    Units follow LAMMPS real: energy kcal/mol, length Å, charge ``e``. The
    scale factors themselves are dimensionless.

    References:
        Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. "Development and
        Testing of the OPLS All-Atom Force Field on Conformational Energetics
        and Properties of Organic Liquids" J. Am. Chem. Soc. 118, 11225 (1996).
        https://doi.org/10.1021/ja9621760

    Args:
        name: Force-field name (default ``"OPLS-AA"``).
        units: Unit set passed to molpy (default ``"real"``).
        nonbonded_scaling: Optional dimensionless 1–2 / 1–3 / 1–4 scales; the
            default is the OPLS-AA ``f14 = 0.5`` pair above.
    """

    def __init__(
        self,
        name: str = "OPLS-AA",
        units: str = "real",
        *,
        nonbonded_scaling: NonbondedScaling | None = None,
    ) -> None:
        if nonbonded_scaling is None:
            nonbonded_scaling = NonbondedScaling(
                charge=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            )
        super().__init__(name, units, nonbonded_scaling=nonbonded_scaling)
        self.def_atomstyle("full")
        self.def_bondstyle("harmonic")
        self.def_anglestyle("harmonic")
        self.def_dihedralstyle("opls")
        self.def_improperstyle("periodic")
        self.def_pairstyle("lj/cut", mixing="geometric")
        self.def_pairstyle("coulomb")
