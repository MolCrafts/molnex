"""Potential style meta for ``Potentials.get_schema()``.

Schema is owned by potential forms (not a free-standing ForceFieldSchema type).

A *style* — one named functional form of an interaction, keyed
``"<category>:<name>"``, see :mod:`molpot.contract` — is registered here once,
with its parameters and its contract.

Every registered style carries its S1 :class:`~molpot.contract.InteractionContract`:
the batch keys it reads, how it is truncated, which exclusion table it consumes,
which derivatives it supports, and which S-decomposition bucket each of its
``Potentials.term_energies()`` keys belongs to. The registry is therefore the
single source of truth for what a ForceField *is* — an unregistered style raises
instead of being synthesised into a blank entry that silently contributes zero
energy, and combinations that do not compose physically are refused by
:meth:`PotentialMeta.check_composition`.

:class:`~molpot.contract.ParamSpec` — the parameter half of that declaration —
is defined in :mod:`molpot.contract` and re-exported here for the registry
below, so :mod:`molpot.manifest` can record a style's parameters without
importing this package.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S1 contract" (the requirement
    labelled ``S1``) and §"Findings ledger" (the silent-zero findings ``F2`` /
    ``F6`` the raises here close).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from molpot.contract import (
    Contribution,
    CutoffKind,
    Derivative,
    ExclusionClass,
    InteractionContract,
    ParamSpec,
    TermSpec,
)

__all__ = ["ParamSpec", "StyleSchema", "ForceFieldSchemaView", "PotentialMeta"]

#: Every classical analytic term in this registry is twice differentiable in
#: closed form, so autograd supplies all three orders.
_ALL_DERIVATIVES = frozenset({Derivative.FORCES, Derivative.VIRIAL, Derivative.HESSIAN})


@dataclass(frozen=True)
class StyleSchema:
    """Declaration of one registered style: its parameters and its contract.

    Attributes:
        category: Style category (``"bond"``, ``"pair"``, …).
        name: Style name within the category (``"harmonic"``, ``"lj/cut"``).
        index_arity: Number of atoms one interaction of this style spans.
        params: Parameters the style reads from the ForceField tables.
        energy_formula: Functional form, as written in the reference.
        options: Every keyword this style accepts from ``def_*style`` — the
            per-model knobs (cutoffs, solver, combining rule) as opposed to
            :attr:`params`, which are the per-type tables. The list is
            **closed**: a keyword outside it is refused at declaration by
            :meth:`~molpot.forcefield.ForceField.def_pairstyle`, because an
            unknown key used to be stored and then never read, so a misspelled
            ``mixng="geometric"`` ran Lorentz–Berthelot and the manifest
            faithfully reported the model that ran rather than the one that was
            meant. Empty for a style with no options at all. Names only: the
            values, their units and their fallbacks belong to
            :class:`~molpot.forcefield.potentials.PairStyleDefaults` and the
            resolver, and the values a given model declared are recorded by
            :attr:`~molpot.manifest.InteractionRecord.options`. That is also why
            this does not appear in :meth:`ForceFieldSchemaView.as_dict` — it is
            the accepted-keyword vocabulary, not part of what a model is.
        contract: The style's S1 declaration. Keyword-only and required: a style
            without a contract cannot be placed in the S-decomposition, so it
            may not be registered at all.
    """

    category: str
    name: str
    index_arity: int
    params: tuple[ParamSpec, ...]
    energy_formula: str = ""
    options: tuple[str, ...] = ()
    contract: InteractionContract = field(kw_only=True)

    @property
    def key(self) -> str:
        """Registry key of this style, ``"<category>:<name>"``."""
        return f"{self.category}:{self.name}"


@dataclass
class ForceFieldSchemaView:
    """Aggregated style meta for the styles present on one ForceField.

    A read-only, mapping-like view: ``key in view``, ``view[key]`` and
    ``view.keys()`` address styles by their registry key.

    Attributes:
        styles: Registry key ``"<category>:<name>"`` → its
            :class:`StyleSchema`, in the order the ForceField declared them.
        units: Unit set the parameter tables are expressed in. ``"real"`` is
            the LAMMPS unit system this layer defaults to: energy kcal/mol,
            length Å, charge in elementary charges ``e``, angles in rad.
    """

    styles: dict[str, StyleSchema] = field(default_factory=dict)
    units: str = "real"

    def keys(self) -> list[str]:
        """Registry keys of the styles in this view, in declaration order."""
        return list(self.styles.keys())

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` names a style in this view."""
        return key in self.styles

    def __getitem__(self, key: str) -> StyleSchema:
        """Schema of one style.

        Raises:
            KeyError: ``key`` is not among the styles of this ForceField.
        """
        return self.styles[key]

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view of the schema, contracts included.

        Every enum is emitted as its string value and the unordered
        ``derivatives`` set as a sorted list, so the same ForceField always
        serialises to the same bytes. The versioned model manifest
        (:class:`molpot.manifest.ModelManifest`) applies the same rules through
        its own codec rather than reusing this payload, so a ForceField and its
        manifest describe the same styles the same way.

        Returns:
            ``{"units": …, "styles": {key: {…, "params": [{…, "frame": …}],
            "contract": {…, "terms": [{"key", "contribution"}, …]}}}}``.
        """
        return {
            "units": self.units,
            "styles": {
                k: {
                    "category": s.category,
                    "name": s.name,
                    "index_arity": s.index_arity,
                    "params": [
                        {
                            "name": p.name,
                            "shape": p.shape,
                            "unit": p.unit,
                            "description": p.description,
                            "frame": p.frame.value,
                        }
                        for p in s.params
                    ],
                    "energy_formula": s.energy_formula,
                    "contract": self._contract_dict(s.contract),
                }
                for k, s in self.styles.items()
            },
        }

    @staticmethod
    def _contract_dict(contract: InteractionContract) -> dict[str, Any]:
        """One contract as plain JSON-ready values."""
        payload = asdict(contract)
        payload["inputs"] = list(contract.inputs)
        payload["cutoff"] = contract.cutoff.value
        payload["exclusion"] = contract.exclusion.value
        payload["derivatives"] = sorted(d.value for d in contract.derivatives)
        payload["terms"] = [
            {**asdict(term), "contribution": term.contribution.value} for term in contract.terms
        ]
        return payload


class PotentialMeta:
    """Registry of style → schema (class-level, not free functions).

    The single source of truth for which styles exist, which parameters each
    reads (with units), and which :class:`~molpot.contract.InteractionContract`
    it carries. Every entry here is looked up by key, never synthesised: an
    unregistered key raises, so a typo cannot become a style that quietly
    contributes no energy.
    """

    _REGISTRY: dict[str, StyleSchema] = {
        "bond:harmonic": StyleSchema(
            category="bond",
            name="harmonic",
            index_arity=2,
            energy_formula="0.5 * k * (r - r0)**2",
            params=(
                ParamSpec("k", "[n]", "kcal/mol/Å²", "force constant"),
                ParamSpec("r0", "[n]", "Å", "equilibrium length"),
            ),
            contract=InteractionContract(
                inputs=("atoms.pos", "bonds.bond_index"),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("bonds", Contribution.LOCAL),),
            ),
        ),
        "angle:harmonic": StyleSchema(
            category="angle",
            name="harmonic",
            index_arity=3,
            energy_formula="0.5 * k * (theta - theta0)**2",
            params=(
                ParamSpec("k", "[n]", "kcal/mol/rad²", "force constant"),
                ParamSpec("theta0", "[n]", "rad", "equilibrium angle"),
            ),
            contract=InteractionContract(
                inputs=("atoms.pos", "angles.atomi", "angles.atomj", "angles.atomk"),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("angles", Contribution.LOCAL),),
            ),
        ),
        "dihedral:periodic": StyleSchema(
            category="dihedral",
            name="periodic",
            index_arity=4,
            energy_formula="sum_m (k_m / s) * [1 + cos(n_m * phi - gamma_m)]",
            params=(
                ParamSpec("k", "[n, n_terms]", "kcal/mol", "barrier heights"),
                ParamSpec("periodicity", "[n_terms]", "1", "integer periods"),
                ParamSpec("phase", "[n, n_terms]", "rad", "phase offsets"),
                ParamSpec("idivf", "[n]", "1", "AMBER idivf scale"),
            ),
            contract=InteractionContract(
                inputs=(
                    "atoms.pos",
                    "propers.atomi",
                    "propers.atomj",
                    "propers.atomk",
                    "propers.atoml",
                ),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("propers", Contribution.LOCAL),),
            ),
        ),
        "dihedral:fourier": StyleSchema(
            category="dihedral",
            name="fourier",
            index_arity=4,
            energy_formula="sum_m (k_m / s) * [1 + cos(n_m * phi - gamma_m)]",
            params=(
                ParamSpec("k", "[n, n_terms]", "kcal/mol", "Fourier amplitudes"),
                ParamSpec("periodicity", "[n_terms]", "1", "integer periods"),
                ParamSpec("phase", "[n, n_terms]", "rad", "phase offsets"),
                ParamSpec("idivf", "[n]", "1", "scale"),
            ),
            contract=InteractionContract(
                inputs=(
                    "atoms.pos",
                    "propers.atomi",
                    "propers.atomj",
                    "propers.atomk",
                    "propers.atoml",
                ),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("propers", Contribution.LOCAL),),
            ),
        ),
        "dihedral:opls": StyleSchema(
            category="dihedral",
            name="opls",
            index_arity=4,
            energy_formula="OPLS 4-term cosine (k1..k4)",
            params=(
                ParamSpec("k1", "[n]", "kcal/mol", "OPLS k1"),
                ParamSpec("k2", "[n]", "kcal/mol", "OPLS k2"),
                ParamSpec("k3", "[n]", "kcal/mol", "OPLS k3"),
                ParamSpec("k4", "[n]", "kcal/mol", "OPLS k4"),
            ),
            contract=InteractionContract(
                inputs=(
                    "atoms.pos",
                    "propers.atomi",
                    "propers.atomj",
                    "propers.atomk",
                    "propers.atoml",
                ),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("propers", Contribution.LOCAL),),
            ),
        ),
        "improper:periodic": StyleSchema(
            category="improper",
            name="periodic",
            index_arity=4,
            energy_formula="sum_m (k_m / s) * [1 + cos(n_m * chi - gamma_m)]",
            params=(
                ParamSpec("k", "[n, n_terms]", "kcal/mol", "barrier heights"),
                ParamSpec("periodicity", "[n_terms]", "1", "integer periods"),
                ParamSpec("phase", "[n, n_terms]", "rad", "phase offsets"),
                ParamSpec("idivf", "[n]", "1", "scale"),
            ),
            contract=InteractionContract(
                # Improper center is atomi (molnex column convention).
                inputs=(
                    "atoms.pos",
                    "impropers.atomi",
                    "impropers.atomj",
                    "impropers.atomk",
                    "impropers.atoml",
                ),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("impropers", Contribution.LOCAL),),
            ),
        ),
        "improper:harmonic": StyleSchema(
            category="improper",
            name="harmonic",
            index_arity=4,
            energy_formula="0.5 * k * (chi - chi0)**2",
            params=(
                ParamSpec("k", "[n]", "kcal/mol/rad²", "force constant"),
                ParamSpec("chi0", "[n]", "rad", "equilibrium improper angle"),
            ),
            contract=InteractionContract(
                inputs=(
                    "atoms.pos",
                    "impropers.atomi",
                    "impropers.atomj",
                    "impropers.atomk",
                    "impropers.atoml",
                ),
                cutoff=CutoffKind.NONE,
                exclusion=ExclusionClass.NONE,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("impropers", Contribution.LOCAL),),
            ),
        ),
        # ``cutoff`` / ``switch_start`` are radii in Å; ``mixing`` is the
        # combining rule making per-pair ``eps`` / ``sig`` out of the per-atom
        # tables, whose closed vocabulary is
        # :class:`~molpot.potentials.vdw.lj126.MixingRule` and is not restated
        # here — ``ForceField.def_pairstyle`` validates the value against the
        # enum itself, so a rule cannot be spelled here and unimplemented
        # there. All three are resolved in one place,
        # ``ForceField._nonbonded_method_record``, and recorded on the model's
        # ``NonbondedMethodRecord``.
        "pair:lj/cut": StyleSchema(
            category="pair",
            name="lj/cut",
            index_arity=2,
            energy_formula="4*eps*((sig/r)**12 - (sig/r)**6)",
            params=(
                ParamSpec("epsilon", "[n_atom_types]", "kcal/mol", "well depth"),
                ParamSpec("sigma", "[n_atom_types]", "Å", "length scale"),
            ),
            options=("cutoff", "switch_start", "mixing"),
            contract=InteractionContract(
                inputs=("atoms.pos", "atoms.epsilon", "atoms.sigma"),
                cutoff=CutoffKind.SWITCHED,
                exclusion=ExclusionClass.PERMANENT,
                derivatives=_ALL_DERIVATIVES,
                # S3: 4ε(σ/r)¹² is the repulsive wall (local), −4ε(σ/r)⁶ is
                # London dispersion — one evaluation, two labelled halves.
                terms=(
                    TermSpec("lj_repulsion", Contribution.LOCAL),
                    TermSpec("lj_dispersion", Contribution.DISP),
                ),
            ),
        ),
        "pair:coulomb": StyleSchema(
            category="pair",
            name="coulomb",
            index_arity=2,
            energy_formula="ke * qi*qj/r with 1-2/1-3/1-4 scales",
            params=(ParamSpec("q", "[n_atoms]", "e", "partial charges on atoms"),),
            # The seven the nonbonded record resolves, plus ``prefactor``:
            # ``k_e`` is a unit-system constant rather than a nonbonded
            # *method*, so it has no field on the record, but
            # ``Potentials._assemble_pair_potentials`` does read it off the
            # style — an accepted keyword either way.
            options=(
                "cutoff",
                "calculator",
                "smearing",
                "real_space_cutoff",
                "lr_wavelength",
                "mesh_spacing",
                "capacity_factor",
                "prefactor",
            ),
            contract=InteractionContract(
                inputs=("atoms.pos", "atoms.q"),
                # EWALD is a *policy* label, not the solver that runs: this one
                # style is truncated at a hard cutoff under open boundaries and
                # lattice-summed under a cell. Which solver actually ran (open
                # cutoff / Ewald / PME) is runtime state, recorded per model by
                # molpot.ir.nonbonded.NonbondedMethodRecord. Because the label spans
                # both cases it says nothing a composition check could compare
                # against; check_composition therefore only requires that a
                # polarization term have some permanent partner (the
                # solver-compatibility relation arrives with composable-a1-01).
                cutoff=CutoffKind.EWALD,
                exclusion=ExclusionClass.PERMANENT,
                derivatives=_ALL_DERIVATIVES,
                terms=(TermSpec("coulomb", Contribution.PERM),),
            ),
        ),
    }

    @classmethod
    def schema_for(
        cls,
        style_keys: list[str] | tuple[str, ...],
        *,
        units: str = "real",
    ) -> ForceFieldSchemaView:
        """Schema view over the energy-bearing styles of one ForceField.

        Declaration-only ``atom:*`` keys are skipped: an atom style names the
        per-atom record layout (``atom:full``), not an energy term, so it has no
        parameters and no contract and never appears in the view.

        Args:
            style_keys: Style keys, typically ``ForceField.style_names()``.
            units: Unit set the parameter tables are expressed in.

        Returns:
            One :class:`StyleSchema` per energy-bearing key, in the given order.

        Raises:
            KeyError: A non-``atom:*`` key is not registered. Such a key used to
                be synthesised into a blank ``"(unregistered style)"`` entry,
                which let a typo'd style contribute zero energy in silence (F6).
        """
        styles: dict[str, StyleSchema] = {}
        for key in style_keys:
            if cls._is_declaration_only(key):
                continue
            styles[key] = cls._schema(key)
        return ForceFieldSchemaView(styles=styles, units=units)

    @classmethod
    def contract_for(cls, key: str) -> InteractionContract:
        """The S1 contract a registered style declares.

        Args:
            key: Style key, ``"<category>:<name>"`` (e.g. ``"pair:lj/cut"``).

        Returns:
            The style's :class:`~molpot.contract.InteractionContract`.

        Raises:
            KeyError: ``key`` is not registered — ``atom:*`` included, since a
                declaration-only style carries no energy and hence no contract.
        """
        return cls._schema(key).contract

    @classmethod
    def options_for(cls, key: str) -> tuple[str, ...]:
        """The keywords a registered style accepts from ``def_*style``.

        The declaring side of :attr:`StyleSchema.options`: whoever validates a
        declaration asks the registry rather than keeping a second list of
        keyword names beside the container.

        Args:
            key: Style key, ``"<category>:<name>"`` (e.g. ``"pair:lj/cut"``).

        Returns:
            The accepted option names, in declaration order; empty for a style
            that takes none.

        Raises:
            KeyError: ``key`` is not registered.
        """
        return cls._schema(key).options

    @classmethod
    def check_composition(cls, style_keys: Sequence[str]) -> None:
        """Refuse style combinations whose energies do not compose (S1).

        Declaration-only ``atom:*`` keys are skipped. The remaining styles are
        checked against two composition rules:

        1. **At most one polarization style.** Two induced-dipole responses on
           the same atoms count the same induction energy twice.
        2. **A polarization style needs at least one permanent partner.** An
           induced dipole has no field to respond to without a style carrying a
           :attr:`~molpot.contract.Contribution.PERM` term. Nothing beyond
           presence is required here: pairing the two by cutoff *family*
           (``HARD`` ↔ ``HARD``, ``EWALD`` ↔ ``EWALD``) is deferred to
           ``composable-a1-01`` as a solver-compatibility relation, because
           ``pair:coulomb`` declares ``EWALD`` as a *policy* label covering both
           the open-boundary cutoff and the lattice sum — which solver actually
           ran is runtime state, carried by
           :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` rather than by
           the label, so
           comparing labels would refuse every real-space-truncated induced
           response on physics it does not actually assert.

        Args:
            style_keys: Style keys of one ForceField, e.g.
                ``ForceField.style_names()``.

        Raises:
            KeyError: A non-``atom:*`` key is not registered.
            ValueError: The combination is refused; the message names the
                offending style keys.
        """
        contracts = {
            key: cls.contract_for(key) for key in style_keys if not cls._is_declaration_only(key)
        }

        polarizing = cls._styles_carrying(contracts, Contribution.POL)
        if len(polarizing) > 1:
            raise ValueError(
                f"styles {polarizing} each declare a polarization term; at most one "
                "polarization style may be composed, otherwise the induced response "
                "is counted twice"
            )
        if not polarizing:
            return

        if not cls._styles_carrying(contracts, Contribution.PERM):
            raise ValueError(
                f"polarization style {polarizing[0]!r} requires at least one "
                f"permanent-electrostatics style to respond to; none among "
                f"{sorted(contracts)}"
            )

    @staticmethod
    def _is_declaration_only(key: str) -> bool:
        """Whether ``key`` names a record layout rather than an energy term."""
        return key.startswith("atom:")

    @staticmethod
    def _styles_carrying(
        contracts: Mapping[str, InteractionContract],
        contribution: Contribution,
    ) -> list[str]:
        """Keys of the styles declaring at least one term in ``contribution``."""
        return [
            key
            for key, contract in contracts.items()
            if any(term.contribution is contribution for term in contract.terms)
        ]

    @classmethod
    def _schema(cls, key: str) -> StyleSchema:
        """Registered schema for ``key``.

        Raises:
            KeyError: ``key`` is not registered.
        """
        try:
            return cls._REGISTRY[key]
        except KeyError:
            raise KeyError(
                f"unregistered style {key!r}; registered styles: {sorted(cls._REGISTRY)}"
            ) from None
