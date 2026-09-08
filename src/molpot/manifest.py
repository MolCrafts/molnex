"""ModelManifest — the versioned description of *which model this is* (S2).

One manifest answers, for a single trained or hand-built model: what
representation and checkpoint produced its parameters, which interaction terms
are enabled and under which contracts, which nonbonded method actually ran,
what the electrostatics are (multipole order, polarization scheme, charge
constraint, damping, exclusion scaling and solver), what the parameter tables
contain (by content hash), at what numerical precision it runs, and where it
came from (reference method, energy offsets, data version, software versions).
``S2`` is simply how the design notes cited below label that requirement; it is
used as a shorthand throughout this package and carries no other meaning.

It is a **composition of small frozen records**, never a god struct: each
record answers one question, and :class:`ModelManifest` only aggregates them.
Composition is likewise the caller's job — :meth:`ForceField.manifest
<molpot.forcefield.ForceField.manifest>` fills what a force field knows and
leaves :attr:`~ModelManifest.representation` and
:attr:`~ModelManifest.provenance` unset; the caller merges in
:meth:`Parameterizer.representation_record
<molpot.forcefield.Parameterizer.representation_record>` and its own run
identity with :func:`dataclasses.replace`. There is no façade that does it for
them.

Serialisation is a single recursive codec (:func:`_encode` / :func:`_decode`)
driven by the records' own type hints: enums are emitted as their values,
unordered sets sorted, tuples listed, and :meth:`ModelManifest.from_dict` is
the exact inverse of :meth:`ModelManifest.to_dict`. The payload is therefore
deterministic — the same model always serialises to the same bytes, which is
what lets two runs be compared (and a manifest itself be hashed).

:data:`MANIFEST_SCHEMA` versions this layout and is deliberately independent of
``molix.engine.export.LAMMPS_META_SCHEMA``: the manifest describes the model,
that constant describes the LAMMPS export contract, and they move for
different reasons.

Layering: this module sits *below* the ForceField package. It imports
:mod:`molpot.contract` (including :class:`~molpot.contract.ParamSpec`),
:mod:`molpot.ir`, :mod:`molix.config`, torch, numpy (only to take the
little-endian fp64 bytes :meth:`ParamTableRecord.from_tables` hashes) and the
standard library, and
nothing from ``molpot.forcefield`` at all — not the container that fills a
manifest, and not the schema module that declares the styles it records. The
records are therefore importable by anything that reads a manifest, and the
import graph, rather than a convention, is what keeps it that way.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table);
    spec ``.claude/specs/composable-a0-04-manifest.md``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch

from molix.config import config
from molpot.contract import InteractionContract, ParamSpec
from molpot.ir.nonbonded import NonbondedMethodRecord
from molpot.ir.scaling import NonbondedScaling

if TYPE_CHECKING:  # the stubs-only protocol "is a dataclass type", for the codec below
    from _typeshed import DataclassInstance

__all__ = [
    "ElectrostaticsRecord",
    "InteractionRecord",
    "MANIFEST_SCHEMA",
    "ModelManifest",
    "NumericsRecord",
    "ParamTableRecord",
    "ProvenanceRecord",
    "RepresentationRecord",
    "SolverRecord",
]

#: Version of the :class:`ModelManifest` layout. Bumping it is a breaking
#: change to every consumer that reads a manifest, not a refactor; it is
#: versioned independently of ``molix.engine.export.LAMMPS_META_SCHEMA``.
#:
#: ``2`` (2026-09-08) added
#: :attr:`~molpot.ir.nonbonded.NonbondedMethodRecord.lj_mixing`, the
#: Lennard-Jones combining rule. The field rides inside a nested record, so a
#: schema-1 payload still parses — :meth:`ModelManifest.from_dict` reads one and
#: fills the field with ``None``, the value that field defines as
#: "Lorentz–Berthelot by the then-default". The bump is nonetheless required in
#: the *other* direction: without it a reader of this vintage would silently
#: accept a newer payload and drop whatever model-identity fields it grew, which
#: is exactly the failure a version exists to prevent.
MANIFEST_SCHEMA = 2

#: Payload versions :meth:`ModelManifest.from_dict` can read, oldest first. A
#: version outside this set is refused rather than parsed best-effort.
_READABLE_SCHEMAS = (1, MANIFEST_SCHEMA)


@dataclass(frozen=True)
class RepresentationRecord:
    """Which representation produced the model's parameters.

    Attributes:
        family: Kind of representation — ``"graph-parameterizer"`` for a
            :class:`~molpot.forcefield.Parameterizer` that predicts classical
            parameter tables from the covalent graph.
        encoder: Class name of the encoder, e.g. ``"PiNet"``.
        checkpoint_id: Identity of the weights the encoder was loaded from.
            ``None`` when they are not persisted (an untrained or in-memory
            model); the training side fills it (``composable-a0-05``).
        spec: The encoder's own configuration, as JSON-native values.
    """

    family: str
    encoder: str
    checkpoint_id: str | None = None
    spec: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse an encoder spec this manifest could not write.

        Raises:
            TypeError: A value in :attr:`spec` (or nested in a list or dict
                inside it) is not JSON-native. The message names the key, so
                the offending entry is fixed where the encoder declared it
                rather than where the manifest is finally written.
        """
        _check_json_native(self.spec, "RepresentationRecord.spec")


@dataclass(frozen=True)
class InteractionRecord:
    """One enabled interaction term: its style, contract, params and options.

    Attributes:
        style: Registry key of the style, ``"<category>:<name>"`` — e.g.
            ``"pair:lj/cut"``.
        contract: The style's S1 declaration
            (:class:`~molpot.contract.InteractionContract`): batch keys read,
            cutoff treatment, exclusion class, supported derivatives and the
            S-decomposition label of each named energy term.
        params: Parameters the style reads from the force field's tables, with
            their shapes, units and transformation laws.
        options: Style options as declared, e.g. ``{"cutoff": 12.0}`` for
            ``pair:lj/cut`` (cutoffs in Å). Empty when the style was declared
            bare and runs on the assembly defaults.
    """

    style: str
    contract: InteractionContract
    params: tuple[ParamSpec, ...] = ()
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a style option this manifest could not write.

        :attr:`options` is payload — it is copied verbatim into
        :meth:`ModelManifest.to_dict`. A tensor among the options survives
        every equality and round-trip check in memory and only fails at
        :meth:`ModelManifest.write` time, in a caller far from the
        ``def_pairstyle`` that introduced it, by which point the manifest has
        already been compared against another run's. It is caught here, where
        it enters.

        Raises:
            TypeError: An option value (or a value nested in a list or dict
                option) is not JSON-native. The message names the option key.
        """
        _check_json_native(self.options, "InteractionRecord.options")


@dataclass(frozen=True)
class SolverRecord:
    """The iterative solver a response or charge model ran, and its tolerance.

    Attributes:
        kind: Solver name, e.g. ``"eem"`` (charge equilibration) or ``"pcg"``
            (preconditioned conjugate gradient for an induced response).
        A direct solve (one exact linear solve, e.g. the bordered-KKT charge
        equilibration) reports ``iterations=1``, ``tolerance=0.0`` and
        ``preconditioner="none"``; iterative solvers report their fixed
        iteration count and tolerance.
        tolerance: Convergence tolerance on the residual norm, in the solver's
            own units; ``None`` for a direct solve.
        preconditioner: Preconditioner name, e.g. ``"diagonal"``; ``None`` when
            unpreconditioned.
    """

    kind: str
    iterations: int | None = None
    tolerance: float | None = None
    preconditioner: str | None = None


@dataclass(frozen=True)
class ElectrostaticsRecord:
    """What the model's electrostatics are, beyond the solver that ran.

    Attributes:
        multipole_order: Highest permanent multipole carried per atom — ``0``
            for point charges, ``1`` with dipoles, ``2`` with quadrupoles.
        polarization_scheme: How an induced response is obtained, e.g.
            ``"variational"``; ``None`` for a non-polarizable model.
        damping: Short-range damping of the multipole interaction, e.g.
            ``"thole"``; ``None`` when undamped.
        charge_constraint: Constraint the charge model enforces, e.g.
            ``"total-charge-kkt"`` for a per-graph total-charge constraint
            imposed through a KKT system, or ``"none"`` for fixed charges.
        total_charge: Total charge in ``e`` when the model pins one for every
            graph; ``None`` when it is a per-batch input rather than model
            identity.
        permanent_scaling: 1–2 / 1–3 / 1–4 scale factors (dimensionless) the
            permanent electrostatics and Lennard-Jones terms apply. Model
            identity, not a runtime detail (S5): the same tables under a
            different 1–4 convention are a different force field.
        induced_scaling: Scaling of the induced response. Always ``None`` in
            profile A0, which declares no induced style; the ``InducedScaling``
            record arrives with ``composable-a1``.
        solver: The solver the charge or response model ran; ``None`` when the
            charges are given rather than solved for.
    """

    multipole_order: int
    polarization_scheme: str | None
    damping: str | None
    charge_constraint: str
    total_charge: float | None
    permanent_scaling: NonbondedScaling
    induced_scaling: None
    solver: SolverRecord | None


@dataclass(frozen=True)
class ParamTableRecord:
    """Content hash and layout of one category of parameter tables.

    Attributes:
        category: Parameter category — ``"bonds"``, ``"angles"``,
            ``"propers"``, ``"impropers"``, ``"pairs"`` or ``"atoms"``.
        names: Table names, sorted; the order the digest concatenates them in.
        shapes: One shape per name, in :attr:`names` order.
        dtype: Storage dtype of the tables, spelled ``str(torch.float64)``.
            Categories whose tables differ (a float table beside an integer
            periodicity) report the promoted type, which is the precision the
            continuous parameters are stored at.
        sha256: Digest defined by :meth:`from_tables`.
    """

    category: str
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtype: str
    sha256: str

    @classmethod
    def from_tables(cls, category: str, tables: Mapping[str, torch.Tensor]) -> ParamTableRecord:
        """Hash one category of parameter tables by content.

        The digest is the SHA-256 of the concatenated little-endian IEEE-754
        binary64 bytes of every table, in :attr:`names` (i.e. sorted) order.
        Tables are up-cast to fp64 first, so the digest is a function of the
        *numbers* and not of the precision they happen to be stored at: the
        same force field hashed after a float32 and a float64 pass agrees.
        Names, shapes and dtype are recorded beside the digest rather than
        folded into it, so a table that was merely renamed is visible as such.

        Args:
            category: Parameter category the tables belong to.
            tables: Table name → tensor, e.g. ``{"k": …, "r0": …}`` for
                ``"bonds"`` (``k`` in kcal mol⁻¹ Å⁻², ``r0`` in Å). Gradients
                and device placement are irrelevant to the digest: tensors are
                detached and moved to the host before hashing.

        Returns:
            The record for this category.

        Raises:
            ValueError: ``tables`` is empty — an empty category describes
                nothing, and hashing it would give every such category the
                same digest as every other.
        """
        if not tables:
            raise ValueError(
                f"ParamTableRecord.from_tables({category!r}, …) got no tables; "
                "an unfilled category has no content to hash and must be omitted"
            )
        names = tuple(sorted(tables))
        values = [tables[name].detach().to(device="cpu") for name in names]
        digest = hashlib.sha256()
        for value in values:
            digest.update(np.asarray(value.to(torch.float64).contiguous()).astype("<f8").tobytes())
        dtype = functools.reduce(torch.promote_types, (value.dtype for value in values))
        return cls(
            category=category,
            names=names,
            shapes=tuple(tuple(int(size) for size in value.shape) for value in values),
            dtype=str(dtype),
            sha256=digest.hexdigest(),
        )


@dataclass(frozen=True)
class NumericsRecord:
    """Numerical precision the model runs at.

    Attributes:
        dtype: Global parameter / batch dtype, spelled ``str(torch.float32)``
            — ``molix.config["ftype"]``.
        matmul_precision: Process-wide fp32 matmul policy, ``"highest"``,
            ``"high"`` or ``"medium"``. It lives only in torch
            (``torch.get_float32_matmul_precision()``), never in the config
            mapping, so the effective policy is what is recorded.
        energy_dtype: Dtype the energy is accumulated in when it differs from
            :attr:`dtype`; ``None`` when it does not.
    """

    dtype: str
    matmul_precision: str
    energy_dtype: str | None = None

    @classmethod
    def from_runtime(cls) -> NumericsRecord:
        """Read the precision this process is actually running at.

        The two sources are read here and nowhere else, so a manifest and the
        model it describes cannot disagree about precision:
        ``molix.config["ftype"]`` for the global parameter / batch dtype, and
        :func:`torch.get_float32_matmul_precision` for the fp32 matmul policy.
        The policy is deliberately *not* read from the config mapping —
        :meth:`molix.config.MolnexConfig.set_precision` skips that key when it
        copies a preset and calls :func:`torch.set_float32_matmul_precision`
        instead, so torch holds the only effective value.

        Returns:
            The record. :attr:`energy_dtype` is ``None``: molnex accumulates
            energies in the global dtype, and a record that named a second one
            would describe an accumulation nothing performs.
        """
        return cls(
            dtype=str(config["ftype"]),
            matmul_precision=torch.get_float32_matmul_precision(),
            energy_dtype=None,
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """Where the model came from — reference data, software, run identity.

    Attributes:
        reference_method: Electronic-structure method the training targets came
            from, e.g. ``"wB97X-D/def2-TZVPD"``.
        energy_offsets_id: Identity of the per-element energy offsets the
            targets were shifted by; ``None`` when unshifted.
        data_version: Version of the training dataset.
        software: Package name → version string, e.g. ``{"molnex": "0.1.0"}``.
        checkpoint_id: Identity of the checkpoint this model was read from.
        run_id: molexp experiment identity, as a plain string — never an
            import, so a manifest stays readable without the experiment
            tracker installed.
        created_at: ISO-8601 UTC timestamp of when the manifest was written.
        git_sha: Commit the model was produced at.
    """

    reference_method: str | None = None
    energy_offsets_id: str | None = None
    data_version: str | None = None
    software: dict[str, str] = field(default_factory=dict)
    checkpoint_id: str | None = None
    run_id: str | None = None
    created_at: str | None = None
    git_sha: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    """Versioned, JSON-round-trippable identity of one model (S2).

    Frozen: a manifest describes one model at one moment, and rewriting a
    field in place would make it describe neither.

    Attributes:
        schema_version: :data:`MANIFEST_SCHEMA` at the time of writing.
        profile: Model profile this manifest belongs to — ``"A0"`` for the
            fixed-charge Class-I form, ``"A1"`` once polarization lands.
        units: Unit set the parameter tables are expressed in. ``"real"`` is
            the LAMMPS convention: energy kcal/mol, length Å, charge ``e``.
        interactions: One record per enabled interaction term, in the order the
            force field declared them.
        nonbonded: Which nonbonded method ran — cutoffs in Å, the switching
            window, the LJ combining rule, the Coulomb solver and its sizing. It comes from the same
            single resolver that
            :attr:`~molpot.ir.PotentialIR.nonbonded_method` and the assembled
            pair terms come from (:meth:`ForceField._nonbonded_method_record
            <molpot.forcefield.container.ForceField._nonbonded_method_record>`),
            which each caller invokes for itself: the records are separate but
            equal objects, resolved from one reading of the style options, so
            they cannot report different models. Always present: a model that
            states no pair term at all gets an all-``None`` record, so a
            consumer branches on a field and never on the record.
        electrostatics: Multipole order, polarization, charge constraint,
            exclusion scaling and solver. Always present, on the same
            convention: a model with no charges reports
            ``charge_constraint="none"`` and no solver rather than no record.
        tables: One content hash per filled parameter category.
        numerics: Precision the model runs at.
        representation: The representation that produced the parameters.
            ``None`` from :meth:`ForceField.manifest
            <molpot.forcefield.ForceField.manifest>` — the caller composes it
            in from :meth:`Parameterizer.representation_record
            <molpot.forcefield.Parameterizer.representation_record>`.
        provenance: Reference data, software versions and run identity;
            ``None`` until the caller fills it.

    Examples:
        Each object fills the part of the manifest it alone knows, and the
        caller joins them with :func:`dataclasses.replace` — ``ff`` is the
        :class:`~molpot.forcefield.ForceField` whose parameter tables are
        filled, ``model`` the :class:`~molpot.forcefield.Parameterizer` that
        filled them, and ``charge_model`` the
        :class:`~molpot.charge.AtomicChargeModel` that solved for the partial
        charges ``atoms.q`` (in ``e``)::

            import dataclasses

            manifest = dataclasses.replace(
                ff.manifest(profile="A0"),
                representation=model.representation_record(),
                electrostatics=charge_model.electrostatics_record(
                    permanent_scaling=ff.nonbonded_scaling
                ),
                provenance=ProvenanceRecord(
                    reference_method="wB97X-D/def2-TZVPD",
                    data_version="v0",
                    software={"molnex": "0.1.0"},
                    run_id="exp-0001",
                ),
            )
            manifest.write("model.json")
            assert ModelManifest.read("model.json") == manifest
    """

    schema_version: int
    profile: str
    units: str
    interactions: tuple[InteractionRecord, ...]
    nonbonded: NonbondedMethodRecord
    electrostatics: ElectrostaticsRecord
    tables: tuple[ParamTableRecord, ...]
    numerics: NumericsRecord
    representation: RepresentationRecord | None = None
    provenance: ProvenanceRecord | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-native values.

        Enums become their string values, the unordered ``derivatives`` set of
        each contract becomes a sorted list, and every tuple becomes a list, so
        the same model always produces the same payload.

        Returns:
            A mapping accepted by :func:`json.dumps` as it stands, and by
            :meth:`from_dict` to rebuild this manifest exactly.
        """
        return {f.name: _encode(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelManifest:
        """Rebuild a manifest from :meth:`to_dict` output — its exact inverse.

        Reads **schema 1 and schema 2**, and nothing else. The difference
        between them is one key of one nested record:
        :attr:`~molpot.ir.nonbonded.NonbondedMethodRecord.lj_mixing`, the
        Lennard-Jones combining rule, which the record grew on 2026-09-08. In a
        schema-1 payload that key is absent and is filled with ``None`` — a
        value that field *defines* as "Lorentz–Berthelot by the then-default",
        which is what every molnex model mixed with when such a payload could
        have been written, so nothing is invented by supplying it. In a schema-2
        payload the key is required like any other: the tolerance is a property
        of the *older* layout, not a permanent softness in the codec.

        A newer version is refused rather than read leniently. The absent key of
        a schema-1 payload has a defined meaning; the fields a schema-3 payload
        might add do not, and accepting one here would mean quietly discarding
        whatever model identity it carries — the failure the version number
        exists to prevent. This is also why the field could not simply be added
        to schema 1 in place: tolerating the absence costs nothing, but a reader
        that tolerates it *and* claims to read every schema-1 payload gives a
        newer writer no way to say "you are missing something".

        Args:
            payload: A mapping produced by :meth:`to_dict` (directly or through
                JSON).

        Returns:
            The reconstructed manifest, with every nested record, enum and
            frozenset restored. Its :attr:`schema_version` is always the current
            :data:`MANIFEST_SCHEMA`: a schema-1 payload is *upgraded* on read —
            the ``None`` rule is now stated rather than implied — so writing it
            back out emits the newer layout, and the payload it came from is not
            silently re-emitted as something this molnex would read differently.

        Raises:
            ValueError: ``payload`` carries a ``schema_version`` outside
                :data:`_READABLE_SCHEMAS`. Refused rather than best-effort
                parsed: a layout this version does not know is not a manifest
                it can honestly report on.
            KeyError: A required top-level key (``profile``, ``units``,
                ``interactions``, ``nonbonded``, ``electrostatics``, ``tables``,
                ``numerics``) is absent. Only ``representation`` and
                ``provenance`` are optional, because they are optional on the
                manifest itself; an absent ``schema_version`` reads as ``None``
                and so raises the ``ValueError`` above instead.
            TypeError: A value has the wrong JSON shape (a list where an object
                belongs, or the reverse), or a *nested* record is missing a
                field. The nested case arrives as a ``TypeError`` from that
                record's own constructor rather than as a ``KeyError``, because
                the codec omits an absent key and lets the frozen record decide
                whether it had a default — which is precisely why ``lj_mixing``
                is defaulted for a schema-1 payload instead of being left out:
                all ten fields of
                :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` are
                required, so an absent key would otherwise arrive as that
                ``TypeError``. In a schema-2 payload it does arrive that way,
                like any other missing field.
        """
        version = payload.get("schema_version")
        if version not in _READABLE_SCHEMAS:
            raise ValueError(
                f"manifest schema_version {version!r} is not one this molnex reads "
                f"({', '.join(str(v) for v in _READABLE_SCHEMAS)})"
            )
        # Schema 1 has no ``lj_mixing`` key; supply the ``None`` that layout
        # implies. Schema 2 is passed through untouched, so a missing key there
        # reaches the record's constructor and raises like any other.
        nonbonded = _as_mapping(payload["nonbonded"])
        if version == 1:
            nonbonded = {"lj_mixing": None, **nonbonded}
        return cls(
            schema_version=MANIFEST_SCHEMA,
            profile=_decode_str(payload["profile"]),
            units=_decode_str(payload["units"]),
            interactions=_decode_tuple(InteractionRecord, payload["interactions"]),
            nonbonded=_decode_record(NonbondedMethodRecord, nonbonded),
            electrostatics=_decode_record(ElectrostaticsRecord, payload["electrostatics"]),
            tables=_decode_tuple(ParamTableRecord, payload["tables"]),
            numerics=_decode_record(NumericsRecord, payload["numerics"]),
            representation=_decode_optional(RepresentationRecord, payload.get("representation")),
            provenance=_decode_optional(ProvenanceRecord, payload.get("provenance")),
        )

    def write(self, path: str | Path) -> Path:
        """Write the manifest as sorted, 2-space-indented JSON.

        Sorted keys and a stable indent keep two manifests diffable, which is
        how a model change is meant to be reviewed.

        Args:
            path: Destination file.

        Returns:
            The path written to.
        """
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2))
        return out

    @classmethod
    def read(cls, path: str | Path) -> ModelManifest:
        """Read a manifest written by :meth:`write`.

        Args:
            path: File to read.

        Returns:
            The reconstructed manifest.

        Raises:
            ValueError: The file carries a different :data:`MANIFEST_SCHEMA`.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))


# --- private: recursive JSON codec over the record tree ---


def _encode(value: object) -> object:
    """One value as JSON-native data, recursively.

    Dataclasses become mappings of their fields, enums their values, sets a
    list sorted by serialised form (so an unordered ``derivatives`` set has one
    deterministic spelling), and tuples plain lists.
    """
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, frozenset | set):
        return sorted((_encode(item) for item in value), key=json.dumps)
    if isinstance(value, tuple | list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _decode(annotation: object, value: object) -> object:
    """Inverse of :func:`_encode`, driven by a field's declared type."""
    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        if value is None:
            return None
        arms = [arm for arm in get_args(annotation) if arm is not NoneType]
        if len(arms) != 1:
            raise TypeError(f"manifest codec needs one non-None union arm, got {annotation!r}")
        return _decode(arms[0], value)
    if origin is tuple:
        args = get_args(annotation)
        items = _as_sequence(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item) for item in items)
        return tuple(_decode(arg, item) for arg, item in zip(args, items, strict=True))
    if origin is frozenset or origin is set:
        (arg,) = get_args(annotation)
        decoded = (_decode(arg, item) for item in _as_sequence(value))
        return frozenset(decoded) if origin is frozenset else set(decoded)
    if origin is dict:
        _, value_type = get_args(annotation)
        return {str(key): _decode(value_type, item) for key, item in _as_mapping(value).items()}
    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return annotation(value)
        if dataclasses.is_dataclass(annotation):
            return _decode_record(annotation, value)
    return value


def _decode_record[T: DataclassInstance](cls: type[T], value: object) -> T:
    """Rebuild one frozen record from its encoded mapping."""
    payload = _as_mapping(value)
    hints = _field_types(cls)
    kwargs = {
        f.name: _decode(hints[f.name], payload[f.name])
        for f in dataclasses.fields(cls)
        if f.name in payload
    }
    return cls(**kwargs)


def _decode_optional[T: DataclassInstance](cls: type[T], value: object) -> T | None:
    """``None`` stays ``None``; anything else is a record of ``cls``."""
    return None if value is None else _decode_record(cls, value)


def _decode_tuple[T: DataclassInstance](cls: type[T], value: object) -> tuple[T, ...]:
    """A JSON list of encoded records back into a tuple of ``cls``."""
    return tuple(_decode_record(cls, item) for item in _as_sequence(value))


def _check_json_native(value: object, path: str) -> None:
    """Refuse a value the manifest payload could not carry, naming where it sits.

    The free-form record fields (:attr:`InteractionRecord.options`,
    :attr:`RepresentationRecord.spec`) are written verbatim into
    :meth:`ModelManifest.to_dict`, so they are the one place a non-JSON object
    can enter the manifest. ``str`` / ``int`` / ``float`` / ``bool`` / ``None``
    pass, as do lists and dicts of those, recursively. A tuple does not: it is
    emitted as a list and would come back as one, so a manifest carrying it
    would fail its own round trip.

    Args:
        value: The field, or one value nested inside it.
        path: Where ``value`` sits, e.g.
            ``"InteractionRecord.options['cutoff']"``, for the message.

    Raises:
        TypeError: ``value`` (or something nested in it) is not JSON-native.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_json_native(item, f"{path}[{key!r}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_native(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} is a {type(value).__name__}; a manifest carries JSON-native values "
        "only (str / int / float / bool / None, or a list or dict of those)"
    )


def _decode_str(value: object) -> str:
    """Narrow one JSON value to the string a manifest field declares."""
    if not isinstance(value, str):
        raise TypeError(f"expected a string in the manifest payload, got {type(value).__name__}")
    return value


def _as_sequence(value: object) -> Sequence[object]:
    """Narrow one JSON value to a list/tuple."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"expected a JSON list in the manifest payload, got {type(value).__name__}")
    return value


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow one JSON value to an object mapping with string keys."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"expected a JSON object in the manifest payload, got {type(value).__name__}"
        )
    return {str(key): item for key, item in value.items()}


@functools.cache
def _field_types(cls: type) -> dict[str, object]:
    """Resolved field annotations of one record type (``from __future__`` safe)."""
    return dict(get_type_hints(cls))
