"""ModelManifest and its records — mirrors ``src/molpot/manifest.py``.

The manifest is the S2 deliverable: one versioned, JSON-round-trippable
description of *which model this is* — representation and checkpoint, enabled
interaction terms with their contracts, the nonbonded method, electrostatics
and its solver, content hashes of the parameter tables, numerical precision,
and provenance. It is a composition of small frozen records, never a god
struct: this module tests those records and the aggregate's serialisation,
nothing else. Who *fills* a manifest from a force field is
``ForceField.manifest`` and is tested in
``tests/test_molpot/test_forcefield/test_container.py``; who supplies the
representation is ``Parameterizer.representation_record`` and is tested in
``test_forcefield/test_parameterizer.py``. Every manifest here is therefore
built inline from the records themselves, so this file stays a leaf test of a
leaf module.

**Hard-coded golden — ``BONDS_SHA256``.** The SHA-256 of a fixed two-parameter
fp64 table, ``k = [1.0, 2.0]`` then ``r0 = [1.5, 1.6]``, hashed as the
concatenated little-endian IEEE-754 binary64 bytes of the tables in ``names``
order (32 bytes, hex
``000000000000f03f 0000000000000040 000000000000f83f 9a9999999999f93f``).
Computed 2026-09-07 on python 3.12 / torch 2.14.0+cpu with::

    python - <<'EOF'
    import hashlib, torch
    tables = {"k": torch.tensor([1.0, 2.0], dtype=torch.float64),
              "r0": torch.tensor([1.5, 1.6], dtype=torch.float64)}
    payload = b"".join(
        tables[n].detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
        for n in sorted(tables)
    )
    print(hashlib.sha256(payload).hexdigest())
    EOF

The literal pins the *hash definition* (fp64 bytes, ``names`` order, no
shape / name / dtype in the digest), which is what makes two runs of the same
trained parameterizer comparable across machines. It is a pure function of
those 32 bytes and needs no third-party oracle.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table);
    spec ``.claude/specs/composable-a0-04-manifest.md``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from molix.config import config
from molpot.contract import (
    Contribution,
    CutoffKind,
    Derivative,
    ExclusionClass,
    InteractionContract,
    TermSpec,
)
from molpot.forcefield.schema import ParamSpec
from molpot.ir import ExclusionScales, NonbondedScaling
from molpot.ir.nonbonded import NonbondedMethodRecord
from molpot.manifest import (
    MANIFEST_SCHEMA,
    ElectrostaticsRecord,
    InteractionRecord,
    ModelManifest,
    NumericsRecord,
    ParamTableRecord,
    ProvenanceRecord,
    RepresentationRecord,
    SolverRecord,
)

#: SHA-256 of ``{"k": [1.0, 2.0], "r0": [1.5, 1.6]}`` as fp64 bytes in ``names``
#: order — hard-coded golden, provenance in the module docstring.
BONDS_SHA256 = "82aa61493826e3643c33ffe09c6caa9be2e66b93e190deda6665f6adefa9d40a"


def _bonds_tables() -> dict[str, torch.Tensor]:
    """The fixed fp64 table :data:`BONDS_SHA256` was computed from."""
    return {
        "k": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "r0": torch.tensor([1.5, 1.6], dtype=torch.float64),
    }


def _without_lj_mixing(nonbonded: object) -> dict[str, object]:
    """An encoded nonbonded record with the combining rule removed.

    What a payload written before the field existed looks like — the key is
    absent, not ``None``.
    """
    assert isinstance(nonbonded, dict)
    return {str(key): value for key, value in nonbonded.items() if key != "lj_mixing"}


def _manifest() -> ModelManifest:
    """A fully populated manifest — every optional record present.

    Deliberately richer than what ``ForceField.manifest`` emits (which leaves
    ``representation`` / ``provenance`` / ``solver`` to the caller), so the
    round trip covers every record type, both enums and the unordered
    ``derivatives`` frozenset.

    ``nonbonded.lj_mixing`` is the OPLS-AA ``"geometric"`` rather than the
    ``"lorentz-berthelot"`` default, so a codec that dropped the Lennard-Jones
    combining rule — or defaulted it back — fails the round trip instead of
    agreeing by coincidence. It rides inside the nested ``nonbonded`` record,
    which is why carrying it needs no ``MANIFEST_SCHEMA`` bump.
    """
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA,
        profile="A0",
        units="real",
        interactions=(
            InteractionRecord(
                style="pair:lj/cut",
                contract=InteractionContract(
                    inputs=("atoms.pos", "atoms.epsilon", "atoms.sigma"),
                    cutoff=CutoffKind.SWITCHED,
                    exclusion=ExclusionClass.PERMANENT,
                    derivatives=frozenset(
                        {Derivative.FORCES, Derivative.VIRIAL, Derivative.HESSIAN}
                    ),
                    terms=(
                        TermSpec("lj_repulsion", Contribution.LOCAL),
                        TermSpec("lj_dispersion", Contribution.DISP),
                    ),
                ),
                params=(
                    ParamSpec("epsilon", "[n_atom_types]", "kcal/mol", "well depth"),
                    ParamSpec("sigma", "[n_atom_types]", "Å", "length scale"),
                ),
                options={"cutoff": 12.0, "switch_start": 10.0},
            ),
        ),
        nonbonded=NonbondedMethodRecord(
            lj_cutoff=12.0,
            lj_switch_start=10.0,
            lj_mixing="geometric",
            coulomb_cutoff=None,
            coulomb_calculator="pme",
            real_space_cutoff=15.0,
            smearing=5.0,
            lr_wavelength=1.0,
            mesh_spacing=1.0,
            capacity_factor=1.0,
        ),
        electrostatics=ElectrostaticsRecord(
            multipole_order=0,
            polarization_scheme=None,
            damping=None,
            charge_constraint="per_graph_total",
            total_charge=0.0,
            permanent_scaling=NonbondedScaling(
                q=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            ),
            induced_scaling=None,
            solver=SolverRecord(
                kind="eem", iterations=64, tolerance=1e-10, preconditioner="diagonal"
            ),
        ),
        tables=(
            ParamTableRecord(
                category="bonds",
                names=("k", "r0"),
                shapes=((2,), (2,)),
                dtype="torch.float64",
                sha256=BONDS_SHA256,
            ),
        ),
        numerics=NumericsRecord(
            dtype="torch.float64", matmul_precision="highest", energy_dtype=None
        ),
        representation=RepresentationRecord(
            family="graph-parameterizer",
            encoder="PiNet",
            checkpoint_id=None,
            spec={"depth": 3},
        ),
        provenance=ProvenanceRecord(
            reference_method="wB97X-D/def2-TZVPD",
            energy_offsets_id=None,
            data_version="v0",
            software={"molnex": "0.1.0"},
            checkpoint_id=None,
            run_id="exp-0001",
            created_at="2026-09-07T00:00:00Z",
            git_sha=None,
        ),
    )


class TestModelManifest:
    def test_schema_version_is_two(self) -> None:
        """Pinned literal: bumping it is a breaking change, not a refactor.

        Went to ``2`` on 2026-09-08 with
        :attr:`~molpot.ir.nonbonded.NonbondedMethodRecord.lj_mixing`, the
        Lennard-Jones combining rule. The field is *required* on the record, so
        every schema-2 payload states the rule and a reader never has to guess
        which force field a manifest describes — which is the whole point of
        recording it. That is exactly what a version bump is for: a schema-1
        payload is still readable (below), but it is honestly labelled as
        coming from before the field existed rather than being retro-fitted
        into the current layout.

        Versioned independently of ``molix.engine.export.LAMMPS_META_SCHEMA``.
        """
        assert MANIFEST_SCHEMA == 2

    def test_from_dict_inverts_to_dict(self) -> None:
        """Exact inverse — enums, frozensets and tuples all come back."""
        manifest = _manifest()
        assert ModelManifest.from_dict(manifest.to_dict()) == manifest

    def test_to_dict_is_json_native_with_values_for_enums_and_sorted_derivatives(
        self,
    ) -> None:
        """No enum / frozenset / tuple survives into the payload.

        The unordered ``derivatives`` set is emitted sorted so the same model
        always serialises to the same bytes — the determinism a content hash
        of the manifest would rest on.
        """
        payload = _manifest().to_dict()
        contract = payload["interactions"][0]["contract"]
        assert isinstance(payload["interactions"], list)
        assert contract["cutoff"] == "switched"
        assert contract["exclusion"] == "permanent"
        assert contract["derivatives"] == ["forces", "hessian", "virial"]
        assert contract["terms"][0]["contribution"] == "local"
        assert payload["interactions"][0]["params"][0]["frame"] == "scalar"
        json.dumps(payload)

    def test_read_inverts_write(self, tmp_path: Path) -> None:
        """Round trip through a file, and ``write`` reports where it wrote."""
        manifest = _manifest()
        path = manifest.write(tmp_path / "m.json")
        assert path == tmp_path / "m.json"
        assert ModelManifest.read(path) == manifest

    def test_write_emits_sorted_indented_json(self, tmp_path: Path) -> None:
        """Sorted keys + 2-space indent: manifests are meant to be diffed."""
        manifest = _manifest()
        path = manifest.write(tmp_path / "m.json")
        assert path.read_text() == json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
        # The version is *in* the file: it is what a reader branches on before
        # it may interpret a single other key.
        assert json.loads(path.read_text())["schema_version"] == 2

    def test_is_frozen(self) -> None:
        """A manifest describes one model; rewriting a field in place lies."""
        manifest = _manifest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.profile = "B1"  # ty: ignore[invalid-assignment]

    def test_a_schema_one_payload_reads_back_with_no_declared_mixing_rule(self) -> None:
        """The previous layout is still readable, and honestly so.

        A schema-1 manifest was written before the combining rule was recorded
        at all, so the only true thing to say about its rule is "none was
        declared" — ``None``, which the field documents as "Lorentz–Berthelot by
        the then-default", the rule every molnex model mixed with until
        2026-09-08. Refusing the payload would strand every manifest already on
        disk; inventing ``"lorentz-berthelot"`` would claim the run *declared* a
        rule it never did, and two runs would then compare equal on a field
        neither of them recorded.
        """
        payload = _manifest().to_dict()
        payload["schema_version"] = 1
        payload["nonbonded"] = _without_lj_mixing(payload["nonbonded"])
        assert ModelManifest.from_dict(payload).nonbonded.lj_mixing is None

    def test_a_schema_two_payload_without_the_mixing_rule_is_defective(self) -> None:
        """Tolerance is for the *old* layout, not for a hole in the current one.

        A payload labelled schema 2 claims to have been written by a molnex
        that records the rule, so a missing key is a truncated or hand-edited
        file rather than an older one. Decoding it to ``None`` would give it
        the same reading as a genuine schema-1 manifest and quietly erase the
        distinction the version bump exists to draw.
        """
        payload = _manifest().to_dict()
        payload["nonbonded"] = _without_lj_mixing(payload["nonbonded"])
        with pytest.raises(TypeError):
            ModelManifest.from_dict(payload)

    def test_from_dict_refuses_a_schema_version_this_molnex_cannot_read(self) -> None:
        """A layout this version does not know is not one it may parse.

        The refusal names the offending version, so the payload can be routed
        to the molnex that wrote it instead of being best-effort decoded into
        a manifest that describes a model nobody built.
        """
        payload = {**_manifest().to_dict(), "schema_version": 99}
        with pytest.raises(ValueError, match="99"):
            ModelManifest.from_dict(payload)


class TestParamTableRecord:
    def test_sha256_of_the_fixed_fp64_table_is_the_golden(self) -> None:
        """Hard-coded golden; the table is rebuilt from scratch each call.

        Also pins the descriptive fields: ``names`` in the order the bytes
        were concatenated, one shape per name, and ``dtype`` spelled
        ``str(torch.float64)``.
        """
        record = ParamTableRecord.from_tables("bonds", _bonds_tables())
        assert record.category == "bonds"
        assert record.names == ("k", "r0")
        assert record.shapes == ((2,), (2,))
        assert record.dtype == "torch.float64"
        assert record.sha256 == BONDS_SHA256
        assert ParamTableRecord.from_tables("bonds", _bonds_tables()) == record

    def test_changing_one_entry_changes_only_that_tables_sha256(self) -> None:
        """Per-category content hash: one edited force constant is localised."""
        atoms = {"epsilon": torch.tensor([0.066], dtype=torch.float64)}
        edited = _bonds_tables()
        edited["k"][1] = 2.5
        assert ParamTableRecord.from_tables("bonds", edited).sha256 != BONDS_SHA256
        assert (
            ParamTableRecord.from_tables("atoms", atoms).sha256
            == ParamTableRecord.from_tables("atoms", dict(atoms)).sha256
        )

    def test_sha256_hashes_the_fp64_bytes_of_a_float32_table(self) -> None:
        """The digest is precision-independent: fp32 tables are up-cast first.

        Values are exactly representable in binary32, so the up-cast is exact
        and the two tables are the same numbers — a digest taken over the raw
        fp32 bytes would disagree.
        """
        values = ([1.0, 2.0], [1.5, 1.75])
        fp32 = {
            "k": torch.tensor(values[0], dtype=torch.float32),
            "r0": torch.tensor(values[1], dtype=torch.float32),
        }
        fp64 = {
            "k": torch.tensor(values[0], dtype=torch.float64),
            "r0": torch.tensor(values[1], dtype=torch.float64),
        }
        assert (
            ParamTableRecord.from_tables("bonds", fp32).sha256
            == ParamTableRecord.from_tables("bonds", fp64).sha256
        )


class TestInteractionRecord:
    def test_a_non_json_native_option_is_refused_at_construction(self) -> None:
        """Style options are payload, so a tensor is caught where it enters.

        ``options`` is written verbatim into ``to_dict()``. A tensor there
        survives every equality and round-trip check in memory and only fails
        at ``write`` time, in a caller far from the ``def_pairstyle`` that
        introduced it — by which point the manifest has already been compared
        against another run's. The record refuses it on construction instead.
        """
        contract = _manifest().interactions[0].contract
        with pytest.raises((TypeError, ValueError)):
            InteractionRecord(
                style="pair:lj/cut",
                contract=contract,
                options={"cutoff": torch.tensor(1.0)},
            )


class TestNumericsRecord:
    def test_from_runtime_reads_the_process_wide_precision(self) -> None:
        """The precision a model runs at has exactly two sources, both read here.

        ``molix.config["ftype"]`` is the global parameter / batch dtype.
        ``matmul_precision`` is **not** a config key despite the name in the
        S2 record table: ``MolnexConfig.set_precision`` skips it when copying
        the preset and calls ``torch.set_float32_matmul_precision`` instead,
        so the process-wide policy only ever lives in torch and
        ``config["matmul_precision"]`` raises ``KeyError``. The effective
        policy is what a manifest has to record.
        """
        record = NumericsRecord.from_runtime()
        assert record.dtype == str(config["ftype"])
        assert record.matmul_precision == torch.get_float32_matmul_precision()
