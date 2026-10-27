"""Tests for :mod:`molix.deploy.metadata`."""

from __future__ import annotations

import json

import pytest

from molix.deploy import DeployMetadata
from molix.deploy import metadata as metadata_module


def _metadata(**overrides) -> DeployMetadata:
    fields = dict(
        format="aoti", cutoff=5.0, species=(8, 1, 6, 1), dtype="float32", receptive_field_layers=2
    )
    fields.update(overrides)
    return DeployMetadata(**fields)


class TestDeployMetadata:
    def test_species_are_sorted_and_deduplicated(self):
        assert _metadata().species == (1, 6, 8)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("species", ()),
            ("species", (0,)),
            ("cutoff", 0.0),
            ("receptive_field_layers", 0),
            ("receptive_field_layers", True),
            ("dtype", "float16"),
        ],
    )
    def test_invalid_fields_are_refused(self, field, value):
        with pytest.raises(ValueError, match=field):
            _metadata(**{field: value})

    def test_round_trip_and_the_written_document(self, tmp_path):
        written = _metadata(op_libraries={"molnex_equivariance": "molnex_equivariance==1"})
        path = written.write(tmp_path, "pinet")
        assert path.name == "pinet.json"
        document = json.loads(path.read_text())
        assert document["energy_unit"] == "eV" and document["length_unit"] == "angstrom"
        assert "energy_conv" not in document and "units" not in document
        assert DeployMetadata.read(path) == written

    def test_an_unknown_field_is_ignored(self, tmp_path):
        path = _metadata().write(tmp_path)
        document = json.loads(path.read_text())
        document["comment"] = "written by hand"
        path.write_text(json.dumps(document))
        assert DeployMetadata.read(path) == _metadata()


class TestOpLibrariesFor:
    def test_aten_only_package_needs_no_library(self):
        assert DeployMetadata.op_libraries_for([]) == {}

    def test_each_namespace_maps_to_the_requirement_its_library_reports(self, monkeypatch):
        monkeypatch.setattr(
            metadata_module, "_library_requirement", lambda library: f"{library}==7"
        )
        assert DeployMetadata.op_libraries_for(["molnex_equivariance", "molix"]) == {
            "molnex_opLib": "molnex_opLib==7",
            "molnex_equivariance": "molnex_equivariance==7",
        }

    def test_a_namespace_without_native_registration_is_refused(self):
        with pytest.raises(ValueError, match="no native registration"):
            DeployMetadata.op_libraries_for(["somebody_else"])
