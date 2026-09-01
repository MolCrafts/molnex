"""Tests for MacePolar and polar_remap."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from molix import config
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar
from molzoo.mace.spec import MACEPolarSpec
from tests.test_molzoo.test_mace.conftest import polar_kwargs


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


polar_remap = CheckpointRemap(polar_keys, on_unexpected="return")


class TestMacePolar:
    """Isolated-molecule Polar energy on the tiny Polar backbone."""

    def test_official_submodule_names(self, tiny_polar_spec: MACEPolarSpec):
        """Keys polar_remap copies by name."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec, compute_forces=False))
        names = set(model._modules)
        assert {
            "lr_source_maps",
            "layer_feature_mixer",
            "fukui_source_map",
            "field_dependent_charges_maps",
            "local_electron_energy",
            "electric_potential_descriptor",
            "coulomb_energy",
        } <= names
        assert len(model.lr_source_maps) == tiny_polar_spec.num_interactions
        assert len(model.field_dependent_charges_maps) == tiny_polar_spec.num_recursion_steps

    def test_energy_is_finite_and_per_graph(self, tiny_polar_spec: MACEPolarSpec, omol_cluster):
        """Neutral singlet cluster: one finite eV."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec, compute_forces=False))
        out = model(omol_cluster)
        energy = out["graphs", "energy"]
        assert energy.shape == (1,)
        assert torch.isfinite(energy).all()

    def test_energy_is_translation_invariant(self, tiny_polar_spec: MACEPolarSpec, omol_cluster):
        """Isolated electrostatics + local MACE: E(r+Δ)=E(r)."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec, compute_forces=False))
        e0 = model(omol_cluster.clone())["graphs", "energy"].clone()
        shifted = omol_cluster.clone()
        shifted["atoms", "pos"] = shifted["atoms", "pos"] + 2.5
        e1 = model(shifted)["graphs", "energy"]
        assert torch.allclose(e0, e1, atol=1e-8)

    def test_defaults_total_charge_and_spin(self, tiny_polar_spec: MACEPolarSpec, cluster):
        """Missing graphs.total_* → Q=0, M=1."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec, compute_forces=False))
        energy = model(cluster)["graphs", "energy"]
        assert energy.shape == (1,)
        assert torch.isfinite(energy).all()


class TestPOLARRemap:
    """Official-dialect keys land on MacePolar parameters."""

    def test_synthetic_state_fills_parameters(self, tiny_polar_spec: MACEPolarSpec):
        """A native-key dump plus one remapped nested weight loads without unfilled params."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec, compute_forces=False))
        native = {k: v.detach().clone() for k, v in model.state_dict().items()}
        official = {}
        for key, value in native.items():
            if key.endswith("readout.linear_mid.linear.weight"):
                official[key.replace(".linear.weight", ".weight")] = value
            elif key.endswith("readout.linear_2.linear.weight"):
                official[key.replace(".linear.weight", ".weight")] = value
            elif key.endswith("linear_up_q.linear.weight"):
                official["local_electron_energy.linear_up_q.weight"] = value
            elif key.endswith("linear_up_v.linear.weight"):
                official["local_electron_energy.linear_up_v.weight"] = value
            else:
                official[key] = value
        official["kspace_cutoff"] = torch.tensor(1.0)
        official["aux_head.weight"] = torch.ones(1)
        missing, unexpected = polar_remap.load(model, official)
        assert not any(k in dict(model.named_parameters()) for k in missing)
        assert "aux_head.weight" in unexpected


class TestPolarPackageImports:
    def test_polar_potential_does_not_import_forbidden_stacks(self):
        path = Path(__file__).resolve().parents[3] / "src" / "molzoo" / "mace" / "polar.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = {"e3nn", "mace", "ase", "graph_longrange"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in banned for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned
