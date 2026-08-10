"""Tests for molzoo.chem ChemPerception recipe."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from tensordict import TensorDict

from molzoo.chem import ChemPerception, ChemPerceptionSpec
from molzoo.chem.encoder import ChemPerception as ChemPerceptionDirect


class TestChemPerceptionSpec:
    def test_defaults(self):
        spec = ChemPerceptionSpec()
        assert spec.atom_dim > 0
        assert spec.bond_dim > 0
        assert spec.num_elements > 0


class TestChemPerception:
    def test_constructs_and_forwards(self):
        model = ChemPerception(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            num_elements=20,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "Z": torch.tensor([8, 1, 1], dtype=torch.long),
                        "batch": torch.zeros(3, dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 0], dtype=torch.long),
                        "atomj": torch.tensor([1, 2], dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([1], dtype=torch.long),
                        "atomj": torch.tensor([0], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        out = model(batch)
        assert out["atoms", "chem_features"].shape == (3, 8)
        assert out["bonds", "chem_features"].shape == (2, 8)
        assert out["angles", "chem_features"].shape == (1, 8)
        # No energy keys from a pure perception recipe
        nested = {str(k) for k in out.keys(include_nested=True)}
        assert not any("energy" in k for k in nested)

    def test_from_spec(self):
        spec = ChemPerceptionSpec(
            atom_dim=4,
            bond_dim=4,
            angle_dim=4,
            proper_dim=4,
            improper_dim=4,
            num_elements=10,
        )
        model = ChemPerception(spec=spec)
        assert model.config.atom_dim == 4
        assert isinstance(model, ChemPerceptionDirect)

    def test_lazy_export_from_molzoo(self):
        import molzoo

        assert "ChemPerception" in molzoo.__all__
        assert "ChemPerceptionSpec" in molzoo.__all__
        # Attribute access resolves via lazy table
        assert molzoo.ChemPerception is ChemPerception
        assert molzoo.ChemPerceptionSpec is ChemPerceptionSpec


class TestNoMolpotImport:
    def test_source_tree_has_no_molpot(self):
        root = Path(__file__).resolve().parents[3] / "src" / "molzoo" / "chem"
        assert root.is_dir()
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molpot"), path
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("molpot"), path
