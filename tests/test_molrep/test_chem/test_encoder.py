"""Tests for ChemEncoder TensorDict I/O and isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from tensordict import TensorDict

from molrep.chem.encoder import ChemEncoder
from molrep.chem.features import ChemEmbeddings


class TestChemEncoder:
    """ChemEncoder reads valence namespaces and writes chem features."""

    def test_forward_writes_keys(self, mini_batch: TensorDict):
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            num_elements=20,
        )
        out = enc(mini_batch)
        assert ("atoms", "chem_features") in out.keys(include_nested=True)
        assert ("bonds", "chem_features") in out.keys(include_nested=True)
        assert ("angles", "chem_features") in out.keys(include_nested=True)
        assert ("propers", "chem_features") in out.keys(include_nested=True)
        assert ("impropers", "chem_features") in out.keys(include_nested=True)

        assert out["atoms", "chem_features"].shape == (4, 8)
        assert out["bonds", "chem_features"].shape == (3, 8)
        assert out["angles", "chem_features"].shape == (2, 8)
        assert out["propers", "chem_features"].shape == (1, 8)
        assert out["impropers", "chem_features"].shape == (1, 8)

    def test_compose_returns_chem_embeddings(self, mini_batch: TensorDict):
        enc = ChemEncoder(atom_dim=8, bond_dim=6, angle_dim=6, proper_dim=6, improper_dim=6)
        emb = enc.compose(mini_batch)
        assert isinstance(emb, ChemEmbeddings)
        assert emb.atom.shape == (4, 8)
        assert emb.bond.shape == (3, 6)
        assert emb.angle.shape == (2, 6)
        assert emb.proper.shape == (1, 6)
        assert emb.improper.shape == (1, 6)

    def test_embeddings_view(self, mini_batch: TensorDict):
        enc = ChemEncoder(atom_dim=8, bond_dim=8, angle_dim=8, proper_dim=8, improper_dim=8)
        out = enc(mini_batch)
        viewed = enc.embeddings(out)
        assert torch.allclose(viewed.atom, out["atoms", "chem_features"])
        assert torch.allclose(viewed.bond, out["bonds", "chem_features"])

    def test_write_batch(self, mini_batch: TensorDict):
        enc = ChemEncoder(atom_dim=8, bond_dim=8, angle_dim=8, proper_dim=8, improper_dim=8)
        emb = enc.compose(mini_batch)
        written = enc.write_batch(mini_batch, emb)
        assert written is mini_batch or isinstance(written, TensorDict)
        assert written["atoms", "chem_features"].shape[0] == 4

    def test_bond_index_coo_path(self):
        """Bonds may carry COO bond_index (2, N) instead of atomi/atomj."""
        n = 3
        # bond_index contract: COO (2, N_bonds)
        bond_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"Z": torch.tensor([6, 8, 1], dtype=torch.long)},
                    batch_size=[n],
                ),
                "bonds": TensorDict(
                    {"bond_index": bond_index},
                    batch_size=[],
                ),
            },
            batch_size=[],
        )
        enc = ChemEncoder(atom_dim=4, bond_dim=4, angle_dim=4, proper_dim=4, improper_dim=4)
        out = enc(batch)
        assert out["bonds", "chem_features"].shape == (2, 4)

    def test_missing_optional_namespaces_empty(self):
        """Missing angles/propers/impropers write empty (0, D) features."""
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"Z": torch.tensor([1, 1], dtype=torch.long)},
                    batch_size=[2],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        enc = ChemEncoder(atom_dim=4, bond_dim=4, angle_dim=5, proper_dim=5, improper_dim=5)
        emb = enc.compose(batch)
        assert emb.angle.shape == (0, 5)
        assert emb.proper.shape == (0, 5)
        assert emb.improper.shape == (0, 5)

    def test_no_energy_keys(self, mini_batch: TensorDict):
        enc = ChemEncoder(atom_dim=4, bond_dim=4, angle_dim=4, proper_dim=4, improper_dim=4)
        out = enc(mini_batch)
        flat_keys = {str(k) for k in out.keys(include_nested=True)}
        assert not any("energy" in k for k in flat_keys)
        assert not any("force" in k for k in flat_keys)


class TestNoMolpotImport:
    """molrep.chem must not import molpot."""

    def test_source_tree_has_no_molpot(self):
        root = Path(__file__).resolve().parents[3] / "src" / "molrep" / "chem"
        assert root.is_dir()
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molpot"), path
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("molpot"), path
