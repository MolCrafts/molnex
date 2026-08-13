"""Tests for ChemEncoder TensorDict I/O and isolation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
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

    def test_default_n_blocks_is_five(self):
        params = inspect.signature(ChemEncoder.__init__).parameters
        assert params["n_blocks"].default == 5

    def test_default_local_attention_is_false(self):
        params = inspect.signature(ChemEncoder.__init__).parameters
        assert params["local_attention"].default is False

    def test_default_stacks_five_chem_graph_blocks(self):
        from molrep.chem.blocks import ChemGraphBlock

        enc = ChemEncoder(atom_dim=4, bond_dim=4, angle_dim=4, proper_dim=4, improper_dim=4)
        n_blocks = sum(isinstance(module, ChemGraphBlock) for module in enc.modules())
        assert n_blocks == 5

    @pytest.mark.parametrize("n_blocks", [4, 5, 6])
    def test_accepted_n_blocks_forward_feature_counts(self, mini_batch: TensorDict, n_blocks: int):
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=n_blocks,
        )
        out = enc(mini_batch)
        assert out["atoms", "chem_features"].shape == (4, 8)
        assert out["bonds", "chem_features"].shape == (3, 8)
        assert out["angles", "chem_features"].shape == (2, 8)
        assert out["propers", "chem_features"].shape == (1, 8)
        assert out["impropers", "chem_features"].shape == (1, 8)

    @pytest.mark.parametrize("width", [128, 256, 384])
    def test_recommended_sml_widths_construct(self, width: int):
        enc = ChemEncoder(atom_dim=width, bond_dim=width, n_blocks=5)
        assert enc.atom_dim == width
        assert enc.bond_dim == width

    @pytest.mark.parametrize("n_blocks", [3, 7])
    def test_n_blocks_outside_sml_raises(self, n_blocks: int):
        with pytest.raises(ValueError):
            ChemEncoder(atom_dim=4, bond_dim=4, n_blocks=n_blocks)

    def test_empty_bonds_yield_zero_bond_features(self):
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"Z": torch.tensor([8, 1], dtype=torch.long)},
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
        enc = ChemEncoder(
            atom_dim=4,
            bond_dim=4,
            angle_dim=4,
            proper_dim=4,
            improper_dim=4,
            n_blocks=5,
        )
        emb = enc.compose(batch)
        assert emb.atom.shape == (2, 4)
        assert emb.bond.shape == (0, 4)
        assert emb.angle.shape == (0, 4)
        assert emb.proper.shape == (0, 4)
        assert emb.improper.shape == (0, 4)

    def test_same_topology_different_pos_embeddings_identical(self, mini_batch: TensorDict):
        torch.manual_seed(0)
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=5,
        )
        enc.eval()
        zero_pos = enc.compose(mini_batch)
        other = mini_batch.clone()
        other["atoms", "pos"] = torch.randn_like(mini_batch["atoms", "pos"])
        rand_pos = enc.compose(other)
        assert torch.allclose(zero_pos.atom, rand_pos.atom, atol=1e-7)
        assert torch.allclose(zero_pos.bond, rand_pos.bond, atol=1e-7)
        assert torch.allclose(zero_pos.angle, rand_pos.angle, atol=1e-7)
        assert torch.allclose(zero_pos.proper, rand_pos.proper, atol=1e-7)
        assert torch.allclose(zero_pos.improper, rand_pos.improper, atol=1e-7)

    def test_reverse_bond_endpoints_features_equal(self, mini_batch: TensorDict):
        torch.manual_seed(0)
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=5,
        )
        enc.eval()
        forward = enc.compose(mini_batch)
        reversed_bonds = mini_batch.clone()
        reversed_bonds["bonds", "atomi"] = mini_batch["bonds", "atomj"].clone()
        reversed_bonds["bonds", "atomj"] = mini_batch["bonds", "atomi"].clone()
        reverse = enc.compose(reversed_bonds)
        assert torch.allclose(forward.bond, reverse.bond, rtol=1e-5, atol=1e-6)


class TestNoMolpotImport:
    """molrep.chem must not import molpot or pinet."""

    def test_source_tree_has_no_molpot(self):
        root = Path(__file__).resolve().parents[3] / "src" / "molrep" / "chem"
        assert root.is_dir()
        banned = ("molpot", "molrep.interaction.pinet")
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not any(alias.name.startswith(p) for p in banned), path
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not any(node.module.startswith(p) for p in banned), path
