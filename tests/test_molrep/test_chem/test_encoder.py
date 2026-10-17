"""Tests for ChemEncoder TensorDict I/O and isolation."""

from __future__ import annotations

import inspect

import pytest
import torch
from tensordict import TensorDict

from molrep.chem.embed import DiscreteChannel
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

    @pytest.mark.parametrize("n_blocks", [1, 4, 8])
    def test_n_blocks_is_a_count_not_a_tier(self, mini_batch: TensorDict, n_blocks: int):
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=n_blocks,
        )
        from molrep.chem.blocks import ChemGraphBlock

        assert sum(isinstance(module, ChemGraphBlock) for module in enc.modules()) == n_blocks
        out = enc(mini_batch)
        assert out["atoms", "chem_features"].shape == (4, 8)
        assert out["bonds", "chem_features"].shape == (3, 8)

    def test_negative_n_blocks_raises(self):
        with pytest.raises(ValueError, match="n_blocks"):
            ChemEncoder(atom_dim=4, bond_dim=4, n_blocks=-1)

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

    def test_features_stay_unit_scale_however_the_trunk_weights_grow(self, mini_batch: TensorDict):
        """A head reads unit-scale features even after the trunk's weights blow up.

        Without the norms a Rondo hessian stage grew the feature RMS from ~1 to
        1e3-1e4 and drove every head into its flat region (zero gradient).
        """
        torch.manual_seed(0)
        enc = ChemEncoder(
            atom_dim=8, bond_dim=8, angle_dim=8, proper_dim=8, improper_dim=8, n_blocks=4
        )
        with torch.no_grad():
            for module in enc.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.mul_(100.0)
        emb = enc.compose(mini_batch)
        for name in ("atom", "bond", "angle", "proper", "improper"):
            rms = getattr(emb, name).pow(2).mean(dim=-1).sqrt()
            assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), name

    def test_padded_valence_rows_stay_zero_after_the_norm(self, mini_batch: TensorDict):
        """LayerNorm of a zero row is its bias; the mask must still win."""
        torch.manual_seed(0)
        enc = ChemEncoder(atom_dim=8, bond_dim=8, angle_dim=8, proper_dim=8, improper_dim=8)
        with torch.no_grad():
            enc.bond_norm.bias.fill_(1.0)
            enc.angle_norm.bias.fill_(1.0)
        batch = mini_batch.clone()
        batch["bonds", "mask"] = torch.tensor([True, True, False])
        batch["angles", "mask"] = torch.tensor([True, False])
        emb = enc.compose(batch)
        assert torch.all(emb.bond[2] == 0.0)
        assert torch.all(emb.angle[1] == 0.0)
        assert emb.bond[0].abs().sum() > 0.0


class TestChemEncoderChannels:
    """Declared integer columns change the features; an undeclared batch fails closed."""

    @staticmethod
    def _encoder(**channels) -> ChemEncoder:
        return ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=1,
            **channels,
        )

    def test_bond_order_changes_the_bond_features(self, mini_batch: TensorDict):
        torch.manual_seed(0)
        enc = self._encoder(bond_channels=(DiscreteChannel(name="bond_type", num_classes=5),))
        single = enc.compose(mini_batch).bond
        double = mini_batch.clone()
        double["bonds", "bond_type"] = torch.tensor([2, 1, 1])
        assert not torch.allclose(enc.compose(double).bond[0], single[0])
        torch.testing.assert_close(enc.compose(double).bond[1:], single[1:])

    def test_a_formal_charge_changes_the_atom_features(self, mini_batch: TensorDict):
        torch.manual_seed(0)
        enc = self._encoder(
            atom_channels=(DiscreteChannel(name="formal_charge", num_classes=5, offset=2),)
        )
        neutral = mini_batch.clone()
        neutral["atoms", "formal_charge"] = torch.zeros(4, dtype=torch.long)
        charged = mini_batch.clone()
        charged["atoms", "formal_charge"] = torch.tensor([-1, 0, 0, 0])
        assert not torch.allclose(enc.compose(charged).atom[0], enc.compose(neutral).atom[0])

    def test_a_per_graph_channel_is_read_from_graphs(self, mini_batch: TensorDict):
        torch.manual_seed(0)
        enc = self._encoder(
            atom_channels=(
                DiscreteChannel(name="total_charge", num_classes=11, offset=5, per="graph"),
            )
        )
        batch = mini_batch.clone()
        batch["graphs"] = TensorDict({"total_charge": torch.tensor([1.0])}, batch_size=[1])
        other = batch.clone()
        other["graphs", "total_charge"] = torch.tensor([0.0])
        assert not torch.allclose(enc.compose(batch).atom, enc.compose(other).atom)
        assert ("graphs", "total_charge") in enc.in_keys
        assert ("atoms", "batch") in enc.in_keys

    def test_a_missing_declared_column_is_refused_by_name(self, mini_batch: TensorDict):
        enc = self._encoder(bond_channels=(DiscreteChannel(name="in_ring", num_classes=2),))
        with pytest.raises(KeyError, match="bonds.in_ring"):
            enc.compose(mini_batch)

    def test_in_keys_list_the_declared_columns(self):
        enc = self._encoder(
            atom_channels=(DiscreteChannel(name="formal_charge", num_classes=5, offset=2),),
            bond_channels=(DiscreteChannel(name="bond_type", num_classes=5),),
        )
        assert enc.in_keys == [("atoms", "Z"), ("atoms", "formal_charge"), ("bonds", "bond_type")]


class TestChemEncoderTermBonds:
    """Every valence context reads the features of the term's own bonds."""

    @staticmethod
    def _cyclobutadiene() -> TensorDict:
        """A four-ring of carbons with alternating single / double bonds.

        Vertex-transitive with the bond types: every atom has one single and
        one double bond, so the atom features are identical, and only the
        bond features tell the torsion about the double bond (1-2) from the
        one about the single bond (2-3).
        """
        n = 4
        return TensorDict(
            {
                "atoms": TensorDict({"Z": torch.full((n,), 6, dtype=torch.long)}, batch_size=[n]),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                        "atomj": torch.tensor([1, 2, 3, 0], dtype=torch.long),
                        "bond_type": torch.tensor([1, 2, 1, 2], dtype=torch.long),
                    },
                    batch_size=[4],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1], dtype=torch.long),
                        "atomj": torch.tensor([1, 2], dtype=torch.long),
                        "atomk": torch.tensor([2, 3], dtype=torch.long),
                        "atoml": torch.tensor([3, 0], dtype=torch.long),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )

    def test_a_torsion_knows_its_central_bond_order(self):
        torch.manual_seed(11)
        enc = ChemEncoder(
            atom_dim=8,
            bond_dim=8,
            angle_dim=8,
            proper_dim=8,
            improper_dim=8,
            n_blocks=2,
            bond_channels=(DiscreteChannel(name="bond_type", num_classes=5),),
        )
        emb = enc.compose(self._cyclobutadiene())
        # The ring is vertex-transitive: no atom feature can tell the torsions apart.
        assert torch.allclose(emb.atom, emb.atom[:1].expand_as(emb.atom), atol=1e-5)
        assert not torch.allclose(emb.proper[0], emb.proper[1], atol=1e-5)

    def test_a_term_without_bonds_is_refused_by_name(self):
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"Z": torch.tensor([1, 8, 1], dtype=torch.long)}, batch_size=[3]
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        enc = ChemEncoder(atom_dim=4, bond_dim=4, angle_dim=4, proper_dim=4, improper_dim=4)
        with pytest.raises(KeyError, match="angles but no bonds"):
            enc.compose(batch)
