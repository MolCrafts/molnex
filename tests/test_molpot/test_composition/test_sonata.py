"""Unit tests for :mod:`molpot.composition.sonata` — construction and the written columns.

The physics (invariances, finite differences, batch independence) lives in
the sibling ``test_sonata_*.py`` files on the float64 pipeline; this file
checks what the constructor refuses, which columns one forward writes and
that two models built from one spec share a state dict.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.schema import ENERGY_KEY, FORCES_KEY
from molpot import Polarization
from molpot.composition import Sonata, SonataSpec
from molpot.heads import EdgeEnergyHead, PermMultipoleHeadSpec
from molpot.potentials import EwaldMultipoleEnergySpec
from molzoo.allegro import AllegroEncoder
from tests.test_molpot.test_composition.conftest import (
    AVG_NUM_NEIGHBORS,
    allegro_encoder,
    full_bidirectional_edges,
    make_batch,
    sonata_spec,
)


@pytest.fixture
def encoder() -> AllegroEncoder:
    return allegro_encoder(dtype=torch.float32)


def _batch() -> TensorDict:
    """B=2 graphs, N=8 atoms, E=24 edges (full bidirectional within each graph), float32."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.7, 1.2, 0.0],
            [-0.7, 1.2, 0.0],
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 1.4, 0.0],
            [1.4, 1.4, 0.0],
        ],
        dtype=torch.float32,
    )
    return make_batch(
        pos=pos,
        Z=torch.tensor([1, 6, 8, 7, 1, 6, 8, 7], dtype=torch.long),
        pair_index=full_bidirectional_edges([(0, 4), (4, 4)]),
        batch_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        total_charge=torch.zeros(2, dtype=torch.float32),
    )


def _short_range_head(encoder: AllegroEncoder) -> EdgeEnergyHead:
    return EdgeEnergyHead(
        input_dim=encoder.output_dim,
        hidden_dim=32,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        out_key="energy_short",
        dtype=torch.float32,
    )


class TestSonata:
    def test_an_encoder_without_a_tensor_track_is_refused(self) -> None:
        encoder = allegro_encoder(dtype=torch.float32, expose_tensor_track=False)
        with pytest.raises(ValueError, match="expose_tensor_track"):
            Sonata(sonata_spec(encoder), encoder, dtype=torch.float32)

    def test_dipoles_need_an_encoder_with_l_max_two(self) -> None:
        encoder = allegro_encoder(dtype=torch.float32, l_max=1)
        with pytest.raises(ValueError, match="l_max"):
            Sonata(sonata_spec(encoder), encoder, dtype=torch.float32)

    def test_the_head_width_must_be_the_encoders(self, encoder: AllegroEncoder) -> None:
        spec = SonataSpec(
            head=PermMultipoleHeadSpec(input_dim=encoder.output_dim + 1),
            ewald=EwaldMultipoleEnergySpec(),
        )
        with pytest.raises(ValueError, match="input_dim"):
            Sonata(spec, encoder, dtype=torch.float32)

    def test_polarization_is_refused_among_the_short_range_heads(
        self, encoder: AllegroEncoder
    ) -> None:
        with pytest.raises(ValueError, match="double-count"):
            Sonata(
                sonata_spec(encoder),
                encoder,
                short_range_heads=[_short_range_head(encoder), Polarization()],
                dtype=torch.float32,
            )

    def test_forward_writes_the_energy_terms_the_moments_and_the_forces(
        self, encoder: AllegroEncoder
    ) -> None:
        sonata = Sonata(
            sonata_spec(encoder),
            encoder,
            short_range_heads=[_short_range_head(encoder)],
            dtype=torch.float32,
        )
        out = sonata(_batch())
        assert {
            "energy",
            "energy_short",
            "energy_es",
            "molecular_dipole",
            "charge_sum_pre_proj",
            "charge_sum_post_proj",
        } <= set(out["graphs"].keys())
        assert out[ENERGY_KEY].shape == (2,)
        assert out["graphs", "energy_short"].shape == (2,)
        assert out["graphs", "energy_es"].shape == (2,)
        assert out["graphs", "molecular_dipole"].shape == (2, 3)
        assert out["atoms", "atomic_charges"].shape == (8,)
        assert out["atoms", "atomic_dipoles"].shape == (8, 3)
        assert out["atoms", "atomic_quadrupoles"].shape == (8, 5)
        assert out["atoms", "phi"].shape == (8,)
        assert out["atoms", "field"].shape == (8, 3)
        assert out[FORCES_KEY].shape == (8, 3)
        torch.testing.assert_close(
            out[ENERGY_KEY], out["graphs", "energy_short"] + out["graphs", "energy_es"]
        )

    def test_two_models_from_one_spec_share_a_state_dict(self, encoder: AllegroEncoder) -> None:
        first = Sonata(sonata_spec(encoder), encoder, dtype=torch.float32)
        second = Sonata(first.spec, encoder, dtype=torch.float32)
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            out_first = first(_batch())
            out_second = second(_batch())
        torch.testing.assert_close(out_second[ENERGY_KEY], out_first[ENERGY_KEY])
        torch.testing.assert_close(
            out_second["atoms", "atomic_charges"], out_first["atoms", "atomic_charges"]
        )
