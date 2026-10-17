"""Unit tests for `molpot.composition.sonata` — Sonata composer.

Sub-spec 01 of the Sonata model line. Tests cover:

* `Sonata.from_encoder` returns a wired `nn.Module` with the right
  sub-module types.
* `Sonata.__init__` refuses `kappa_head=`, `alpha_head=`,
  `induced_*` kwargs (future `LesPolarizable` composer territory).
* `short_range_head` containing a `Polarization` instance is
  refused (would double-count induction).
* `Sonata.forward` returns the documented output dict (key set,
  tensor shapes).
* `total = short + es` energy decomposition holds.
* `compute_forces=True` adds a finite `(N, 3)` forces tensor.
* `compute_stress=True` adds a symmetric `(B, 3, 3)` stress.
* `Sonata.from_spec(sonata.config, encoder)` round-trips when
  state_dict is transferred.
* `Sonata.from_encoder` validates `encoder.expose_tensor_track`
  and `encoder.l_max`.
* `Sonata` and `SonataSpec` are exported from `molpot` and
  `molpot.composition`.
* anti-double-counting is now the constructor refusal here
  plus `PotentialMeta.check_composition` (see
  `tests/test_molpot/test_forcefield/test_schema.py`); the session-flag
  `UserWarning` it used to be asserted through is deleted.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot import Polarization
from molpot.composition import Sonata, SonataSpec
from molpot.heads import EdgeEnergyHead, PermMultipoleHead
from molpot.potentials import EwaldMultipoleEnergy
from molzoo.allegro import AllegroEncoder, AllegroSpec

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_encoder(*, expose_tensor_track: bool = True, l_max: int = 2) -> AllegroEncoder:
    return AllegroEncoder(
        AllegroSpec(
            atomic_numbers=[1, 6, 7, 8],
            num_scalar_features=16,
            num_tensor_features=4,
            r_max=5.0,
            num_bessels=8,
            num_layers=2,
            l_max=l_max,
            radial_chemical_embed_dim=16,
            allegro_mlp_hidden_layers_depth=1,
            allegro_mlp_hidden_layers_width=32,
            avg_num_neighbors=12.0,
            expose_tensor_track=expose_tensor_track,
        )
    )


@pytest.fixture
def encoder() -> AllegroEncoder:
    return _make_encoder()


@pytest.fixture
def encoder_no_tensor_track() -> AllegroEncoder:
    return _make_encoder(expose_tensor_track=False)


@pytest.fixture
def encoder_lmax1() -> AllegroEncoder:
    return _make_encoder(l_max=1)


def _make_batch(*, with_cell: bool = False) -> TensorDict:
    """B=2 graphs, N=8 atoms, E=24 edges (full bidirectional within each graph)."""
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

    edge_pairs = []
    for offset in (0, 4):
        for i in range(4):
            for j in range(4):
                if i != j:
                    edge_pairs.append([offset + i, offset + j])
    pair_index = torch.tensor(edge_pairs, dtype=torch.long)  # (24, 2)

    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    pair_dist = pair_diff.norm(dim=-1)
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    Z = torch.tensor([1, 6, 8, 7, 1, 6, 8, 7], dtype=torch.long)
    total_charge = torch.zeros(2, dtype=torch.float32)

    graphs_kwargs: dict = {
        "total_charge": total_charge,
        "batch_size": [2],
    }
    if with_cell:
        cell = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1) * 10.0
        graphs_kwargs["cell"] = cell

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[8]),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=pair_diff,
            pair_dist=pair_dist,
            batch_size=[24],
        ),
        graphs=TensorDict(**graphs_kwargs),
        batch_size=[],
    )


@pytest.fixture
def batch() -> TensorDict:
    return _make_batch()


@pytest.fixture
def batch_with_cell() -> TensorDict:
    return _make_batch(with_cell=True)


def _short_range_head(encoder: AllegroEncoder) -> EdgeEnergyHead:
    return EdgeEnergyHead(
        input_dim=encoder.output_dim,
        hidden_dim=32,
        avg_num_neighbors=12.0,
        out_key="energy_short",
    )


# ---------------------------------------------------------------------------
# construction returns a wired nn.Module
# ---------------------------------------------------------------------------


def test_from_encoder_returns_wired_model(encoder):
    model = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
    )
    assert isinstance(model, nn.Module)
    assert model.encoder is encoder
    assert isinstance(model.perm_multipole_head, PermMultipoleHead)
    assert isinstance(model.ewald, EwaldMultipoleEnergy)


# ---------------------------------------------------------------------------
# refuse kappa_head, alpha_head, induced_* kwargs
# ---------------------------------------------------------------------------


def _stub_head(encoder: AllegroEncoder) -> PermMultipoleHead:
    return PermMultipoleHead(
        input_dim=encoder.output_dim,
        avg_num_neighbors=12.0,
        charge=True,
        dipole=True,
        quadrupole=True,
        tensor_irreps=encoder.tensor_track_irreps,
    )


def test_refuse_kappa_head_kwarg(encoder):
    with pytest.raises(ValueError, match="LesPolarizable"):
        Sonata(
            encoder=encoder,
            perm_multipole_head=_stub_head(encoder),
            ewald=EwaldMultipoleEnergy(),
            kappa_head=None,
        )


def test_refuse_alpha_head_kwarg(encoder):
    with pytest.raises(ValueError, match="LesPolarizable"):
        Sonata(
            encoder=encoder,
            perm_multipole_head=_stub_head(encoder),
            ewald=EwaldMultipoleEnergy(),
            alpha_head=object(),
        )


def test_refuse_induced_kwarg(encoder):
    with pytest.raises(ValueError, match="LesPolarizable"):
        Sonata(
            encoder=encoder,
            perm_multipole_head=_stub_head(encoder),
            ewald=EwaldMultipoleEnergy(),
            induced_q=None,
        )


# ---------------------------------------------------------------------------
# refuse Polarization in short_range_head (single + list)
# ---------------------------------------------------------------------------


def test_refuse_polarization_short_range(encoder):
    pol = Polarization()
    with pytest.raises(ValueError, match=r"double-count|induction"):
        Sonata(
            encoder=encoder,
            perm_multipole_head=_stub_head(encoder),
            ewald=EwaldMultipoleEnergy(),
            short_range_head=pol,
        )


def test_refuse_polarization_in_short_range_list(encoder):
    pol = Polarization()
    edge = _short_range_head(encoder)
    with pytest.raises(ValueError, match=r"double-count|induction"):
        Sonata(
            encoder=encoder,
            perm_multipole_head=_stub_head(encoder),
            ewald=EwaldMultipoleEnergy(),
            short_range_head=[edge, pol],
        )


# ---------------------------------------------------------------------------
# forward output schema
# ---------------------------------------------------------------------------


def test_forward_output_schema(encoder, batch):
    sonata = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
        short_range_head=_short_range_head(encoder),
    )
    out = sonata(batch)
    assert {
        "energy",
        "energy_short",
        "energy_es",
        "molecular_dipole",
        "charge_sum_pre_proj",
        "charge_sum_post_proj",
    } <= set(out["graphs"].keys())
    assert {
        "atomic_charges",
        "atomic_dipoles",
        "atomic_quadrupoles",
        "phi",
        "field",
    } <= set(out["atoms"].keys())
    assert out["graphs", "energy"].shape == (2,)
    assert out["graphs", "energy_short"].shape == (2,)
    assert out["graphs", "energy_es"].shape == (2,)
    assert out["atoms", "atomic_charges"].shape == (8,)
    assert out["atoms", "atomic_dipoles"].shape == (8, 3)
    assert out["atoms", "atomic_quadrupoles"].shape == (8, 5)
    assert out["graphs", "molecular_dipole"].shape == (2, 3)
    assert out["atoms", "phi"].shape == (8,)
    assert out["atoms", "field"].shape == (8, 3)


# ---------------------------------------------------------------------------
# energy decomposition total = short + es
# ---------------------------------------------------------------------------


def test_energy_decomposition(encoder, batch):
    sonata = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
        short_range_head=_short_range_head(encoder),
    )
    out = sonata(batch)
    assert torch.allclose(
        out["graphs", "energy"],
        out["graphs", "energy_short"] + out["graphs", "energy_es"],
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# compute_forces yields finite (N, 3)
# ---------------------------------------------------------------------------


def test_compute_forces(encoder, batch):
    sonata = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
        short_range_head=_short_range_head(encoder),
    )
    sonata.train()
    out = sonata(batch, compute_forces=True)
    assert out["atoms", "forces"].shape == (8, 3)
    assert torch.isfinite(out["atoms", "forces"]).all()
    assert out["atoms", "forces"].grad_fn is not None


# ---------------------------------------------------------------------------
# compute_stress yields symmetric (B, 3, 3)
# ---------------------------------------------------------------------------


def test_compute_stress(encoder, batch_with_cell):
    sonata = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
        short_range_head=_short_range_head(encoder),
    )
    out = sonata(batch_with_cell, compute_stress=True)
    assert out["graphs", "stress"].shape == (2, 3, 3)
    assert torch.allclose(
        out["graphs", "stress"], out["graphs", "stress"].transpose(-1, -2), atol=1e-5
    )


# ---------------------------------------------------------------------------
# SonataSpec round-trip preserves forward output
# ---------------------------------------------------------------------------


def test_spec_round_trip(encoder, batch):
    sonata1 = Sonata.from_encoder(
        encoder,
        charge=True,
        dipole=True,
        quadrupole=True,
        avg_num_neighbors=12.0,
    )
    spec = sonata1.config
    assert isinstance(spec, SonataSpec)

    sonata2 = Sonata.from_spec(spec, encoder)
    sonata2.load_state_dict(sonata1.state_dict())

    sonata1.eval()
    sonata2.eval()
    with torch.no_grad():
        out1 = sonata1(batch)
        out2 = sonata2(batch)

    columns = (
        ("graphs", "energy"),
        ("graphs", "energy_short"),
        ("graphs", "energy_es"),
        ("graphs", "molecular_dipole"),
        ("graphs", "charge_sum_pre_proj"),
        ("graphs", "charge_sum_post_proj"),
        ("atoms", "atomic_charges"),
        ("atoms", "atomic_dipoles"),
        ("atoms", "atomic_quadrupoles"),
        ("atoms", "phi"),
        ("atoms", "field"),
    )
    for key in columns:
        assert torch.allclose(out1[key], out2[key], atol=1e-6, rtol=1e-6), (
            f"output[{key!r}] mismatch after from_spec round-trip"
        )


# ---------------------------------------------------------------------------
# from_encoder validates encoder
# ---------------------------------------------------------------------------


def test_from_encoder_requires_expose_tensor_track(encoder_no_tensor_track):
    with pytest.raises(ValueError, match="expose_tensor_track"):
        Sonata.from_encoder(
            encoder_no_tensor_track,
            charge=True,
            dipole=True,
            quadrupole=True,
            avg_num_neighbors=12.0,
        )


def test_from_encoder_requires_lmax_for_dipole(encoder_lmax1):
    with pytest.raises(ValueError, match="l_max"):
        Sonata.from_encoder(
            encoder_lmax1,
            charge=True,
            dipole=True,
            quadrupole=True,
            avg_num_neighbors=12.0,
        )


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------
