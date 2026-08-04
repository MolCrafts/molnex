"""Tests for the paired-trajectory protocol + TrajectoryArtifact (tiny PiNet)."""

import pytest
import torch

from molix.md import (
    PotentialForceField,
    TrajectoryArtifact,
    build_paired_trajectory,
)
from molix.quant import Quantizer
from molzoo.pinet import PiNetPotential
from tests.conftest import make_graph_batch

_DEVICE = torch.device("cpu")


def _tiny_potential() -> PiNetPotential:
    torch.manual_seed(0)
    return (
        PiNetPotential(
            atom_types=[1, 6, 7, 8],
            r_max=4.0,
            n_basis=3,
            pp_nodes=[8, 8],
            pi_nodes=[8, 8],
            ii_nodes=[8, 8],
            depth=2,
            rank=3,
            hidden_dim=16,
        )
        .to(_DEVICE)
        .eval()
    )


def _template():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.3, 1.2, 0.2], [1.4, 1.1, -0.1]],
        dtype=torch.float32,
        device=_DEVICE,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long, device=_DEVICE)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
        device=_DEVICE,
    )
    batch = torch.zeros(4, dtype=torch.long, device=_DEVICE)
    return make_graph_batch(pos, z, edge_index, batch)


def _warmed_ref_and_quant(template):
    ref = _tiny_potential()
    quant = _tiny_potential()
    ref(template.clone(), compute_forces=False)  # warmup lazy params
    quant(template.clone(), compute_forces=False)
    quant.load_state_dict(Quantizer("int4").quantize_state_dict(ref.state_dict()))
    return ref, quant


def _run(ref, quant, template, n_steps=5):
    pos0 = template["atoms", "pos"].clone()
    vel0 = torch.zeros_like(pos0)
    return build_paired_trajectory(
        ref,
        quant,
        template,
        pos0,
        vel0,
        n_steps,
        dt=0.001,
        gamma=1.0,
        kbt=0.0257,
        mass=12.0,
        seed=3,
        condition={"scheme": "int4", "dataset": "qm9"},
    )


def test_paired_trajectory_schema():
    template = _template()
    ref, quant = _warmed_ref_and_quant(template)
    art = _run(ref, quant, template, n_steps=5)

    assert isinstance(art, TrajectoryArtifact)
    n = 4
    assert art.pos.shape == (5, n, 3)
    assert art.vel.shape == (5, n, 3)
    assert art.energy.shape == (5,)
    assert art.f_ref.shape == (5, n, 3)
    assert art.f_quant.shape == (5, n, 3)
    assert art.df.shape == (5, n, 3)
    assert torch.allclose(art.df, art.f_quant - art.f_ref)
    assert art.metadata["dof"] == 3 * n
    assert art.metadata["integrator"].startswith("velocity-verlet")
    assert art.metadata["scheme"] == "int4"


def test_to_dict_has_all_keys():
    template = _template()
    ref, quant = _warmed_ref_and_quant(template)
    d = _run(ref, quant, template).to_dict()
    for key in ("pos", "vel", "energy", "f_ref", "f_quant", "df", "metadata"):
        assert key in d


def test_null_paired_trajectory_zero_df():
    template = _template()
    ref = _tiny_potential()
    ref(template.clone(), compute_forces=False)
    twin = _tiny_potential()
    twin(template.clone(), compute_forces=False)
    twin.load_state_dict(ref.state_dict())  # identical -> ΔF == 0
    pos0 = template["atoms", "pos"].clone()
    vel0 = torch.zeros_like(pos0)
    art = build_paired_trajectory(
        ref,
        twin,
        template,
        pos0,
        vel0,
        4,
        dt=0.001,
        gamma=1.0,
        kbt=0.0257,
        mass=12.0,
        seed=1,
    )
    assert art.df.abs().max().item() == pytest.approx(0.0, abs=1e-10)


def test_force_seam_tracks_live_geometry():
    """PotentialForceField must recompute edge geometry from the live positions,
    not a frozen template ``edge_diff`` — regression for the constant-PES bug
    where swapping only ``pos`` left energy/force pinned to the initial geometry.
    """
    template = _template()
    ref = _tiny_potential()
    ref(template.clone(), compute_forces=False)  # warmup lazy params
    ff = PotentialForceField(ref, template)
    pos0 = template["atoms", "pos"]
    out0 = ff(pos0)
    torch.manual_seed(1)
    pos1 = pos0 + 0.3 * torch.randn_like(pos0)  # non-rigid displacement
    out1 = ff(pos1)
    assert (out1.energy - out0.energy).abs().item() > 1e-6, "energy frozen at initial geometry"
    assert (out1.forces - out0.forces).abs().max().item() > 1e-6, (
        "forces frozen at initial geometry"
    )


def test_paired_trajectory_energy_varies():
    """End-to-end: the PES must be sampled, so energy is not constant over steps."""
    template = _template()
    ref, quant = _warmed_ref_and_quant(template)
    art = _run(ref, quant, template, n_steps=8)
    assert art.energy.std().item() > 1e-9, "energy constant — PES not sampled"
