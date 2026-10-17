"""Unit tests for molzoo.allegro.potential.AllegroPotential.

The two-method contract (``forward`` is the ``Derivative`` of ``energy``)
is asserted once on a stub in ``tests/test_molix/test_autograd/test_potential.py``;
the encoder's symmetries in ``test_encoder.py``. Here: what Allegro's own
``energy`` reads, writes and adds, plus one force-equivariance and one
finite-difference gate on the product.

The official-package parity case needs the out-of-tree assets named in
``src/molzoo/specs/allegro.md`` §7.3, skips without
``MOLNEX_ALLEGRO_WEIGHTS_DIR`` and is ``slow``.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.compile import Compiler
from molzoo.allegro import AllegroPotential, AllegroSpec
from tests.conftest import make_graph_batch, rotate_graph, rotation_matrix_x
from tests.test_molzoo.test_allegro.conftest import DTYPE, TINY, WEIGHTS_DIR

FORCES = ("atoms.pos",)
ATOL = 1e-10

#: Rotation angle (rad) of the force-equivariance test, about ``x``.
ROTATION_X = -0.4

#: Central-difference step (Å) and tolerances. On this cluster (``|F| ≤ 2.3``
#: eV/Å, ``E ≈ 1`` eV) the second-order stencil's error at this step is
#: ≤ 5e-9 measured, i.e. the gate has a 20× margin.
FD_STEP = 1e-5
FD_ATOL = 1e-7
FD_RTOL = 1e-6

needs_assets = pytest.mark.skipif(
    not (WEIGHTS_DIR / "goldens.pt").exists(), reason="MOLNEX_ALLEGRO_WEIGHTS_DIR not set"
)


def _model(*, wrt=FORCES, **overrides) -> AllegroPotential:
    torch.manual_seed(0)
    return AllegroPotential(AllegroSpec(**{**TINY, **overrides}), wrt=wrt, dtype=DTYPE)


def _energy(model: AllegroPotential, batch: TensorDict) -> torch.Tensor:
    return model(batch.clone())["graphs", "energy"].detach()


def _moved(batch: TensorDict, pos: torch.Tensor) -> TensorDict:
    """``batch`` with its pair geometry rebuilt from ``pos``."""
    return make_graph_batch(
        pos, batch["atoms", "Z"], batch["pairs", "pair_index"], batch["atoms", "batch"]
    )


def _golden_batch(record: dict) -> TensorDict:
    n = record["pos"].shape[0]
    pair_index = record["edge_index"].T.contiguous()
    batch = TensorDict(
        atoms=TensorDict(
            Z=record["Z"].long(),
            pos=record["pos"].double(),
            batch=torch.zeros(n, dtype=torch.long),
            batch_size=[n],
        ),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=record["edge_vectors"].double(),
            batch_size=[pair_index.shape[0]],
        ),
        graphs=TensorDict({}, batch_size=[1]),
        batch_size=[],
    )
    if record.get("cell") is not None:
        batch["graphs", "cell"] = record["cell"].double().view(1, 3, 3)
    return batch


class TestAllegroPotential:
    def test_atomic_energies_sum_to_the_graph_energy(self, cluster):
        model = _model()
        frame = model.energy(cluster.clone())
        torch.testing.assert_close(frame["atoms", "energy"].sum(), frame["graphs", "energy"][0])

    def test_a_pair_beyond_the_cutoff_adds_nothing(self, cluster):
        model = _model(wrt=())
        ref = _energy(model, cluster)
        pos = torch.cat(
            [cluster["atoms", "pos"], torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)]
        )
        pairs = [[i, j] for i in range(6) for j in range(6) if i != j]
        far = make_graph_batch(
            pos,
            torch.tensor([6, 1, 1, 8, 1, 1]),
            torch.tensor(pairs),
            torch.zeros(6, dtype=torch.long),
        )
        far["atoms", "mask"] = torch.tensor([True] * 5 + [False])
        torch.testing.assert_close(_energy(model, far), ref, atol=ATOL, rtol=0)

    def test_forces_rotate_with_the_system(self, cluster):
        """Forces are equivariant: ``F(Rx) = R F(x)``."""
        model = _model()
        rotation = rotation_matrix_x(ROTATION_X, torch.float64)
        base = model(cluster.clone())["atoms", "forces"]
        turned = model(rotate_graph(cluster, rotation))["atoms", "forces"]
        torch.testing.assert_close(turned, base @ rotation.T, atol=ATOL, rtol=0.0)

    def test_forces_match_central_differences(self, cluster):
        """``F = -dE/dx`` against a central difference of ``energy`` (see :data:`FD_STEP`)."""
        model = _model(wrt=())
        forces = _model()(cluster.clone())["atoms", "forces"]
        pos = cluster["atoms", "pos"]
        fd = torch.zeros(2, 3, dtype=pos.dtype)
        with torch.no_grad():
            for row, i in enumerate((0, 3)):
                for k in range(3):
                    plus, minus = pos.clone(), pos.clone()
                    plus[i, k] += FD_STEP
                    minus[i, k] -= FD_STEP
                    fd[row, k] = -(
                        model.energy(_moved(cluster, plus))["graphs", "energy"]
                        - model.energy(_moved(cluster, minus))["graphs", "energy"]
                    ).sum() / (2 * FD_STEP)
        torch.testing.assert_close(forces[[0, 3]], fd, atol=FD_ATOL, rtol=FD_RTOL)

    def test_scale_and_shift_act_by_type(self, cluster):
        base = _energy(_model(wrt=()), cluster)
        shifted = _energy(_model(wrt=(), per_type_energy_shifts=[1.0, 10.0, 100.0]), cluster)
        # C + 3 H + O
        torch.testing.assert_close(
            shifted - base, torch.tensor([10.0 + 3.0 + 100.0], dtype=torch.float64)
        )

    def test_zbl_adds_a_repulsion(self, cluster):
        assert _energy(_model(wrt=(), zbl=True), cluster) > _energy(_model(wrt=()), cluster)

    def test_exposing_the_tensor_track_is_an_encoder_option(self):
        with pytest.raises(ValueError, match="AllegroEncoder"):
            _model(expose_tensor_track=True)

    @pytest.mark.slow
    def test_forward_compiles_to_one_graph_in_inference(self, cluster):
        model = _model()
        model(cluster.clone())  # spend the eager element gate
        with torch.no_grad():
            explanation = Compiler.explain(model, cluster)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )

    @pytest.mark.slow
    @needs_assets
    @pytest.mark.parametrize(
        ("package", "goldens"),
        [
            ("allegro_small_default.nequip.zip", "goldens.pt"),
            ("allegro_small_zbl.nequip.zip", "goldens_zbl.pt"),
        ],
    )
    @pytest.mark.parametrize("structure", ["molecule", "periodic"])
    def test_load_matches_the_official_package(self, package, goldens, structure):
        """fp64 upstream goldens (nequip 0.19.1 / allegro 0.8.3), periodic case with self-images."""
        record = torch.load(WEIGHTS_DIR / goldens, weights_only=True)[structure]
        periodic = record.get("cell") is not None
        wrt = ("atoms.pos", "graphs.strain") if periodic else ("atoms.pos",)
        model = AllegroPotential.load(WEIGHTS_DIR / package, wrt=wrt, dtype=DTYPE)
        out = model(_golden_batch(record))
        n = record["pos"].shape[0]
        energy = out["graphs", "energy"].detach()
        assert float((energy - record["total_energy"].double()).abs().max()) / n <= 1e-8
        torch.testing.assert_close(
            out["atoms", "forces"].detach(), record["forces"].double(), atol=1e-7, rtol=0
        )
        if periodic:  # nequip's virial is -dE/deps; molnex graphs.virial is +dE/deps
            torch.testing.assert_close(
                out["graphs", "virial"].detach().view(3, 3),
                -record["virial"].double().view(3, 3),
                atol=1e-7,
                rtol=0,
            )
