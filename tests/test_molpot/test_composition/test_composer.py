"""Tests for :mod:`molpot.composition.composer` — the composed force field as a potential.

The pooling tests at the top exercise :mod:`molpot.pooling`, the step that
turns an encoder's layered features into the ``atoms.node_features`` the
composer reads.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.schema import ENERGY_KEY, FORCES_KEY, VIRIAL_KEY
from molpot.composition import LJParameterHead, PotentialComposer
from molpot.pooling import EdgeToNodePooling, LayerPooling
from molpot.potentials import LJ126
from molzoo.allegro import AllegroEncoder, AllegroSpec

#: Finite-difference step (Å for positions, dimensionless for strain) and the
#: gate of the fourth-order stencil in float64.
FD_STEP = 1e-3
FD_ATOL = 1e-8
FD_RTOL = 1e-6

#: Atom/axis pairs probed by the force finite-difference test.
FD_PROBES = ((0, 0), (1, 1), (4, 2))

#: Strain components probed by the virial finite-difference test.
STRAIN_PROBES = ((0, 0), (1, 1), (0, 1), (0, 2))


def _four_point(values: tuple[float, float, float, float], step: float) -> float:
    """``f'(x)`` from ``f(x ± step)``, ``f(x ± 2 step)`` — the fourth-order central stencil."""
    plus2, plus1, minus1, minus2 = values
    return (-plus2 + 8.0 * plus1 - 8.0 * minus1 + minus2) / (12.0 * step)


def _batch(
    pos: torch.Tensor, *, cell: torch.Tensor | None = None, num_graphs: int = 2
) -> TensorDict:
    """Two chains (3 + 2 atoms) on *pos* as two molecules, or as one graph; features seeded.

    A strain leaf is one lattice, so the virial test takes the one-graph form.
    """
    pair_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 3]],
        dtype=torch.long,
    )
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    graphs = TensorDict(batch_size=[num_graphs])
    if cell is not None:
        graphs["cell"] = cell
    batch_idx = torch.tensor([0, 0, 0, 1, 1] if num_graphs == 2 else [0, 0, 0, 0, 0])
    torch.manual_seed(0)
    return TensorDict(
        atoms=TensorDict(
            Z=torch.tensor([1, 6, 8, 1, 6], dtype=torch.long),
            pos=pos,
            batch=batch_idx,
            node_features=torch.randn(5, 16, dtype=pos.dtype),
            batch_size=[5],
        ),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=pair_diff,
            pair_dist=pair_diff.norm(dim=-1),
            batch_size=[6],
        ),
        graphs=graphs,
        batch_size=[],
    )


def _positions(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.1, 0.0],
            [2.8, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [1.5, 0.0, -0.1],
        ],
        dtype=dtype,
    )


def _composer(*, wrt: tuple[str, ...]) -> PotentialComposer:
    torch.manual_seed(1)
    return PotentialComposer(
        head=LJParameterHead(
            feature_dim=16, hidden_dim=32, init_epsilon=0.05, init_sigma=1.0, dtype=torch.float64
        ),
        potentials={
            "lj_main": LJ126(energy_scale=1.0),
            "lj_aux": LJ126(energy_scale=0.25),
        },
        wrt=wrt,
    )


def test_layer_pooling_mean():
    pool = LayerPooling("mean")
    features_3d = torch.randn(5, 3, 8)
    out = pool(features_3d)
    assert out.shape == (5, 8)
    assert torch.allclose(out, features_3d.mean(dim=1))

    features_2d = torch.randn(5, 8)
    out = pool(features_2d)
    assert out.shape == (5, 8)
    assert torch.equal(out, features_2d)


def test_layer_pooling_sum():
    pool = LayerPooling("sum")
    features = torch.randn(5, 3, 8)
    out = pool(features)
    assert torch.allclose(out, features.sum(dim=1))


def test_layer_pooling_last():
    pool = LayerPooling("last")
    features = torch.randn(5, 3, 8)
    out = pool(features)
    assert torch.equal(out, features[:, -1])


def test_edge_to_node_pooling():
    edge_features = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    pair_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)

    pool = EdgeToNodePooling("mean")
    node_features = pool(edge_features, pair_index, num_nodes=3)
    assert node_features.shape == (3, 2)

    pool_sum = EdgeToNodePooling("sum")
    node_features_sum = pool_sum(edge_features, pair_index, num_nodes=3)
    assert node_features_sum.shape == (3, 2)


class TestPotentialComposer:
    def test_an_empty_term_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one potential"):
            PotentialComposer(head=LJParameterHead(feature_dim=16), potentials={})

    def test_energy_sums_the_terms_and_writes_them_with_the_parameters(self) -> None:
        out = _composer(wrt=()).energy(_batch(_positions(torch.float64)))
        assert out[ENERGY_KEY].shape == (2,)
        torch.testing.assert_close(
            out[ENERGY_KEY], out["graphs", "energy_lj_main"] + out["graphs", "energy_lj_aux"]
        )
        torch.testing.assert_close(
            out["graphs", "energy_lj_aux"], 0.25 * out["graphs", "energy_lj_main"]
        )
        assert out["atoms", "epsilon"].shape == (5,)
        assert out["atoms", "sigma"].shape == (5,)

    def test_forces_match_central_differences(self) -> None:
        """``F = -dE/dx`` against a fourth-order central difference of ``energy``."""
        composer = _composer(wrt=("atoms.pos",))
        pos = _positions(torch.float64)
        with torch.no_grad():
            forces = composer(_batch(pos))[FORCES_KEY]

        def total(atom: int, axis: int, steps: float) -> float:
            shifted = pos.clone()
            shifted[atom, axis] += steps * FD_STEP
            with torch.no_grad():
                return float(composer.energy(_batch(shifted))[ENERGY_KEY].sum())

        for atom, axis in FD_PROBES:
            numerical = -_four_point(
                tuple(total(atom, axis, steps) for steps in (2.0, 1.0, -1.0, -2.0)), FD_STEP
            )
            assert float(forces[atom, axis]) == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)

    def test_virial_matches_central_differences_of_the_strained_energy(self) -> None:
        """``W = ∂E/∂ε`` against a fourth-order central difference of the strained energy.

        The reference strains the same tensors the derivation strains,
        ``v @ (I + ε_sym)ᵀ``: positions, cell and the edge displacements.
        Cauchy convention: no doubled shear, so ``σ = W / V``.
        """
        composer = _composer(wrt=("atoms.pos", "graphs.strain"))
        pos = _positions(torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        with torch.no_grad():
            virial = composer(_batch(pos, cell=cell.unsqueeze(0), num_graphs=1))[VIRIAL_KEY][0]

        def strained_energy(i: int, j: int, steps: float) -> float:
            delta = torch.zeros(3, 3, dtype=torch.float64)
            delta[i, j] = steps * FD_STEP
            transform = (torch.eye(3, dtype=torch.float64) + 0.5 * (delta + delta.T)).T
            frame = _batch(pos @ transform, cell=(cell @ transform).unsqueeze(0), num_graphs=1)
            with torch.no_grad():
                return float(composer.energy(frame)[ENERGY_KEY].sum())

        for i, j in STRAIN_PROBES:
            numerical = _four_point(
                tuple(strained_energy(i, j, steps) for steps in (2.0, 1.0, -1.0, -2.0)), FD_STEP
            )
            assert float(virial[i, j]) == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)

    def test_pooled_allegro_features_feed_the_composer(self) -> None:
        """An encoder's layered edge features, pooled to ``atoms.node_features``, price."""
        batch = _batch(_positions(torch.float32))
        encoder = AllegroEncoder(
            AllegroSpec(
                atomic_numbers=[1, 6, 7, 8],
                num_scalar_features=16,
                num_tensor_features=8,
                r_max=5.0,
                num_layers=2,
                radial_chemical_embed_dim=16,
                allegro_mlp_hidden_layers_depth=1,
                allegro_mlp_hidden_layers_width=16,
                avg_num_neighbors=4.0,
            ),
            dtype=torch.float32,
        )
        encoder(batch)
        # Allegro writes a flat ``(E, F·(L+1))`` DenseNet stack; reshape to
        # ``(E, L+1, F)`` so LayerPooling reduces over the layer axis.
        edge_features = batch["pairs", "edge_features"]
        edge_features = edge_features.view(edge_features.shape[0], -1, encoder.num_scalar_features)
        batch["atoms", "node_features"] = EdgeToNodePooling("mean")(
            LayerPooling("mean")(edge_features), batch["pairs", "pair_index"], num_nodes=5
        )
        composer = PotentialComposer(
            head=LJParameterHead(feature_dim=16, hidden_dim=32, dtype=torch.float32),
            potentials={"lj126": LJ126()},
            wrt=("atoms.pos",),
        )
        out = composer(batch)
        assert out[ENERGY_KEY].shape == (2,)
        assert out[FORCES_KEY].shape == (5, 3)
