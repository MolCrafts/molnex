"""Unit tests for espaloma_mm_protocol_train Parameterizer + OPLSAA path."""

from __future__ import annotations

import torch
from scripts.mm_param_learning.espaloma_mm_protocol_train import (
    MMEnergyModel,
    build_espaloma_like_model,
)
from tensordict import TensorDict

from molpot.forcefield import OPLSAAForceField, Parameterizer


def _two_atom_bond_batch(*, dtype: torch.dtype = torch.float64) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=dtype)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
                    "batch": torch.zeros(2, dtype=torch.long),
                    "q": torch.zeros(2, dtype=dtype),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
            "angles": TensorDict(
                {
                    "atomi": torch.zeros(0, dtype=torch.long),
                    "atomj": torch.zeros(0, dtype=torch.long),
                    "atomk": torch.zeros(0, dtype=torch.long),
                },
                batch_size=[0],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.zeros(0, dtype=torch.long),
                    "atomj": torch.zeros(0, dtype=torch.long),
                    "atomk": torch.zeros(0, dtype=torch.long),
                    "atoml": torch.zeros(0, dtype=torch.long),
                },
                batch_size=[0],
            ),
            "impropers": TensorDict(
                {
                    "atomi": torch.zeros(0, dtype=torch.long),
                    "atomj": torch.zeros(0, dtype=torch.long),
                    "atomk": torch.zeros(0, dtype=torch.long),
                    "atoml": torch.zeros(0, dtype=torch.long),
                },
                batch_size=[0],
            ),
            "graphs": TensorDict(
                {"num_atoms": torch.tensor([2], dtype=torch.long)},
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


class TestBuildEspalomaLikeModel:
    def test_returns_parameterizer_with_oplsaa_template(self) -> None:
        model = build_espaloma_like_model(energy_scope="bonded", hidden=16, n_blocks=1)
        assert isinstance(model, Parameterizer)
        assert isinstance(model.template, OPLSAAForceField)

    def test_full_scope_also_uses_oplsaa_template(self) -> None:
        model = build_espaloma_like_model(energy_scope="full", hidden=16, n_blocks=1)
        assert isinstance(model, Parameterizer)
        assert isinstance(model.template, OPLSAAForceField)
        assert model.atom_head is not None
        assert model.charge_model is not None


class TestMMEnergyModel:
    def test_forward_energy_shape_is_batch(self) -> None:
        imm = build_espaloma_like_model(energy_scope="full", hidden=16, n_blocks=1)
        wrap = MMEnergyModel(imm)
        batch = _two_atom_bond_batch()
        out = wrap(batch)
        assert set(out) == {"energy"}
        assert out["energy"].shape == (1,)
        assert out["energy"].dtype == torch.float64
