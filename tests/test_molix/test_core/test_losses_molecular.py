"""Tests for molecule-centered energy losses."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.core.losses.molecular import (
    center_by_group,
    molecule_centered_energy_mse,
    parameter_bag_mse,
)


class TestCenterByGroup:
    def test_exact_centering(self):
        values = torch.tensor([1.0, 3.0, 10.0, 14.0])
        groups = torch.tensor([0, 0, 1, 1])
        c = center_by_group(values, groups)
        assert torch.allclose(c, torch.tensor([-1.0, 1.0, -2.0, 2.0]))


class TestMoleculeCenteredEnergyMse:
    def test_zero_when_identical_after_center(self):
        # same relative pattern, different offsets per molecule
        pred = torch.tensor([1.0, 2.0, 5.0, 7.0])
        true = torch.tensor([10.0, 11.0, 0.0, 2.0])
        groups = torch.tensor([0, 0, 1, 1])
        batch = TensorDict(
            {
                "graphs": TensorDict(
                    {
                        "mm_energy": true,
                        "molecule_id_index": groups,
                    },
                    batch_size=[4],
                )
            },
            batch_size=[],
        )
        loss = molecule_centered_energy_mse()({"energy": pred}, batch)
        assert float(loss) < 1e-10


class TestParameterBagMse:
    def test_zero_and_positive(self):
        a = {"k": torch.tensor([1.0, 2.0])}
        assert float(parameter_bag_mse(a, a)) == 0.0
        b = {"k": torch.tensor([1.0, 4.0])}
        assert float(parameter_bag_mse(a, b)) == ((0.0 + 4.0) / 2.0)
