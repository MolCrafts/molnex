"""LammpsWriter — molpy.io data/settings from ForceField + TensorDict."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.ff_export import LammpsWriter
from molpot.forcefield import ForceField, OPLSForceField


def _two_atom_system() -> tuple[ForceField, TensorDict]:
    ff = OPLSForceField().with_params(
        bonds={
            "k": torch.tensor([200.0], dtype=torch.float64),
            "r0": torch.tensor([1.53], dtype=torch.float64),
        },
        propers={
            "k1": torch.tensor([1.0], dtype=torch.float64),
            "k2": torch.tensor([2.0], dtype=torch.float64),
            "k3": torch.tensor([0.0], dtype=torch.float64),
            "k4": torch.tensor([0.0], dtype=torch.float64),
        },
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]], dtype=torch.float64)
    system = TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
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
    return ff, system


class TestLammpsWriter:
    def test_write_settings_bond_coeff(self, tmp_path):
        ff, system = _two_atom_system()
        text = LammpsWriter().write_settings(tmp_path / "s.ff", ff, system).read_text()
        assert "bond_style harmonic" in text
        assert "bond_coeff" in text
        assert "100" in text  # K = k/2

    def test_write_data_identity_types(self, tmp_path):
        _, system = _two_atom_system()
        text = LammpsWriter().write_data(tmp_path / "s.data", system).read_text()
        assert "2 atoms" in text
        assert "1 bonds" in text
        assert "Atoms" in text

    def test_write_data_reads_atoms_q(self, tmp_path):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        system = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": pos,
                        "q": torch.tensor([0.2, -0.2], dtype=torch.float64),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
        text = LammpsWriter().write_data(tmp_path / "s.data", system).read_text()
        assert "0.2" in text
        assert "-0.2" in text

    def test_write_data_requires_atoms_pos(self, tmp_path):
        system = TensorDict(
            {"atoms": TensorDict({"Z": torch.tensor([6, 6])}, batch_size=[2])},
            batch_size=[],
        )
        with pytest.raises(KeyError):
            LammpsWriter().write_data(tmp_path / "s.data", system)
