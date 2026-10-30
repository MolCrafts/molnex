"""Butane conformers collated as a run collates them, and a tiny MACE conformer encoder."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molrep.encoders.mace_conformer import MaceConformerEncoder, MaceConformerEncoderSpec
from tests.test_molzoo.test_rondo.conftest import DTYPE, butane_frame, collate

#: Small sizes; the fields are the spec's.
TINY = dict(
    num_features=8,
    radial_mlp=(8,),
    avg_num_neighbors=4.0,
    width=8,
    bond_width=6,
    hidden=8,
    charge_dim=2,
    bond_lengths=(0.8, 2.4, 4),
)


def encoder() -> MaceConformerEncoder:
    torch.manual_seed(0)
    return MaceConformerEncoder(MaceConformerEncoderSpec(**TINY), dtype=DTYPE).eval()


def place(batch: TensorDict, pos: torch.Tensor) -> TensorDict:
    """The batch at new positions, with the pair vectors rebuilt from them."""
    index = batch["pairs", "pair_index"]
    batch["atoms", "pos"] = pos
    batch["pairs", "pair_diff"] = pos[index[:, 1]] - pos[index[:, 0]]
    batch["pairs", "pair_dist"] = torch.linalg.vector_norm(batch["pairs", "pair_diff"], dim=-1)
    return batch


def conformers(profile: list[int], *, jitter: float = 0.1) -> TensorDict:
    """``len(profile)`` butanes; every one after the first is displaced at random."""
    batch = collate([butane_frame() for _ in profile])
    generator = torch.Generator().manual_seed(1)
    pos = batch["atoms", "pos"].clone()
    n = pos.shape[0] // len(profile)
    pos[n:] += jitter * torch.randn(pos[n:].shape, generator=generator, dtype=pos.dtype)
    batch["graphs", "profile"] = torch.tensor(profile)
    return place(batch, pos)


@pytest.fixture
def butane() -> TensorDict:
    return conformers([0])
