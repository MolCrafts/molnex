"""Optional kernel-local stack helpers for valence columns → COO.

Spec: learnable-classical-ff-02-valence-topology (ac-005).

These helpers build ``[arity, N]`` long tensors only at potential call sites;
collate output stays column form under nested TensorDict namespaces.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data.collate import collate_molecules
from molix.datasets._valence_columns import (
    stack_angle_index,
    stack_improper_index,
    stack_proper_index,
)


def test_stack_angle_index_shape_and_values():
    angles = TensorDict(
        {
            "atomi": torch.tensor([0, 1], dtype=torch.long),
            "atomj": torch.tensor([1, 2], dtype=torch.long),
            "atomk": torch.tensor([2, 3], dtype=torch.long),
            "type": torch.tensor([0, 1], dtype=torch.long),
        },
        batch_size=[2],
    )
    idx = stack_angle_index(angles)
    assert idx.shape == (3, 2)
    assert idx.dtype == torch.long
    assert idx.tolist() == [[0, 1], [1, 2], [2, 3]]


def test_stack_proper_and_improper_index():
    propers = TensorDict(
        {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
        },
        batch_size=[1],
    )
    impropers = TensorDict(
        {
            "atomi": torch.tensor([1], dtype=torch.long),  # center
            "atomj": torch.tensor([0], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
        },
        batch_size=[1],
    )
    p = stack_proper_index(propers)
    i = stack_improper_index(impropers)
    assert p.shape == (4, 1)
    assert i.shape == (4, 1)
    assert p.tolist() == [[0], [1], [2], [3]]
    assert i.tolist() == [[1], [0], [2], [3]]  # center-first preserved


def test_stack_from_collated_batch_not_required_by_schema():
    """Helper is opt-in after collate; batch itself has no angle_index leaf."""
    samples = [
        {
            "Z": torch.ones(3, dtype=torch.long),
            "pos": torch.zeros(3, 3),
            "angles": {
                "atomi": torch.tensor([0], dtype=torch.long),
                "atomj": torch.tensor([1], dtype=torch.long),
                "atomk": torch.tensor([2], dtype=torch.long),
            },
        }
    ]
    batch = collate_molecules(samples)
    assert "angle_index" not in batch["angles"].keys()
    idx = stack_angle_index(batch["angles"])
    assert idx.tolist() == [[0], [1], [2]]


def test_stack_length_mismatch_raises():
    # Plain mapping (not TensorDict) so construction itself does not validate shapes.
    angles = {
        "atomi": torch.tensor([0, 1], dtype=torch.long),
        "atomj": torch.tensor([1], dtype=torch.long),
        "atomk": torch.tensor([2, 3], dtype=torch.long),
    }
    with pytest.raises(ValueError, match="share length"):
        stack_angle_index(angles)
