"""``column_match``: nested TensorDict columns, mask as a weight, no row gather."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.core.losses import column_match


def _features() -> TensorDict:
    """Three atoms. The last row is a ghost: mask 0, garbage features."""
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "node_features": torch.tensor(
                        [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]], dtype=torch.float32
                    ),
                    "node_features_teacher": torch.zeros(3, 2),
                    "mask": torch.tensor([1.0, 1.0, 0.0]),
                },
                batch_size=[3],
            )
        },
        batch_size=[],
    )


def _energy_pair() -> tuple[TensorDict, TensorDict]:
    """Prediction energy ``[1, 3]`` against a teacher column ``[0, 1]``."""
    preds = TensorDict(
        {"graphs": TensorDict({"energy": torch.tensor([1.0, 3.0])}, batch_size=[2])},
        batch_size=[],
    )
    batch = TensorDict(
        {"graphs": TensorDict({"energy_teacher": torch.tensor([0.0, 1.0])}, batch_size=[2])},
        batch_size=[],
    )
    return preds, batch


class TestColumnMatch:
    """Nested columns, mask weighting, and the refusals around them."""

    def test_masked_mean_matches_the_hand_calculation(self) -> None:
        batch = _features()
        # Squared rows (1, 1), (4, 4), (81, 81). Mask keeps the first two.
        # Weighted sum 10, row-weight sum 2, two elements per row → 10 / 4.
        loss = column_match(
            ("atoms", "node_features"),
            ("atoms", "node_features_teacher"),
            mask_key=("atoms", "mask"),
        )

        got = loss(batch, batch)

        assert torch.allclose(got, torch.tensor(2.5))

    def test_reads_both_sides_as_nested_columns(self) -> None:
        preds, batch = _energy_pair()
        loss = column_match(("graphs", "energy"), ("graphs", "energy_teacher"))

        got = loss(preds, batch)

        assert torch.allclose(got, torch.tensor(2.5))

    def test_dict_prediction_raises(self) -> None:
        _, batch = _energy_pair()
        loss = column_match(("graphs", "energy"), ("graphs", "energy_teacher"))

        with pytest.raises(TypeError, match="TensorDict"):
            loss({"energy": torch.tensor([1.0, 3.0])}, batch)

    def test_shape_mismatch_raises(self) -> None:
        preds, _ = _energy_pair()
        batch = TensorDict(
            {"graphs": TensorDict({"energy_teacher": torch.tensor([0.0])}, batch_size=[1])},
            batch_size=[],
        )
        loss = column_match(("graphs", "energy"), ("graphs", "energy_teacher"))

        with pytest.raises(ValueError, match="shape"):
            loss(preds, batch)

    def test_unknown_reduction_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            column_match(
                ("graphs", "energy"),
                ("graphs", "energy_teacher"),
                reduction="median",
            )
