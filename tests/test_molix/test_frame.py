"""Unit tests for :mod:`molix.frame`."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix import Compiler
from molix.data import collate_molecules
from molix.frame import TensorFrame, as_tensor_frame
from tests.conftest import frame


class TestTensorFrame:
    """The typed view over a collated batch."""

    def test_atoms_is_the_only_namespace_a_batch_must_carry(self) -> None:
        """``pairs`` is built downstream and the term namespaces depend on the
        frames, so declaring them required would reject a legal batch."""
        atoms = TensorDict(Z=torch.tensor([1, 8]), pos=torch.zeros(2, 3), batch_size=[2])
        assert TensorFrame(atoms=atoms, batch_size=[]).atoms is not None
        with pytest.raises(TypeError, match="atoms"):
            TensorFrame(graphs=TensorDict(batch_size=[1]), batch_size=[])

    def test_wrapping_a_collated_batch_shares_every_leaf_storage(self) -> None:
        """The point of the type: collate → model costs no copy."""
        batch = collate_molecules(
            [
                frame(
                    Z=[8, 1],
                    pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    bonds=([0], [1]),
                    energy=-1.0,
                ),
                frame(
                    Z=[6, 1],
                    pos=[[0.0, 0.0, 0.0], [0.0, 1.1, 0.0]],
                    bonds=([0], [1]),
                    energy=-2.0,
                ),
            ]
        )
        view = TensorFrame.from_tensordict(batch)
        shared = {
            key: view[key].data_ptr() == leaf.data_ptr()
            for key, leaf in batch.items(include_nested=True, leaves_only=True)
        }
        assert shared and all(shared.values()), shared

    def test_a_namespace_the_batch_lacks_is_named_in_the_error(self) -> None:
        """A missing optional namespace must not read back as an empty one."""
        atoms = TensorDict(Z=torch.tensor([1, 8]), pos=torch.zeros(2, 3), batch_size=[2])
        with pytest.raises(AttributeError, match="impropers"):
            _ = TensorFrame(atoms=atoms, batch_size=[]).impropers

    @pytest.mark.slow
    def test_reading_through_the_typed_view_traces_to_one_graph(self) -> None:
        """Iron law: the container a model reads costs zero graph breaks."""

        def energy(td: TensorFrame) -> torch.Tensor:
            diff = td["atoms", "pos"][td["pairs", "pair_index"][:, 1]]
            return (diff * td.atoms["Z"].unsqueeze(-1)).sum()

        batch = TensorDict(
            atoms=TensorDict(Z=torch.tensor([1.0, 8.0]), pos=torch.zeros(2, 3), batch_size=[2]),
            pairs=TensorDict(pair_index=torch.tensor([[0, 1], [1, 0]]), batch_size=[2]),
            batch_size=[],
        )
        explanation = Compiler.explain(energy, TensorFrame.from_tensordict(batch))
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0


def _atoms_only_batch() -> TensorDict:
    atoms = TensorDict(
        {"pos": torch.zeros(2, 3, dtype=torch.float64), "Z": torch.tensor([1, 8])},
        batch_size=[2],
    )
    return TensorDict({"atoms": atoms}, batch_size=[])


class TestAsTensorFrame:
    def test_view_passes_through_unchanged(self):
        view = TensorFrame.from_tensordict(_atoms_only_batch())
        assert as_tensor_frame(view) is view

    def test_batch_is_wrapped_over_its_own_storage(self):
        batch = _atoms_only_batch()
        view = as_tensor_frame(batch)
        assert isinstance(view, TensorFrame)
        assert view["atoms", "pos"].data_ptr() == batch["atoms", "pos"].data_ptr()

    def test_mapping_of_namespaces_keeps_the_leaves(self):
        batch = _atoms_only_batch()
        view = as_tensor_frame({"atoms": batch["atoms"]})
        assert isinstance(view, TensorFrame)
        assert view["atoms", "pos"].data_ptr() == batch["atoms", "pos"].data_ptr()

    def test_anything_else_is_refused_naming_the_contract(self):
        with pytest.raises(TypeError, match=r"collate_molecules"):
            as_tensor_frame([_atoms_only_batch()])  # type: ignore[arg-type]


class TestBesideLabels:
    def test_the_write_moves_to_its_target_and_the_label_comes_back(self):
        from molix.frame import beside_labels

        batch = collate_molecules([frame(Z=[1], pos=[[0.0, 0.0, 0.0]], energy=5.0)])
        label = batch["graphs", "energy"].clone()

        def body() -> str:
            batch["graphs", "energy"] = batch["graphs", "energy"] * 0 + 7.0
            return "ran"

        assert (
            beside_labels(batch, {("graphs", "energy"): ("graphs", "energy_pred")}, body) == "ran"
        )
        assert torch.equal(batch["graphs", "energy"], label)
        assert batch["graphs", "energy_pred"].tolist() == [7.0]

    def test_a_write_with_no_label_stays_where_it_was_written(self):
        from molix.frame import beside_labels

        batch = collate_molecules([frame(Z=[1], pos=[[0.0, 0.0, 0.0]])])
        beside_labels(
            batch,
            {("graphs", "energy"): ("graphs", "energy_pred")},
            lambda: batch.set(("graphs", "energy"), torch.tensor([3.0])),
        )
        assert batch["graphs", "energy"].tolist() == [3.0]
        assert batch["graphs", "energy_pred"].tolist() == [3.0]
