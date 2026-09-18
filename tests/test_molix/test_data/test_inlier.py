"""``ForceInlierWeight`` — the per-atom column that drops reference outliers."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data import ForceInlierWeight


def _batch(forces: torch.Tensor, mask: torch.Tensor | None = None) -> TensorDict:
    n = forces.shape[0]
    atoms = TensorDict(forces=forces, batch_size=[n])
    if mask is not None:
        atoms["mask"] = mask
    return TensorDict(atoms=atoms, batch_size=[])


class TestForceInlierWeight:
    def test_bounds_by_magnitude_not_by_component(self):
        # A force of (3, 4, 0) has magnitude 5: no single component exceeds a
        # bound of 4.5, but the atom does.
        batch = _batch(torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]))
        ForceInlierWeight(threshold=4.5).execute(batch)
        assert batch["atoms", "force_weight"].tolist() == [0.0, 1.0]

    def test_keeps_an_atom_exactly_at_the_bound(self):
        batch = _batch(torch.tensor([[3.0, 4.0, 0.0]]))
        ForceInlierWeight(threshold=5.0).execute(batch)
        assert batch["atoms", "force_weight"].tolist() == [1.0]

    def test_ghost_rows_never_count(self):
        # A padder's ghost carries a placeholder force that may well be inside
        # the bound; the structural mask has to win.
        batch = _batch(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            mask=torch.tensor([False, True]),
        )
        ForceInlierWeight(threshold=10.0).execute(batch)
        assert batch["atoms", "force_weight"].tolist() == [0.0, 1.0]

    def test_writes_into_the_caller_batch(self):
        batch = _batch(torch.tensor([[1.0, 0.0, 0.0]]))
        returned = ForceInlierWeight(threshold=2.0).execute(batch)
        assert returned is batch
        assert "force_weight" in batch["atoms"].keys()

    def test_refuses_a_plain_dict(self):
        with pytest.raises(TypeError):
            ForceInlierWeight(threshold=1.0).execute({"atoms": {}})

    def test_refuses_a_batch_with_no_reference_forces(self):
        batch = TensorDict(atoms=TensorDict(Z=torch.ones(2), batch_size=[2]), batch_size=[])
        with pytest.raises(KeyError):
            ForceInlierWeight(threshold=1.0).execute(batch)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_refuses_a_bound_that_states_nothing(self, bad):
        with pytest.raises(ValueError):
            ForceInlierWeight(threshold=bad)

    def test_task_id_separates_two_bounds(self):
        # The column changes the collated batch, so two bounds must not share
        # a preprocessing cache.
        assert (
            ForceInlierWeight(threshold=200.0).task_id != ForceInlierWeight(threshold=50.0).task_id
        )
