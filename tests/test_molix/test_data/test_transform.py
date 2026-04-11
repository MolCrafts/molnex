import torch

from molix.data.task import SampleTask, DatasetTask, Runnable
from molix.data.tasks import AtomicDress, NeighborList
from molix.data.pipeline import pipeline


def _compute_neighbor_list_naive(positions: torch.Tensor, cutoff: float):
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.norm(diff, dim=2)
    mask = (dist < cutoff) & (dist > 0)
    i_idx, j_idx = torch.where(mask)
    valid = i_idx > j_idx
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    edge_vec = positions[i_idx] - positions[j_idx]
    edge_dist = dist[i_idx, j_idx]
    return i_idx, j_idx, edge_vec, edge_dist


class TestNeighborList:
    def test_is_sample_task(self):
        t = NeighborList(cutoff=5.0)
        assert isinstance(t, SampleTask)
        assert isinstance(t, Runnable)

    def test_small_system_pairs(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": positions,
                  "targets": {"U0": torch.tensor([0.0])}}

        task = NeighborList(cutoff=2.0, max_num_pairs=10)
        result = task(sample)

        edge_index = result["edge_index"]
        assert edge_index.ndim == 2
        assert edge_index.shape[1] == 2

        op_dist = result["bond_dist"]
        ref_i, ref_j, _, ref_dist = _compute_neighbor_list_naive(positions, 2.0)

        assert edge_index[:, 0].numel() == ref_i.numel()
        assert torch.allclose(op_dist.sort().values, ref_dist.sort().values, atol=1e-5)

    def test_cutoff_behavior(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {"Z": torch.tensor([1, 1, 1]), "pos": positions,
                  "targets": {"U0": torch.tensor([0.0])}}

        result = NeighborList(cutoff=2.0, max_num_pairs=10)(sample)
        assert torch.all(result["bond_dist"] <= 2.0)

    def test_task_id_deterministic(self):
        t1 = NeighborList(cutoff=5.0, max_num_pairs=512)
        t2 = NeighborList(cutoff=5.0, max_num_pairs=512)
        assert t1.task_id == t2.task_id

        t3 = NeighborList(cutoff=6.0)
        assert t1.task_id != t3.task_id


class TestAtomicDress:
    def test_is_dataset_task(self):
        t = AtomicDress(elements=[1, 6])
        assert isinstance(t, DatasetTask)
        assert isinstance(t, Runnable)

    def test_fit_and_execute(self):
        samples = [
            {"Z": torch.tensor([1, 1]), "pos": torch.zeros(2, 3), "targets": {"U0": torch.tensor([2.0])}},
            {"Z": torch.tensor([6, 1]), "pos": torch.zeros(2, 3), "targets": {"U0": torch.tensor([7.0])}},
            {"Z": torch.tensor([6, 6]), "pos": torch.zeros(2, 3), "targets": {"U0": torch.tensor([12.0])}},
        ]

        task = AtomicDress(elements=[1, 6], target_key="U0", output_key="U0_dressed")
        task.fit(samples)

        dressed = torch.stack([task(s)["targets"]["U0_dressed"].reshape(-1)[0] for s in samples])
        assert torch.allclose(dressed, torch.zeros_like(dressed), atol=1e-5)

    def test_state_dict_roundtrip(self):
        samples = [
            {"Z": torch.tensor([1, 1]), "pos": torch.zeros(2, 3), "targets": {"U0": torch.tensor([2.0])}},
            {"Z": torch.tensor([6, 1]), "pos": torch.zeros(2, 3), "targets": {"U0": torch.tensor([7.0])}},
        ]

        t1 = AtomicDress(elements=[1, 6])
        t1.fit(samples)
        state = t1.state_dict()

        t2 = AtomicDress(elements=[1, 6])
        t2.load_state_dict(state)
        assert t1.atomic_energies == t2.atomic_energies


class TestPipeline:
    def test_three_registration_methods(self):
        pipe = pipeline("test")

        # Method 1: decorator
        @pipe.task
        def add_tag(sample):
            return {**sample, "tag": "test"}

        # Method 2: Task subclass
        pipe.add(NeighborList(cutoff=5.0, max_num_pairs=10))

        # Method 3: bare callable
        pipe.add(lambda s: {**s, "extra": 1}, name="extra")

        spec = pipe.build()
        assert len(spec.entries) == 3
        assert spec.pipeline_id  # non-empty hash

    def test_isinstance_dispatch(self):
        pipe = pipeline("test")
        pipe.add(AtomicDress(elements=[1, 6]))
        pipe.add(NeighborList(cutoff=5.0, max_num_pairs=10))
        spec = pipe.build()

        assert len(spec.prepare_tasks) == 2
        assert len(spec.batch_tasks) == 0

        assert isinstance(spec.prepare_tasks[0].task, DatasetTask)
        assert isinstance(spec.prepare_tasks[1].task, SampleTask)
