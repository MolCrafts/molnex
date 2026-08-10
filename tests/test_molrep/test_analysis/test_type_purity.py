import torch

from molrep.analysis import AtomLatentTable, NearestNeighbourTypePurity


class TestNearestNeighbourTypePurity:
    def test_separated_clusters(self):
        feats = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        y = torch.tensor([0, 0, 1, 1])
        t = AtomLatentTable(feats, ["m"] * 4, y)
        r = NearestNeighbourTypePurity(k=1).score(t)
        assert r.mean_purity == 1.0

    def test_alternating_line(self):
        feats = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        y = torch.tensor([0, 1, 0, 1])
        t = AtomLatentTable(feats, ["m"] * 4, y)
        r = NearestNeighbourTypePurity(k=1).score(t)
        assert r.mean_purity == 0.0

    def test_all_unlabeled(self):
        feats = torch.zeros(3, 2)
        y = torch.tensor([-1, -1, -1])
        t = AtomLatentTable(feats, ["m"] * 3, y)
        r = NearestNeighbourTypePurity(k=1).score(t)
        assert r.mean_purity is None
