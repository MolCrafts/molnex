import torch

from molrep.analysis import AtomLatentTable, LatentAnalysisArtifacts, LatentPCA2D


class TestLatentPCA2D:
    def test_shape(self):
        t = AtomLatentTable(torch.randn(5, 4), ["m"] * 5)
        c = LatentPCA2D().project(t)
        assert c.shape == (5, 2)


class TestLatentAnalysisArtifacts:
    def test_keys(self, tmp_path):
        t = AtomLatentTable(
            torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]]),
            ["a", "a", "b", "b"],
            torch.tensor([0, 0, 1, 1]),
        )
        m = LatentAnalysisArtifacts(tmp_path).write(t)
        assert set(m) >= {
            "nn_type_purity_mean",
            "nn_type_purity_k",
            "n_atoms",
            "n_labeled",
            "n_scored",
        }
        assert (tmp_path / "latent_points.jsonl").is_file()
        assert (tmp_path / "type_purity.txt").is_file()
        assert m["nn_type_purity_mean"] == 1.0
