import torch

from molpot.pooling import MaxPooling


class TestMaxPooling:
    def test_max_pooling(self):
        pooling = MaxPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pooling(x, batch)
        assert torch.allclose(out, torch.tensor([[3.0, 2.0], [10.0, 20.0]]))

    def test_max_pooling_1d(self):
        out = MaxPooling()(torch.tensor([1.0, 5.0, 2.0, -3.0]), torch.tensor([0, 0, 1, 1]))
        assert torch.equal(out, torch.tensor([5.0, 2.0]))

    def test_max_pooling_unsorted_batch(self):
        """Rows need not be grouped by graph — the scatter handles any order."""
        x = torch.tensor([[7.0], [1.0], [9.0], [2.0]])
        out = MaxPooling()(x, torch.tensor([1, 0, 1, 0]), 2)
        assert torch.equal(out, torch.tensor([[2.0], [9.0]]))

    def test_max_pooling_empty_graph_is_neg_inf(self):
        """A graph with no atoms keeps the -inf identity (documented behaviour)."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = MaxPooling()(x, torch.tensor([0, 2]), 3)
        assert torch.equal(out[0], torch.tensor([1.0, 2.0]))
        assert torch.isinf(out[1]).all() and (out[1] < 0).all()
        assert torch.equal(out[2], torch.tensor([3.0, 4.0]))

    def test_max_pooling_is_differentiable(self):
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        MaxPooling()(x, torch.tensor([0, 0]), 1).sum().backward()
        # Gradient reaches exactly the arg-max element of each feature column.
        assert torch.equal(x.grad, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
