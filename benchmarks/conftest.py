"""Shared fixtures for MolNex benchmarks."""

import pytest
import torch

from molix.data.types import AtomData, EdgeData, GraphBatch

# ---------- size parametrization ----------

SMALL = {"n_nodes": 64, "n_edges": 256, "n_graphs": 4}
MEDIUM = {"n_nodes": 512, "n_edges": 4096, "n_graphs": 16}
LARGE = {"n_nodes": 2048, "n_edges": 16384, "n_graphs": 64}


@pytest.fixture(params=["small", "medium", "large"], ids=["small", "medium", "large"])
def graph_size(request):
    """Return graph size parameters."""
    return {"small": SMALL, "medium": MEDIUM, "large": LARGE}[request.param]


@pytest.fixture
def graph_data(graph_size):
    """Synthetic molecular graph data as raw tensors."""
    n_nodes = graph_size["n_nodes"]
    n_edges = graph_size["n_edges"]
    n_graphs = graph_size["n_graphs"]

    Z = torch.randint(0, 5, (n_nodes,))
    pos = torch.randn(n_nodes, 3)
    batch = torch.arange(n_graphs).repeat_interleave(n_nodes // n_graphs)

    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (n_edges,)),
            torch.randint(0, n_nodes, (n_edges,)),
        ],
        dim=1,
    )
    bond_diff = torch.randn(n_edges, 3)
    bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1, max=4.9)

    return {
        "Z": Z,
        "pos": pos,
        "batch": batch,
        "edge_index": edge_index,
        "bond_diff": bond_diff,
        "bond_dist": bond_dist,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_graphs": n_graphs,
    }


@pytest.fixture
def graph_batch_td(graph_data):
    """Build a GraphBatch TensorDict from graph_data."""
    atoms = AtomData(
        Z=graph_data["Z"],
        pos=graph_data["pos"],
        batch=graph_data["batch"],
        batch_size=[graph_data["n_nodes"]],
    )
    edges = EdgeData(
        edge_index=graph_data["edge_index"],
        bond_diff=graph_data["bond_diff"],
        bond_dist=graph_data["bond_dist"],
        batch_size=[graph_data["n_edges"]],
    )
    return GraphBatch(atoms=atoms, edges=edges, batch_size=[])
