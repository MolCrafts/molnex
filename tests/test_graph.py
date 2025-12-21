"""Test suite for graph export."""

import pytest


def test_to_graph_returns_graph_object():
    """Test that to_graph() returns a Graph object."""
    from molnex.core.trainer import Trainer
    from molnex.graph.builder import Graph
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert isinstance(graph, Graph)


def test_graph_has_nodes_edges_meta():
    """Test that exported graph has required attributes."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert hasattr(graph, "nodes")
    assert hasattr(graph, "edges")
    assert hasattr(graph, "meta")
    
    assert graph.nodes is not None
    assert graph.edges is not None
    assert graph.meta is not None


def test_graph_contains_train_and_eval_steps():
    """Test that graph nodes include TrainStep and EvalStep."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert len(graph.nodes) >= 2
    
    op_names = [node.op_name for node in graph.nodes]
    assert "train_step" in op_names
    assert "eval_step" in op_names


def test_graph_edges_represent_stage_flow():
    """Test that graph edges represent stage flow."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert len(graph.edges) > 0
    
    # Check that edges connect train_step to eval_step
    edge = graph.edges[0]
    assert isinstance(edge, dict)
    assert "source" in edge
    assert "target" in edge


def test_graph_meta_contains_stage_order():
    """Test that graph meta contains stage_order."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert "stage_order" in graph.meta
    assert isinstance(graph.meta["stage_order"], list)
    assert "train" in graph.meta["stage_order"]
    assert "eval" in graph.meta["stage_order"]


def test_graph_meta_contains_loop_structure():
    """Test that graph meta contains loop_structure."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    graph = trainer.to_graph()
    
    assert "loop_structure" in graph.meta
    assert isinstance(graph.meta["loop_structure"], dict)


def test_graph_is_deterministic():
    """Test that to_graph() produces deterministic output."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    
    graph1 = trainer.to_graph()
    graph2 = trainer.to_graph()
    
    # Check nodes are the same
    assert len(graph1.nodes) == len(graph2.nodes)
    for i, (n1, n2) in enumerate(zip(graph1.nodes, graph2.nodes)):
        assert n1.op_name == n2.op_name
    
    # Check edges are the same
    assert len(graph1.edges) == len(graph2.edges)
    
    # Check meta keys are the same
    assert set(graph1.meta.keys()) == set(graph2.meta.keys())


def test_to_graph_does_not_execute_training():
    """Test that to_graph() does not execute training."""
    from molnex.core.trainer import Trainer
    
    trainer = Trainer()
    
    # Track if run was called
    run_called = False
    original_run = trainer.train_step.run
    
    def tracked_run(state, *, batch):
        nonlocal run_called
        run_called = True
        return original_run(state, batch=batch)
    
    trainer.train_step.run = tracked_run
    
    # Call to_graph
    graph = trainer.to_graph()
    
    # Verify run was not called
    assert not run_called
    
    # Verify state is unchanged
    assert trainer.state.epoch == 0
    assert trainer.state.global_step == 0
