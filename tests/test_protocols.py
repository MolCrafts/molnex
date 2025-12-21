"""Test suite for protocol conformance."""

import pytest


def test_train_step_has_op_name():
    """Test that TrainStep has op_name attribute."""
    from molnex.steps.train_step import TrainStep
    
    step = TrainStep()
    assert hasattr(step, "op_name")
    assert isinstance(step.op_name, str)
    assert step.op_name == "train_step"


def test_train_step_has_input_schema():
    """Test that TrainStep has input_schema method."""
    from molnex.steps.train_step import TrainStep
    
    step = TrainStep()
    assert hasattr(step, "input_schema")
    assert callable(step.input_schema)
    
    schema = step.input_schema()
    assert isinstance(schema, dict)
    assert "train_state" in schema
    assert "batch" in schema


def test_train_step_has_output_schema():
    """Test that TrainStep has output_schema method."""
    from molnex.steps.train_step import TrainStep
    
    step = TrainStep()
    assert hasattr(step, "output_schema")
    assert callable(step.output_schema)
    
    schema = step.output_schema()
    assert isinstance(schema, dict)
    assert "loss" in schema
    assert "result" in schema
    assert "logs" in schema


def test_train_step_has_run_method():
    """Test that TrainStep has run method with correct signature."""
    from molnex.core.state import TrainState
    from molnex.steps.train_step import TrainStep
    
    step = TrainStep()
    assert hasattr(step, "run")
    assert callable(step.run)
    
    # Test run method works
    state = TrainState()
    batch = {"data": [1, 2, 3]}
    result = step.run(state, batch=batch)
    
    assert isinstance(result, dict)
    assert "loss" in result
    assert "result" in result
    assert "logs" in result


def test_eval_step_protocol_conformance():
    """Test that EvalStep conforms to OpLike protocol."""
    from molnex.core.state import TrainState
    from molnex.steps.eval_step import EvalStep
    
    step = EvalStep()
    
    # Check attributes and methods
    assert hasattr(step, "op_name")
    assert step.op_name == "eval_step"
    assert hasattr(step, "input_schema")
    assert hasattr(step, "output_schema")
    assert hasattr(step, "run")
    
    # Test schemas
    input_schema = step.input_schema()
    assert "train_state" in input_schema
    assert "batch" in input_schema
    
    output_schema = step.output_schema()
    assert "loss" in output_schema
    assert "result" in output_schema
    assert "logs" in output_schema
    
    # Test run
    state = TrainState()
    result = step.run(state, batch={})
    assert isinstance(result, dict)


def test_test_step_protocol_conformance():
    """Test that TestStep conforms to OpLike protocol."""
    from molnex.steps.test_step import TestStep
    
    step = TestStep()
    assert hasattr(step, "op_name")
    assert step.op_name == "test_step"
    assert hasattr(step, "input_schema")
    assert hasattr(step, "output_schema")
    assert hasattr(step, "run")


def test_predict_step_protocol_conformance():
    """Test that PredictStep conforms to OpLike protocol."""
    from molnex.steps.predict_step import PredictStep
    
    step = PredictStep()
    assert hasattr(step, "op_name")
    assert step.op_name == "predict_step"
    assert hasattr(step, "input_schema")
    assert hasattr(step, "output_schema")
    assert hasattr(step, "run")


def test_graph_has_required_attributes():
    """Test that Graph has required GraphLike protocol attributes."""
    from molnex.graph.builder import Graph
    
    graph = Graph()
    
    # Check required attributes
    assert hasattr(graph, "nodes")
    assert hasattr(graph, "edges")
    assert hasattr(graph, "meta")
    
    # Check types
    assert isinstance(graph.nodes, (list, tuple))
    assert isinstance(graph.edges, (list, tuple))
    assert isinstance(graph.meta, dict)


def test_graph_with_nodes():
    """Test that Graph can hold step nodes."""
    from molnex.graph.builder import Graph
    from molnex.steps.train_step import TrainStep
    from molnex.steps.eval_step import EvalStep
    
    train_step = TrainStep()
    eval_step = EvalStep()
    
    graph = Graph(
        nodes=[train_step, eval_step],
        edges=[{"source": "train_step", "target": "eval_step"}],
        meta={"stage_order": ["train", "eval"]},
    )
    
    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_name == "train_step"
    assert graph.nodes[1].op_name == "eval_step"
    assert len(graph.edges) == 1
    assert "stage_order" in graph.meta
