"""Test suite for MolNex isolation from MolExp."""

import sys
import pytest


def test_molnex_imports_without_molexp():
    """Test that MolNex can be imported without MolExp."""
    # This test verifies that importing molnex doesn't require molexp
    try:
        import molnex
        from molnex import Trainer, TrainState, Stage
        from molnex.steps.train_step import TrainStep
        from molnex.steps.eval_step import EvalStep
        
        # If we get here, imports succeeded
        assert True
    except ImportError as e:
        if "molexp" in str(e).lower():
            pytest.fail(f"MolNex import failed due to MolExp dependency: {e}")
        else:
            raise


def test_no_molexp_imports_in_codebase():
    """Test that MolNex code doesn't import MolExp."""
    # Check that molexp is not in sys.modules after importing molnex
    # (This is a runtime check, not a static analysis)
    
    # Clear any existing molexp imports
    molexp_modules = [key for key in sys.modules.keys() if key.startswith("molexp")]
    for mod in molexp_modules:
        del sys.modules[mod]
    
    # Import molnex
    import molnex
    
    # Check that molexp was not imported
    molexp_modules_after = [key for key in sys.modules.keys() if key.startswith("molexp")]
    assert len(molexp_modules_after) == 0, f"MolExp modules found: {molexp_modules_after}"


def test_training_runs_without_molexp():
    """Test that training execution works without MolExp."""
    from molnex import Trainer
    
    class DummyDataModule:
        def train_dataloader(self):
            yield {"data": 1}
            yield {"data": 2}
        
        def val_dataloader(self):
            yield {"data": 3}
    
    trainer = Trainer()
    datamodule = DummyDataModule()
    
    # This should work without molexp
    final_state = trainer.train(datamodule, max_epochs=1)
    
    assert final_state.epoch == 1
    assert final_state.global_step == 2


def test_graph_export_without_molexp():
    """Test that graph export works without MolExp."""
    from molnex import Trainer
    
    trainer = Trainer()
    
    # This should work without molexp
    graph = trainer.to_graph()
    
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert len(graph.meta) > 0


def test_step_execution_without_molexp():
    """Test that step execution works without MolExp."""
    from molnex import TrainState
    from molnex.steps.train_step import TrainStep
    
    step = TrainStep()
    state = TrainState()
    batch = {"data": [1, 2, 3]}
    
    # This should work without molexp
    result = step.run(state, batch=batch)
    
    assert isinstance(result, dict)
    assert "loss" in result
    assert "result" in result
    assert "logs" in result
