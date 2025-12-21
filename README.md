# MolNex

**Unified modeling of molecular potentials and properties with physics-aware ML**

MolNex is a standalone ML training system with structural protocol compatibility to MolExp. It provides a clean, ML-focused API for training while maintaining zero coupling with MolExp through duck typing.

## Features

- **Standalone Training System**: Complete independence from MolExp
- **ML-First API**: Uses training terminology (`Trainer`, `TrainState`, `StepResult`)
- **Structural Protocol Compatibility**: Duck-typed compatibility with MolExp for visualization
- **Graph Export**: Semantic graph representation for introspection
- **Zero Coupling**: No imports or runtime dependencies on MolExp
- **Fully Tested**: Comprehensive test suite ensuring isolation and correctness

## Installation

```bash
cd molnex
pip install -e .
```

## Quick Start

```python
from molnex import Trainer, TrainStep, EvalStep

# Define your data module
class MyDataModule:
    def train_dataloader(self):
        # Return your training batches
        for batch in training_data:
            yield batch
    
    def val_dataloader(self):
        # Return your validation batches
        for batch in validation_data:
            yield batch

# Create trainer with custom steps
def forward_pass(batch):
    # Your model forward pass
    return model(batch)

def optimizer_step():
    # Your optimizer step
    optimizer.step()

train_step = TrainStep(forward_fn=forward_pass, optimizer_fn=optimizer_step)
eval_step = EvalStep(forward_fn=forward_pass)

trainer = Trainer(train_step=train_step, eval_step=eval_step)

# Train
datamodule = MyDataModule()
final_state = trainer.train(datamodule, max_epochs=10)

print(f"Training complete: {final_state.epoch} epochs, {final_state.global_step} steps")
```

## Graph Export for Visualization

MolNex can export a semantic graph representation that MolExp can read and visualize:

```python
from molnex import Trainer

trainer = Trainer()
graph = trainer.to_graph()

# Graph structure (compatible with MolExp)
print(f"Nodes: {[node.op_name for node in graph.nodes]}")
print(f"Edges: {graph.edges}")
print(f"Meta: {graph.meta}")

# MolExp can read this graph purely structurally (duck typing)
# No imports or coupling required
```

## Core Components

### Stage

Training stage enumeration:
- `Stage.TRAIN` - Training phase
- `Stage.EVAL` - Evaluation/validation phase
- `Stage.TEST` - Testing phase
- `Stage.PREDICT` - Prediction/inference phase

### TrainState

Training state container tracking:
- `epoch`: Current epoch number
- `global_step`: Global step counter
- `stage`: Current training stage

### StepResult

Result from executing a training step:
- `loss`: Optional loss value
- `result`: Main result/output
- `logs`: Additional logging information

### Step Objects

- **TrainStep**: Executes one training iteration
- **EvalStep**: Executes one validation iteration
- **TestStep**: Executes one test iteration
- **PredictStep**: Executes one inference iteration

All steps conform to the `OpLike` protocol:
- `op_name`: Operation identifier
- `input_schema()`: Returns input specification
- `output_schema()`: Returns output specification
- `run(train_state, *, batch)`: Executes the step

### Trainer

Main training system:
- `train(datamodule, max_epochs)`: Execute training loop
- `to_graph()`: Export semantic graph representation

## Protocol Compatibility

MolNex defines local Protocol classes that duplicate MolExp's expected structural shape:

```python
# OpLike Protocol (MolNex definition)
class OpLike(Protocol):
    op_name: str
    def input_schema(self) -> Mapping[str, Any]: ...
    def output_schema(self) -> Mapping[str, Any]: ...
    def run(self, train_state, *, batch) -> Mapping[str, Any]: ...

# GraphLike Protocol (MolNex definition)
class GraphLike(Protocol):
    nodes: Sequence[OpLike]
    edges: Sequence[Any]
    meta: Mapping[str, Any]
```

**Important**: These protocols are LOCAL to MolNex. Compatibility is achieved through duck typing (structural compatibility), not inheritance or shared code.

## Design Principles

1. **Zero Coupling**: No imports between MolNex and MolExp
2. **Structural Compatibility**: Duck typing via Protocol shapes, not inheritance
3. **ML-First Naming**: Public API uses training terminology, not workflow concepts
4. **Independent Execution**: MolNex owns its training loop
5. **Full Testability**: All components testable without MolExp

## Testing

Run the test suite:

```bash
# With pytest (if available)
pytest tests/ -v

# Or use the standalone test runner
python3 run_tests.py
```

Test coverage:
- Protocol conformance (OpLike, GraphLike)
- Training execution (epoch/step loops, state updates)
- Graph export (structure, determinism)
- Isolation (runs without MolExp)

## Examples

See `examples/basic_training.py` for a complete working example.

## Architecture

```
molnex/
├── src/molnex/
│   ├── core/
│   │   ├── protocols.py    # OpLike, GraphLike protocols
│   │   ├── state.py        # Stage, TrainState, StepResult
│   │   └── trainer.py      # Trainer class
│   ├── steps/
│   │   ├── train_step.py   # TrainStep
│   │   ├── eval_step.py    # EvalStep
│   │   ├── test_step.py    # TestStep
│   │   └── predict_step.py # PredictStep
│   └── graph/
│       └── builder.py      # Graph class
└── tests/
    ├── test_protocols.py   # Protocol conformance tests
    ├── test_trainer.py     # Training execution tests
    ├── test_graph.py       # Graph export tests
    └── test_isolation.py   # Isolation tests
```

## License

See LICENSE file.
