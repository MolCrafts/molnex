# molix

`molix` is the training and execution layer.

## Ownership

`molix` owns:

- training and evaluation loops
- state tracking and lifecycle management
- orchestration logic
- hooks and execution-time extension points
- the framework-side movement of data through execution

## Non-Goals

`molix` should not own:

- learned representation modules
- potential construction
- reference model families
- model-specific architecture policy

## Stack Position

`molix` runs the stack. It should be able to execute models assembled from `molrep`, `molpot`, and `molzoo` without absorbing responsibility for how those models are defined.

## Next Pages

- [molix implementation docs](../molix/index.md)
- [Architecture](../architecture.md)
