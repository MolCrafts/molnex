"""Protocol definitions for structural compatibility with MolExp.

These protocols are LOCAL to MolNex and exist for:
- Documentation
- Local type checking
- Internal consistency

They do NOT imply any shared dependency with MolExp.
Compatibility is achieved through duck typing (structural compatibility).
"""

from typing import Any, Mapping, Protocol, Sequence


class TaskProtocol(Protocol):
    """Protocol defining the shape for executable tasks.

    This is a MolNex-local definition for documentation and type checking.
    No inheritance or runtime checks against MolExp are performed.

    Task objects (TrainTask, EvalTask, DataNode, etc.) satisfy this protocol structurally.
    """

    task_id: str
    """Unique identifier for this task."""

    def input_schema(self) -> Mapping[str, Any]:
        """Return input schema specification.

        Returns:
            Mapping describing expected inputs
        """
        ...

    def output_schema(self) -> Mapping[str, Any]:
        """Return output schema specification.

        Returns:
            Mapping describing output structure
        """
        ...

    def execute(self, **inputs) -> Mapping[str, Any]:
        """Execute the task.

        Args:
            **inputs: Input data

        Returns:
            Mapping with execution results
        """
        ...


class WorkflowProtocol(Protocol):
    """Protocol defining the shape for workflows.

    This is a MolNex-local definition for documentation and type checking.
    No inheritance or runtime checks against MolExp are performed.

    Workflow objects (Trainer, DataPipeline, etc.) satisfy this protocol structurally.
    """

    tasks: Sequence[TaskProtocol]
    """Sequence of task nodes."""

    links: Sequence[Any]
    """Sequence of links representing flow between tasks."""

    metadata: Mapping[str, Any]
    """Metadata about the workflow (stage ordering, loop structure, etc.)."""
