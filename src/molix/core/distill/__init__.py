"""Online distillation: a frozen teacher stamps TensorDict columns outside the compiled step.

The student loss reads those columns with :func:`~molix.core.losses.column_match`
(or any other loss that addresses the batch by nested key). The teacher model
has the same forward as the student: ``TensorDict`` in, ``TensorDict`` out.
"""

from molix.core.distill.step import DistillEvalStep, DistillTrainStep
from molix.core.distill.teacher import Teacher

__all__ = [
    "DistillEvalStep",
    "DistillTrainStep",
    "Teacher",
]
