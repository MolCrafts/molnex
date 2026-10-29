"""The element-table precondition every encoder shares.

An atomic number outside a model's element table would be snapped onto a
neighbouring row by ``searchsorted`` (MACE, PiNet) or read a ``-1`` type
(Allegro) and silently produce a wrong energy. The membership test host-syncs,
so it is an **eager precondition**, not a per-step op: a potential's ``energy``
(and an encoder's ``forward``) spends it through
:meth:`ElementTable._validate_once` on its first eager call, after which the
gate is closed for the life of the module — ``Z`` is constant over a
trajectory, and a wrongly wired model/dataset pair fails on its first batch.

Compile-safe because it is skipped under tracing: ``torch.compiler.is_compiling()``
is a tracing-time constant, so dynamo folds the host-syncing branch out of the
graph and leaves the flag unspent. A traced call therefore validates nothing,
which is why a compiled model must have run one eager forward first.
"""

from __future__ import annotations

import torch

__all__ = ["ElementTable"]


class ElementTable:
    """Element-table gate mixed into every encoder.

    A plain mixin — no ``__init__``, no ``nn.Module`` base — listed before the
    encoder's module base. The encoder registers ``z_table`` itself (a
    ``torch.long`` buffer of its spec's ``atomic_numbers``, in whatever order
    its type machinery wants) and keeps its own ``Z → index`` lookup; only the
    membership precondition lives here.

    Attributes:
        z_table: The element table ``(n_elements,)``, registered by the encoder.
        _elements_validated: Whether the gate has been spent.
    """

    z_table: torch.Tensor
    _elements_validated: bool = False

    def validate_elements(self, Z: torch.Tensor) -> None:
        """Raise if any atomic number in ``Z`` is outside the element table.

        Host-syncs, so **eager only**: :meth:`_validate_once` is how ``forward``
        and a product's ``energy`` spend it; whoever drives a flat seam directly
        (compile / export) calls it once first.

        Args:
            Z: Atomic numbers ``(N,)``.

        Raises:
            ValueError: Naming the atomic numbers outside the table and the table.
        """
        unknown = torch.unique(Z[~torch.isin(Z, self.z_table)])
        if unknown.numel():
            raise ValueError(
                f"atomic numbers {unknown.tolist()} are outside this model's "
                f"{self.z_table.numel()}-element table {self.z_table.tolist()}"
            )

    def _validate_once(self, Z: torch.Tensor) -> None:
        """Spend the gate on the first eager call; never inside a trace."""
        if self._elements_validated or torch.compiler.is_compiling():
            return
        self.validate_elements(Z)
        self._elements_validated = True
