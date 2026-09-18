"""What a :class:`~molix.autograd.Derivative` differentiates against.

The public surface of this module is two pure functions: :func:`leaf`, the
``requires_grad`` copy every derivative in molnex is taken against, and
:func:`symmetrize_strain_grad`, the strain-gradient convention every virial
goes through. Behind them sit the **private per-key handlers** the op selects
by TensorFrame key — :func:`variable` is that table lookup:

==================  ========================  ======  =====================================
variable key        output key                scale   what the leaf perturbs
==================  ========================  ======  =====================================
``atoms.pos``       ``atoms.forces``          −1      the column itself
``graphs.strain``   ``graphs.virial``         +1      ``atoms.pos``, ``graphs.cell`` and the
                                                      lattice remainder, all by
                                                      ``v ↦ v (I + ε_sym)ᵀ``; the gradient is
                                                      symmetrised (Cauchy: ``σ = W / V``,
                                                      a shear is **not** doubled)
any other column    ``<ns>.<name>_grad``      +1      the column itself
==================  ========================  ======  =====================================

There is no public variable class — a caller only ever names keys — and adding
a kind of variable (a field column for dipoles, say) is one more private
handler plus one more row here.

**The leaf and dynamo.** Dynamo traces :func:`leaf` and the ``autograd.grad``
below it in-graph once ``torch._dynamo.config.trace_autograd_ops`` is on
(:class:`molix.compile.Compiler` sets it; the upstream default is off as of
torch 2.15). What it refuses is a leaf made *inside* the traced function
*escaping*: a returned tensor that still requires grad and derives from it is
"requires_grad_() intermediate leaked as output" and the trace stops there.
Detached outputs (inference) never trip it; a derivative returned attached to
the parameters (training) does until pytorch/pytorch#196653 lands. A leaf the
caller made *before* the region is a graph input, which dynamo accepts — so
every handler reuses a column that already is a live leaf
(:func:`_is_live_leaf`) instead of making one, and
:meth:`~molix.autograd.Derivative.mark_leaves` is the eager prologue that
marks the columns (:meth:`_Variable.mark`). Unmarked, the break is reported
against this file, which owns the seam and its cost, and the training-mode
gate in ``tests/test_molix/test_autograd`` pins exactly that site.

Reference:
    Thompson, Plimpton & Mattson, "General formulation of pressure and stress
    tensor for arbitrary many-body interaction potentials under periodic
    boundary conditions", J. Chem. Phys. 131, 154107 (2009).
    https://doi.org/10.1063/1.3245303
"""

from __future__ import annotations

import torch
from tensordict import TensorDictBase

from molix.schema import FORCES_KEY, POS_KEY, STRAIN_KEY, VIRIAL_KEY

__all__ = ["leaf", "symmetrize_strain_grad"]

#: A TensorFrame key, normalised: ``("atoms", "pos")``.
Key = tuple[str, ...]

#: The cell the strain variable deforms, ``(3, 3)`` or ``(1, 3, 3)`` Å.
CELL_KEY: Key = ("graphs", "cell")


def get_column(frame: TensorDictBase, key: Key) -> torch.Tensor | None:
    """Read ``frame[key]`` by hash lookup, never by enumerating ``keys()``.

    ``key in frame`` / ``key in frame.keys()`` goes through TensorDict
    ``__contains__``, which does ``name in leaf.keys()``. Dynamo then
    guards on *key order* (GH200 2366877: ``graphs.keys()[1] == 'energy'``)
    and recompiles when collate inserts ``graphs`` columns in a different
    sequence. The backend dict ``.get`` does not.
    """
    node: object = frame
    for part in key:
        backend = getattr(node, "_tensordict", None)
        if not isinstance(backend, dict):
            return None
        node = backend.get(part)
        if node is None:
            return None
    return node if torch.is_tensor(node) else None


def leaf(tensor: torch.Tensor) -> torch.Tensor:
    """The fresh ``requires_grad`` leaf a derivative is differentiated against.

    Detaching drops whatever history the caller's tensor carried (so the
    derivative is of this evaluation alone, not of how the positions were
    produced) and ``requires_grad_`` re-attaches the copy as the
    differentiation variable. ``requires_grad_()`` rather than the factory
    keyword: dynamo traces the former on an intermediate and graph-breaks on
    the latter (torch 2.15; pytorch/pytorch#196652 makes them equivalent).

    Args:
        tensor: Any tensor, attached or not — positions ``(N, 3)`` Å, a strain
            ``(3, 3)``, a per-atom column ``(N, …)``.

    Returns:
        A detached copy of ``tensor`` with ``requires_grad=True``, on its dtype
        and device.
    """
    return tensor.detach().requires_grad_(True)


def _is_live_leaf(tensor: torch.Tensor) -> bool:
    """``requires_grad and is_leaf``, spelled as dynamo traces it.

    ``is_leaf`` has no dynamo handler; ``grad_fn`` on a graph input without
    history is the constant ``None`` (on one *with* history it is a break —
    the attached-column case, which is never a training input). The two
    spellings are one predicate: a tensor that requires grad is a leaf iff it
    has no ``grad_fn``.
    """
    return tensor.requires_grad and tensor.grad_fn is None


def symmetrize_strain_grad(grad: torch.Tensor) -> torch.Tensor:
    """molnex strain-gradient protocol: ``W = ∂E/∂ε``, Cauchy ``σ = W / V``.

    ``ε`` is the **symmetric** strain tensor and ``W`` its matrix gradient, so
    ``dE = Σ_αβ W_αβ dε_αβ`` runs over all nine entries and a shear ``W_αβ``
    is *not* doubled. The gradient of a strain-symmetric energy is already
    symmetric; this only removes the asymmetry a finite-precision backward can
    leave behind.

    Doubling the off-diagonal instead is the engineering-shear convention
    (``γ_αβ = 2ε_αβ``, the ``(α,β)``/``(β,α)`` pair as one variable), which
    molnex used until 2026-09-07 and which made every shear component twice
    the Cauchy value — including in the numbers handed to LAMMPS.

    Args:
        grad: Raw ``∂E/∂ε`` ``(..., 3, 3)``.

    Returns:
        The virial ``W`` ``(..., 3, 3)``, symmetric.
    """
    return 0.5 * (grad + grad.transpose(-1, -2))


class _Variable:
    """One row of the key table: where the leaf comes from and where ``∂E/∂·`` goes.

    Five hooks, all straight-line tensor code so the op that calls them stays
    one dynamo graph: :meth:`prepare` (host-side key checks before anything is
    perturbed), :meth:`mark` (a fresh live leaf from the column — what
    :meth:`~molix.autograd.Derivative.mark_leaves` writes into the frame),
    :meth:`leaf` (the column itself when it already is a live leaf, else
    :meth:`mark`), :meth:`apply` (perturb the geometry with it) and
    :meth:`write` (place the gradient). ``leaf`` and ``apply`` are two hooks,
    not one, because the op makes every leaf *before* it forces grad mode on:
    on stock torch the leak break at :func:`leaf` inside a ``with
    torch.enable_grad()`` block resumes with the ``autograd.grad`` outside the
    traced region, and a training backward through the compiled energy is
    then a double backward (``tests/test_molix/test_autograd``).

    Args:
        key: The variable's column.
        output: Where its derivative is written.
    """

    def __init__(self, key: Key, output: Key) -> None:
        self.key = key
        self.output = output

    def prepare(self, frame: TensorDictBase, nested: object) -> None:
        """Check the frame and create what the variable needs; nothing by default."""
        del frame, nested

    def mark(self, buffer: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """A fresh live leaf from the caller's column, shaped as the variable is differentiated.

        Args:
            buffer: The caller's column at :attr:`key`.
            pos: The caller's positions ``(N, 3)`` Å — the dtype and device a
                leaf that is not a position column takes.

        Returns:
            ``buffer.detach().requires_grad_(True)``, cast as the variable needs.
        """
        del pos
        return leaf(buffer)

    def leaf(self, buffer: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """This variable's leaf: the column itself if already a live leaf, else :meth:`mark`.

        Args:
            buffer: The caller's column at :attr:`key`.
            pos: The caller's positions ``(N, 3)`` Å, forwarded to :meth:`mark`.

        Returns:
            The ``requires_grad`` leaf the derivative is taken against.
        """
        return buffer if _is_live_leaf(buffer) else self.mark(buffer, pos)

    def apply(
        self,
        frame: TensorDictBase,
        variable: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor | None,
        shifts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Hand back the geometry as this variable's leaf perturbs it (grad mode on).

        Args:
            frame: The frame being differentiated (a generic column writes
                its leaf straight into it).
            variable: The leaf :meth:`leaf` made.
            pos: Positions ``(N, 3)`` Å as perturbed by the variables before
                this one.
            cell: ``graphs.cell`` as perturbed so far, or ``None``.
            shifts: The lattice remainder ``(E, 3)`` Å as perturbed so far,
                or ``None`` when the frame carries no edge geometry.

        Returns:
            ``(pos, cell, shifts)`` after this variable's perturbation.
        """
        raise NotImplementedError

    def write(self, frame: TensorDictBase, grad: torch.Tensor) -> None:
        """Write ``∂E/∂·`` at :attr:`output` (scale +1)."""
        frame[self.output] = grad


class _Position(_Variable):
    """``atoms.pos → atoms.forces``: the leaf replaces the column, ``F = -∂E/∂pos``."""

    def apply(
        self,
        frame: TensorDictBase,
        variable: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor | None,
        shifts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del frame, pos
        return variable, cell, shifts

    def write(self, frame: TensorDictBase, grad: torch.Tensor) -> None:
        frame[self.output] = -grad


class _Strain(_Variable):
    """``graphs.strain → graphs.virial``: ``ε`` deforms positions, cell and remainder.

    The column is a ``(1, 3, 3)`` zero tensor — created here when absent and
    left on the frame as the variable — and the leaf is its detached copy at
    the same ``(1, 3, 3)``, or the column itself once marked: a reshape of a
    leaf is not a leaf, so the leaf stays the column and only :meth:`apply`
    reads it as ``(3, 3)``. Every Cartesian row vector of the geometry is
    mapped by ``v ↦ v (I + ε_sym)ᵀ`` at ``ε = 0``; the gradient comes back
    symmetrised at ``(1, 3, 3)``.
    """

    def prepare(self, frame: TensorDictBase, nested: object) -> None:
        del nested
        if get_column(frame, CELL_KEY) is None:
            raise ValueError("Derivative wrt graphs.strain requires graphs.cell")
        cell = frame[CELL_KEY]
        if cell.dim() == 3 and cell.shape[0] != 1:
            raise ValueError(
                "Derivative wrt graphs.strain takes one graph per batch — one strain leaf "
                f"is one lattice — but graphs.cell holds {cell.shape[0]}"
            )
        if get_column(frame, self.key) is None:
            frame[self.key] = frame[POS_KEY].new_zeros((1, 3, 3))

    def mark(self, buffer: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return leaf(buffer.to(dtype=pos.dtype, device=pos.device))

    def apply(
        self,
        frame: TensorDictBase,
        variable: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor | None,
        shifts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del frame
        strain = variable.reshape(3, 3)
        symmetric = 0.5 * (strain + strain.T)
        transform = (torch.eye(3, dtype=pos.dtype, device=pos.device) + symmetric).T
        strained_cell = None if cell is None else cell.to(dtype=pos.dtype) @ transform
        strained_shifts = None if shifts is None else shifts.to(dtype=pos.dtype) @ transform
        return pos @ transform, strained_cell, strained_shifts

    def write(self, frame: TensorDictBase, grad: torch.Tensor) -> None:
        frame[self.output] = symmetrize_strain_grad(grad).reshape(1, 3, 3)


class _Column(_Variable):
    """Any other column: the leaf replaces it, ``∂E/∂column`` goes to ``<ns>.<name>_grad``."""

    def apply(
        self,
        frame: TensorDictBase,
        variable: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor | None,
        shifts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        frame[self.key] = variable
        return pos, cell, shifts


def variable(key: Key, output: Key | None = None) -> _Variable:
    """The handler for ``key`` — the table lookup :class:`~molix.autograd.Derivative` does.

    Args:
        key: Normalised TensorFrame key of the variable.
        output: Where its derivative lands; ``None`` takes the table's row.

    Returns:
        The private handler for that key.
    """
    if key == POS_KEY:
        return _Position(key, FORCES_KEY if output is None else output)
    if key == STRAIN_KEY:
        return _Strain(key, VIRIAL_KEY if output is None else output)
    return _Column(key, (*key[:-1], key[-1] + "_grad") if output is None else output)
