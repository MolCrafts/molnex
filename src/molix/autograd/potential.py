"""The potential contract: ``energy`` writes, ``forward`` derives.

A molnex potential has exactly two public methods and no layer between them:

``energy(frame) -> frame``
    The one method a model writes. It reads its columns straight off the
    frame — ``atoms.pos``, ``atoms.Z``, ``pairs.pair_index``, optionally
    ``pairs.pair_diff`` / ``atoms.mask`` and whatever per-graph conditioning it
    takes — computes the per-atom energy ``(N,)`` and writes ``atoms.energy``
    and ``graphs.energy = scatter_sum(per_atom, atoms.batch, B)``. It derives
    nothing, detaches nothing and validates nothing. This is what
    :class:`~molix.autograd.derivative.Derivative` differentiates.

``forward(frame) -> frame``
    ``self.derivative(frame)``: the energy plus the derivatives the ``wrt``
    keys name, one energy forward and one ``autograd.grad``. Provided here;
    a subclass never overrides it.

A subclass builds the op as the **last line** of ``__init__``::

    self._derivative = Derivative(self.energy, wrt=wrt)

A bound method registers no submodule, so the op carries no parameters of its
own and the ``state_dict`` keys of a potential stay flat — the checkpoint of a
potential is the checkpoint of its body.

The ``graphs`` namespace is required: ``B = frame["graphs"].batch_size[0]`` is
a static ``int`` and the only graph count an ``energy`` reads. Every producer
of a frame (collate, the MD bind path, the deploy ``build_batch``) writes that
namespace, so there is no host-sync fallback (``atoms.batch.max().item()``)
anywhere on the path.

Element-table validation is the encoder's eager precondition — run one eager
forward, then compile — not a check inside ``energy``.
"""

from __future__ import annotations

import abc

import torch.nn as nn
from tensordict import TensorDictBase

from molix.autograd.derivative import Derivative

__all__ = ["Potential"]


class Potential(nn.Module, abc.ABC):
    """Base class of every molnex potential: abstract ``energy``, ``forward`` = the op.

    Subclasses set ``self._derivative = Derivative(self.energy, wrt=wrt)`` as
    the last line of ``__init__``; everything else is provided.

    Attributes:
        _derivative: The op :meth:`forward` runs, built by the subclass over
            its own :meth:`energy`.
    """

    _derivative: Derivative

    @abc.abstractmethod
    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)`` in place.

        Args:
            frame: Post-collate batch (or its :class:`~molix.frame.TensorFrame`
                view) carrying ``atoms`` and ``pairs`` and the ``graphs``
                namespace at ``batch_size=[B]``.

        Returns:
            The same ``frame``, both energy columns written.
        """

    def forward(self, frame: TensorDictBase) -> TensorDictBase:
        """The frame with ``graphs.energy`` and the ``wrt`` derivatives written: the op."""
        return self._derivative(frame)

    @property
    def derivative(self) -> Derivative:
        """The op :meth:`forward` runs — what a training step marks its leaves through."""
        return self._derivative
