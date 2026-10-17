"""Potential composer: head -> potential(mixing -> energy), on the potential contract.

Assembles a force field out of two learnable halves. A **parameter head** turns
the per-atom feature vectors an encoder produced into per-atom physical
parameters (a well depth, a radius, a dispersion coefficient); a set of
**potential terms** turn those into an energy. The composer owns the step
between them: applying each term's **combining rule** — the function that makes
one parameter per *pair* out of the two per-atom values at the ends of an edge,
also called a mixing rule — and then summing the terms.

The rule is read off the potential (``potential.mixing_fn``) rather than held
here, so a term and the rule that feeds it travel together and the composer
never has to know which parameters a given term wants. Each rule is called as
``mixing_fn(atom_params, pair_index)`` with the head's per-atom dict and the
``(E, 2)`` edge list, and returns a dict of per-pair ``(E,)`` tensors whose keys
are exactly the keyword arguments that term's ``forward`` accepts — by
convention the per-atom key with an ``_ij`` suffix. Key sets differ per term
(:func:`~molpot.potentials.nonbonded.repulsion_mixing` and
:func:`~molpot.potentials.vdw.lj126.lorentz_berthelot` share no keys at all), so
rules are not interchangeable between terms.

Two public methods, as :mod:`molix.autograd.potential` prescribes: ``energy``
reads ``atoms.node_features`` ``(N, D)`` — the pooled per-atom features the
caller wrote — with ``atoms.{Z,pos,batch}`` and ``pairs.pair_index`` (the
imaged ``pairs.pair_diff`` when present), and writes ``graphs.energy``, one
``graphs.energy_<term>`` per term and one ``atoms.<parameter>`` per head
output; ``forward`` is the :class:`~molix.autograd.Derivative` built from
``wrt`` (``("atoms.pos",)`` adds ``atoms.forces``, ``("atoms.pos",
"graphs.strain")`` adds ``graphs.virial`` on a one-graph periodic batch).

Units are the model's own: whatever the head emits, the terms consume and the
energy comes back in. This path is *not* the unit-pinned Class-I force field —
that is :class:`~molpot.forcefield.Potentials`, whose Lennard-Jones rule is a
recorded :class:`~molpot.potentials.vdw.lj126.MixingRule` rather than a free
callable, because there the rule is part of the exported model's identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch.nn as nn
from tensordict import TensorDictBase

from molix.autograd import Derivative, Potential
from molix.schema import ENERGY_KEY
from molrep.embedding.geometry import edge_lengths, edge_vectors

__all__ = ["PotentialComposer"]


class PotentialComposer(Potential):
    """Compose parameter head + potentials into a force field.

    Pipeline::

        atoms.node_features → head → atom_params (dict)
            for each potential:
                potential.mixing_fn(atom_params, pair_index) → pair_params
                potential(distance, **pair_params) → energy

    Each potential carries its own ``mixing_fn``, the combining rule that turns
    per-atom parameters into the per-pair ones that term prices; this class
    applies it, and the module docstring states the contract in full. Terms are
    summed, so ``head`` must emit every key every rule asks for — one head
    serving several terms is the normal case, and rules ignore keys they do not
    need.

    Args:
        head: Maps per-atom features ``(N, D)`` to a dict of per-atom parameters,
            each ``(N,)``, in the model's own units. Called as
            ``head(node_features, batch=atoms.batch, Z=atoms.Z)``; a head
            that needs neither ignores them (:class:`~molpot.composition.MultiHead`
            forwards them to every member).
        potentials: Name → potential term. Each must expose a ``mixing_fn``
            attribute and accept the keys that rule returns as keyword arguments
            of its ``forward``, alongside ``distance`` / ``pair_batch`` /
            ``num_graphs``. Names label the per-term energies
            ``graphs.energy_<name>``.
        wrt: What :meth:`forward` derives beside the energy, as TensorFrame
            keys (:class:`molix.autograd.Derivative`); ``()`` (the default)
            writes the energy alone.

    Raises:
        ValueError: ``potentials`` is empty — a composed force field with no
            term has no energy to return.
    """

    def __init__(
        self,
        head: nn.Module,
        potentials: Mapping[str, nn.Module],
        *,
        wrt: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if not potentials:
            raise ValueError("PotentialComposer requires at least one potential.")
        self.head = head
        self.potentials = nn.ModuleDict(dict(potentials))
        self._derivative = Derivative(self.energy, wrt=wrt)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``graphs.energy`` ``(B,)`` and the term and parameter columns in place.

        Args:
            frame: Post-collate batch carrying ``atoms.node_features`` ``(N, D)``,
                ``atoms.{Z,pos,batch}``, ``pairs.pair_index`` ``(E, 2)`` and the
                ``graphs`` namespace at ``batch_size=[B]``; ``pairs.pair_diff``
                (the imaged displacement) when periodic.

        Returns:
            The same ``frame`` with ``graphs.energy``, ``graphs.energy_<term>``
            per potential and ``atoms.<parameter>`` per head output written.
        """
        pair_index = frame["pairs", "pair_index"]
        batch = frame["atoms", "batch"]
        keys = frame.keys(include_nested=True)
        edge_diff = frame["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        distance = edge_lengths(
            edge_vectors(frame["atoms", "pos"], pair_index, edge_diff=edge_diff)
        )
        pair_batch = batch[pair_index[:, 0]]
        num_graphs = frame["graphs"].batch_size[0]

        atom_params = self.head(frame["atoms", "node_features"], batch=batch, Z=frame["atoms", "Z"])
        total = None
        for name, potential in self.potentials.items():
            term = potential(
                distance=distance,
                pair_batch=pair_batch,
                num_graphs=num_graphs,
                **potential.mixing_fn(atom_params, pair_index),
            )
            frame["graphs", f"energy_{name}"] = term
            total = term if total is None else total + term
        frame[ENERGY_KEY] = total
        for name, value in atom_params.items():
            frame["atoms", name] = value
        return frame
