"""Potential composer: head -> potential(mixing -> energy) -> forces.

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

Units are the model's own: whatever the head emits, the terms consume and the
energy comes back in. This path is *not* the unit-pinned Class-I force field —
that is :class:`~molpot.forcefield.Potentials`, whose Lennard-Jones rule is a
recorded :class:`~molpot.potentials.vdw.lj126.MixingRule` rather than a free
callable, because there the rule is part of the exported model's identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molpot.derivation import ForceDerivation


class PotentialComposer(nn.Module):
    """Compose parameter head + potentials into a force field.

    Pipeline::

        node_features → head → atom_params (dict)
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
            ``head(node_features, **head_kwargs)``.
        potentials: Name → potential term. Each must expose a ``mixing_fn``
            attribute and accept the keys that rule returns as keyword arguments
            of its ``forward``, alongside ``distance`` / ``pair_batch`` /
            ``num_graphs``. Names label the per-term energies in the output.

    Raises:
        ValueError: ``potentials`` is empty — a composed force field with no
            term has no energy to return.
    """

    def __init__(
        self,
        head: nn.Module,
        potentials: dict[str, nn.Module],
    ):
        super().__init__()
        if not potentials:
            raise ValueError("PotentialComposer requires at least one potential.")
        self.head = head
        self.potentials = nn.ModuleDict(potentials)
        self.force_derivation = ForceDerivation()

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        data: dict[str, torch.Tensor],
        compute_forces: bool = False,
        head_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute composed energy and optionally forces.

        Args:
            node_features: Pooled per-node features ``(N, D)``.
            data: Must contain ``"pair_index"`` ``(E, 2)``,
                ``"batch"`` ``(N,)``. Must contain ``"pos"`` ``(N, 3)``
                if ``compute_forces=True``.
            compute_forces: Derive forces as ``-dE/dpos``.
            head_kwargs: Extra keyword arguments forwarded to the head
                (e.g. ``batch``, ``Z`` for MultiHead).

        Returns:
            Dict with ``"energy"``, ``"term_energies"``,
            ``"parameters"``, and optionally ``"forces"``.
        """
        pair_index = data["pair_index"]
        batch = data["batch"]
        src, dst = pair_index[:, 0], pair_index[:, 1]

        # 1. Head: features → per-atom params
        atom_params = self.head(node_features, **(head_kwargs or {}))

        # 2. Distance (recompute from pos so forces can flow pos -> distance)
        pos = data.get("pos")
        if pos is not None:
            distance = (pos[dst] - pos[src]).norm(dim=-1)
        else:
            distance = data["pair_dist"]

        # 3. Evaluate each potential (each applies its own mixing)
        pair_batch = batch[src]
        num_graphs = data.get("num_graphs")

        def _term_energies(distance: torch.Tensor) -> dict[str, torch.Tensor]:
            terms: dict[str, torch.Tensor] = {}
            for name, potential in self.potentials.items():
                pair_params = potential.mixing_fn(atom_params, pair_index)
                terms[name] = potential(
                    distance=distance,
                    pair_batch=pair_batch,
                    num_graphs=num_graphs,
                    **pair_params,
                )
            return terms

        term_energies = _term_energies(distance)
        total_energy: torch.Tensor | None = None
        for energy in term_energies.values():
            total_energy = energy if total_energy is None else total_energy + energy
        assert total_energy is not None

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "energy": total_energy,
            "term_energies": term_energies,
            "parameters": atom_params,
        }

        # 4. Forces via functorch: F = -∂E/∂pos, with the energy traced as a
        #    pure function of positions (recomputing edge distances inside).
        if compute_forces:
            if pos is None:
                raise RuntimeError("compute_forces=True requires data['pos'].")

            def energy_fn(p: torch.Tensor) -> torch.Tensor:
                d = (p[dst] - p[src]).norm(dim=-1)
                total = None
                for e in _term_energies(d).values():
                    total = e if total is None else total + e
                return total.sum()

            outputs["forces"] = self.force_derivation(energy_fn, pos)

        return outputs
