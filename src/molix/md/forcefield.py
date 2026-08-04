"""ForceField components: bind an energy model to a system, expose live forces.

Mirrors molpy's distinction. In molpy a ``Potential`` is the functional model and
the *bound, evaluable* thing is produced by a ``ForceField`` (cf. molpy
``Potentials`` / the ``PotentialLike`` ``calc_energy`` / ``calc_forces``
protocol). Here:

* a **Potential** is a :class:`molpot.BasePotential` / ``PiNetPotential`` —
  energy from a batch ``TensorDict``;
* a **ForceField** (this module) is an :class:`torch.nn.Module` that *binds* a
  Potential (or an analytic form) to a system and maps positions ``(N, 3)`` to a
  :class:`~molix.md.types.ForceOutput`. The ``Integrator`` consumes a
  ``ForceField`` component — never a closure.

Scope: open (non-periodic) systems, frozen neighbour list, small displacements
(see :class:`PotentialForceField`). PBC / minimum-image and neighbour-list
rebuild are out of scope.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from molix.md.types import ForceOutput


class ForceField(nn.Module):
    """Abstract energy + force provider over positions (molpy ``PotentialLike``).

    Subclasses implement :meth:`forward` returning a
    :class:`~molix.md.types.ForceOutput`. :meth:`calc_energy` / :meth:`calc_forces`
    are conveniences mirroring molpy's protocol; override :meth:`calc_energy` when
    an energy-only path is cheaper than a full force evaluation.
    """

    def forward(self, pos: torch.Tensor) -> ForceOutput:  # noqa: D102
        raise NotImplementedError

    def calc_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Scalar energy ``()`` at ``pos``."""
        return self(pos).energy

    def calc_forces(self, pos: torch.Tensor) -> torch.Tensor:
        """Forces ``(N, 3)`` at ``pos``."""
        return self(pos).forces


class PotentialForceField(ForceField):
    """Bind a molpot Potential to a fixed system template.

    The collated template carries a *precomputed* ``edges.edge_diff`` /
    ``edge_dist`` from its build-time positions; PiNet's ``_edge_bond_diff`` would
    use that as a straight-through *value*, freezing the PES (constant force) if
    left in place. The template is therefore stripped of those keys **once** so
    the Potential recomputes geometry from the live positions every call (correct
    for **open** systems). The neighbour list (``edge_index``) is **not** rebuilt:
    valid for short, small-displacement, non-periodic trajectories only.

    Args:
        potential: A molpot Potential / ``PiNetPotential``; its
            ``forward(td, compute_forces=True)`` returns ``{"energy", "forces"}``
            and must not mutate ``td`` (PiNet clones internally — safe to reuse
            the template).
        template: System ``TensorDict`` (``atoms.Z``, ``edges.edge_index``,
            ``atoms.batch``, ``graphs``); ``("atoms", "pos")`` is replaced per call.
        energy_scale: Unit-bridge multiplier applied to energy and forces, e.g.
            ``1 / molix.md.integrators.EV_PER_AMU_A2_FS2`` to drive an eV/Å
            potential in the integrator's (amu, Å, fs) system. Default ``1.0``.
    """

    _STALE_EDGE_KEYS = ("edge_diff", "edge_dist")

    def __init__(
        self, potential: nn.Module, template: TensorDict, *, energy_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.potential = potential
        batch = template.clone()
        for key in self._STALE_EDGE_KEYS:
            if ("edges", key) in batch.keys(include_nested=True):
                del batch["edges", key]
        # Working batch: reuse structure and only replace pos each step
        # (full TensorDict.clone() every MD step was a measurable alloc cost).
        # Potential paths that need isolation (PiNet) clone internally.
        self._template = batch
        self._work = batch.clone()
        ref = batch["atoms", "pos"]
        self._device = ref.device
        self._dtype = ref.dtype
        self.register_buffer("energy_scale", torch.as_tensor(float(energy_scale)))

    def _batch_at(self, pos: torch.Tensor) -> TensorDict:
        """Bind live positions into the reusable working batch (in-place pos)."""
        self._work["atoms", "pos"] = pos.to(device=self._device, dtype=self._dtype)
        return self._work

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        batch = self._batch_at(pos)
        out = self.potential(batch, compute_forces=True)
        energy = out["energy"].sum().detach() * self.energy_scale
        forces = out["forces"].detach() * self.energy_scale
        return ForceOutput(energy, forces)

    def calc_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Energy only — skips the force derivation (cheaper than :meth:`forward`)."""
        batch = self._batch_at(pos)
        out = self.potential(batch, compute_forces=False)
        return out["energy"].sum().detach() * self.energy_scale


class HarmonicForceField(ForceField):
    """Isotropic harmonic well ``E = ½k‖x‖²``, ``F = -kx``.

    Analytic, model-free reference used to unit-test the integrator (energy
    conservation, equipartition) without a Potential. Replaces the legacy
    ``_harmonic(k)`` closure with a component on the same interface.

    Args:
        k: Spring constant.
    """

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("k", torch.as_tensor(float(k)))

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        energy = 0.5 * self.k * (pos * pos).sum()
        return ForceOutput(energy, -self.k * pos)


class LennardJonesForceField(ForceField):
    """All-pairs Lennard-Jones (no cutoff): ``E = Σ_{i<j} 4ε[(σ/r)¹² − (σ/r)⁶]``.

    Every pair is included, so there is **no neighbour list to go stale** — the
    energy/force are exact for any displacement, which is why this is the
    component used for the long NVE energy-conservation validation (a stiff,
    anharmonic, realistic PES — a far stronger integrator test than a harmonic
    well). Forces are the analytic closed form (compile-friendly, no autograd);
    a test asserts they equal ``-∂E/∂pos``.

    Args:
        epsilon: Well depth ε.
        sigma: Finite-distance σ (where the pair energy is zero).

    Reference:
        Lennard-Jones, "On the Determination of Molecular Fields", Proc. R. Soc.
        Lond. A 106 (1924) 463. https://doi.org/10.1098/rspa.1924.0082
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("epsilon", torch.as_tensor(float(epsilon)))
        self.register_buffer("sigma", torch.as_tensor(float(sigma)))

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        n = pos.shape[0]
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3): diff[i, j] = r_j - r_i
        r2 = (diff * diff).sum(-1)  # (N, N)
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        r2 = r2.masked_fill(eye, 1.0)  # avoid 1/0 on the diagonal (masked out below)
        inv_r2 = (self.sigma * self.sigma) / r2
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6
        pair_e = (4.0 * self.epsilon * (inv_r12 - inv_r6)).masked_fill(eye, 0.0)
        energy = 0.5 * pair_e.sum()
        # F_i = Σ_j 24ε/r²·(2(σ/r)¹² − (σ/r)⁶)·(r_i − r_j);  r_i − r_j = -diff[i, j]
        coef = (24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / r2).masked_fill(eye, 0.0)
        force = (coef.unsqueeze(-1) * (-diff)).sum(1)  # (N, 3)
        return ForceOutput(energy, force)
