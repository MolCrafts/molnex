"""ForceField components: bind an energy model to a system, expose live forces.

Mirrors molpy's distinction. In molpy a ``Potential`` is the functional model and
the *bound, evaluable* thing is produced by a ``ForceField`` (cf. molpy
``Potentials`` / the ``PotentialLike`` ``calc_energy`` / ``calc_forces``
protocol). Here:

* a **Potential** is a :class:`molpot.BasePotential` / ``PiNetPotential`` —
  energy from a batch ``TensorDict``;
* a **ForceField** (this module) is an :class:`torch.nn.Module` that *binds* a
  Potential (or an analytic form, or any callable) to a system and maps
  positions ``(N, 3)`` to a :class:`~molix.md.types.ForceOutput`. The
  ``Integrator`` consumes a ``ForceField`` component — never a closure.

Precision: a force field owns its own dtype, independent of the trajectory
state's (see :class:`molix.md.driver.MD` — ``MD(dtype=)`` governs the MD side
only; :meth:`MD.set_potential_dtype` casts the force field). Implementations
accept positions in any dtype and return their own; the integrator casts the
output back to the state dtype at the component boundary.

Periodic systems and neighbour-list rebuild are supported through
:meth:`ForceField.rebuild_neighbors` +
:class:`~molix.md.neighbors.NeighborList` (driven on a cadence by
:class:`~molix.md.runner.NeighborListHook`). :class:`PotentialForceField`
keeps its list frozen — valid for open systems and trajectories short enough
that no atom changes neighbours; periodic production runs use
:class:`PeriodicPotentialForceField` (TensorDict potentials consuming
``edges.shifts``), :class:`LennardJonesCutForceField` (analytic lj/cut over
the same rebuildable list), or :class:`CallableForceField` with a rebuildable
list.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn

from molix.md.neighbors import NeighborStrategy
from molix.md.types import ForceOutput
from molix.schema import ENERGY_KEY, FORCES_KEY, has_forces


class ForceField(nn.Module):
    """Abstract energy + force provider over positions (molpy ``PotentialLike``).

    Subclasses implement :meth:`forward` returning a
    :class:`~molix.md.types.ForceOutput`. :meth:`calc_energy` / :meth:`calc_forces`
    are conveniences mirroring molpy's protocol; override :meth:`calc_energy` when
    an energy-only path is cheaper than a full force evaluation.
    """

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        """Energy + forces at ``pos`` ``(N, 3)``."""
        raise NotImplementedError

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Refresh any position-derived connectivity this force field caches.

        A no-op by default: force fields with no neighbour list (the analytic
        ones) and those that deliberately freeze it need do nothing.
        :class:`~molix.md.runner.NeighborListHook` calls this on a step cadence,
        before the force evaluation. Implementations must keep every tensor
        **shape** unchanged so a compiled/graph-captured force path stays valid
        — see :class:`~molix.md.neighbors.NeighborList`.
        """

    def calc_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Scalar energy ``()`` at ``pos``."""
        return self(pos).energy

    def calc_forces(self, pos: torch.Tensor) -> torch.Tensor:
        """Forces ``(N, 3)`` at ``pos``."""
        return self(pos).forces


class PotentialForceField(ForceField):
    """Bind a molpot Potential to a fixed system template.

    The collated template carries a *precomputed* ``edges.edge_diff`` /
    ``edge_dist`` from its build-time positions; PiNet's ``edge_bond_diff`` would
    use that as a straight-through *value*, freezing the PES (constant force) if
    left in place. The template is therefore stripped of those keys **once** so
    the Potential recomputes geometry from the live positions every call (correct
    for **open** systems). The neighbour list (``edge_index``) is **not** rebuilt:
    valid for short, small-displacement, non-periodic trajectories only —
    periodic runs use :class:`PeriodicPotentialForceField`.

    ``.to(dtype)`` / ``.to(device)`` move the working batch together with the
    module parameters (``_apply`` is overridden), so an explicit cast reaches
    the whole bound system.

    Args:
        potential: A molpot Potential / ``PiNetPotential``. Its ``forward(td)``
            writes ``graphs.energy`` and ``atoms.forces`` into the batch, per
            :mod:`molix.schema`. Force derivation is fixed at the potential's
            construction (``compute_forces=True``), not requested per call —
            the monomorphic contract ``torch.compile`` needs.
        template: System ``TensorDict`` (``atoms.Z``, ``edges.edge_index``,
            ``atoms.batch``, ``graphs``); ``("atoms", "pos")`` is replaced per call.
        energy_scale: Unit-bridge multiplier applied to energy and forces, e.g.
            ``1 / molix.units.EV_PER_AMU_A2_FS2`` to drive an eV/Å potential in
            the integrator's (amu, Å, fs) system. Default ``1.0``.
    """

    _STALE_EDGE_KEYS = ("edge_diff", "edge_dist")

    def __init__(
        self, potential: nn.Module, template: TensorDict, *, energy_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.potential = potential
        # Working batch: reuse structure and only replace pos each step
        # (full TensorDict.clone() every MD step was a measurable alloc cost).
        # Potential paths that need isolation (PiNet) clone internally.
        work = template.clone()
        for key in self._STALE_EDGE_KEYS:
            if ("edges", key) in work.keys(include_nested=True):
                del work["edges", key]
        self._work = work
        ref = work["atoms", "pos"]
        self._device = ref.device
        self._dtype = ref.dtype
        self.register_buffer("energy_scale", torch.as_tensor(float(energy_scale)))

    def _apply(self, fn, recurse: bool = True):
        """Extend ``nn.Module._apply`` to the working batch.

        Without this, ``.to(dtype)`` walks parameters/buffers only and leaves
        the bound system (``_work``, plain ``TensorDict``) at its construction
        dtype — an explicit cast that silently does nothing.
        """
        module = super()._apply(fn, recurse)
        self._work = self._work.apply(fn)
        ref = self._work["atoms", "pos"]
        self._device = ref.device
        self._dtype = ref.dtype
        return module

    def _batch_at(self, pos: torch.Tensor) -> TensorDict:
        """Bind live positions into the reusable working batch (in-place pos)."""
        self._work["atoms", "pos"] = pos.to(device=self._device, dtype=self._dtype)
        return self._work

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        batch = self._batch_at(pos)
        out = self.potential(batch)
        if not has_forces(out):
            raise RuntimeError(
                f"{type(self.potential).__name__} wrote no {FORCES_KEY} — an MD force field "
                "needs forces. Potentials fix this at construction now (e.g. "
                "PiNetPotential(..., compute_forces=True)); it is no longer a per-call choice."
            )
        energy = out[ENERGY_KEY].sum().detach() * self.energy_scale
        forces = out[FORCES_KEY].detach() * self.energy_scale
        return ForceOutput(energy, forces)

    def calc_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Scalar energy ``()`` at ``pos``.

        No longer cheaper than :meth:`forward`: whether a potential derives
        forces is fixed when it is constructed, so an energy-only evaluation
        means constructing an energy-only potential.
        """
        out = self.potential(self._batch_at(pos))
        return out[ENERGY_KEY].sum().detach() * self.energy_scale


class PeriodicPotentialForceField(PotentialForceField):
    """Bind a TensorDict potential to a periodic system with a rebuilding list.

    The component that joins the pieces the package already ships: the
    ``rebuild_neighbors`` seam, :class:`~molix.md.runner.NeighborListHook`'s
    cadence, and :class:`~molix.md.neighbors.NeighborList`'s
    fixed-capacity buffers. The working batch's ``edges`` namespace holds the
    list's live ``edge_index`` ``(capacity, 2)`` and ``shifts``
    ``(capacity, 3)`` **by reference**, so an in-place rebuild is visible to
    the potential with every tensor shape unchanged (CUDA-graph safe).

    The potential's ``forward(td)`` must consume ``edges.shifts`` for periodic
    correctness (e.g. :class:`molzoo.MACEMatpes`); dead padding edges
    self-annihilate through the cutoff envelope.

    Args:
        potential: TensorDict potential writing ``graphs.energy`` /
            ``atoms.forces`` and reading ``edges.shifts``.
        template: System ``TensorDict``; its ``edges`` namespace is replaced by
            the neighbour list's buffers.
        neighbors: The system's rebuilding neighbour list.
        energy_scale: Unit-bridge multiplier applied to energy and forces.
    """

    def __init__(
        self,
        potential: nn.Module,
        template: TensorDict,
        *,
        neighbors: NeighborStrategy,
        energy_scale: float = 1.0,
    ) -> None:
        super().__init__(potential, template, energy_scale=energy_scale)
        self.neighbors = neighbors
        self._bind_neighbors()

    def _bind_neighbors(self) -> None:
        """Point the working batch's ``edges`` at the list's live buffers."""
        self._work["edges"] = TensorDict(
            {"edge_index": self.neighbors.edge_index, "shifts": self.neighbors.shifts},
            batch_size=[self.neighbors.capacity],
        )

    def _apply(self, fn, recurse: bool = True):
        """Cast the neighbour list alongside the module, then re-bind.

        ``TensorDict.apply`` in the parent produces new leaf tensors, which
        would silently sever the by-reference tie to the list's buffers — a
        later ``rebuild`` would then update tensors the potential no longer
        sees.
        """
        module = super()._apply(fn, recurse)
        neighbors = getattr(self, "neighbors", None)
        if neighbors is not None:
            ref = self._work["atoms", "pos"]
            neighbors.to(ref.device, ref.dtype)
            self._bind_neighbors()
        return module

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Refresh the cutoff graph in place (shapes unchanged)."""
        self.neighbors.rebuild(pos)


class CallableForceField(ForceField):
    """Adapt any ``pos -> (energy, forces)`` callable to the ForceField contract.

    The escape hatch for force providers that are not TensorDict potentials —
    an AOTI-exported ``.pt2``, a :class:`molix.engine.StaticForward`, a
    compiled energy core with hand-rolled autograd, an external engine. The
    callable may return a :class:`~molix.md.types.ForceOutput` or a plain
    ``(energy, forces)`` tuple.

    Args:
        fn: Maps positions ``(N, 3)`` to scalar energy ``()`` and forces
            ``(N, 3)``.
        neighbors: Optional rebuildable neighbour list;
            :meth:`rebuild_neighbors` delegates to it so the standard
            ``MD(rebuild_every=)`` cadence works.
        energy_scale: Unit-bridge multiplier applied to energy and forces.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        *,
        neighbors: NeighborStrategy | None = None,
        energy_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self._fn = fn
        self.neighbors = neighbors
        self.register_buffer("energy_scale", torch.as_tensor(float(energy_scale)))

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Delegate to the neighbour list, when one is bound."""
        if self.neighbors is not None:
            self.neighbors.rebuild(pos)

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        energy, forces = self._fn(pos)
        return ForceOutput(energy * self.energy_scale, forces * self.energy_scale)


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


class LennardJonesCutForceField(ForceField):
    """Truncated(-shifted) Lennard-Jones over a rebuildable neighbour list (``lj/cut``).

    The bulk counterpart of :class:`LennardJonesForceField` (which is all-pairs
    and open): pair interactions are evaluated on the fixed-capacity buffers of
    a :class:`~molix.md.neighbors.NeighborStrategy` and truncated at ``cutoff``,
    LAMMPS ``pair_style lj/cut`` style. Energy and forces are the analytic
    closed form (no autograd) over tensors whose **shapes never change** across
    rebuilds, so ``forward`` stays ``torch.compile(fullgraph=True)`` /
    CUDA-graph capturable while the list rebuilds eagerly between force
    evaluations (``MD(rebuild_every=)``).

    Pairs beyond ``cutoff`` — including the list's dead padding edges, whose
    shift is ``DEAD_EDGE_CUTOFF_FACTOR × cutoff`` — contribute exactly zero
    energy and force. With ``shift=True`` (default) the pair energy is shifted
    by ``E_lj(cutoff)`` so it reaches zero *continuously* at the cutoff;
    without the shift, every pair crossing r_cut steps the total energy by
    ``E_lj(r_cut)``, which reads as noise/drift in an NVE total-energy trace.
    The forces are identical under both conventions.

    Args:
        epsilon: Well depth ε, in the run's energy unit (amu·Å²/fs² for the
            stock integrator — convert eV via ``1/EV_PER_AMU_A2_FS2``).
        sigma: Zero-crossing distance σ (Å).
        neighbors: Rebuildable neighbour list with **full bidirectional**
            edges (each pair present in both directions), e.g.
            :class:`~molix.md.neighbors.NeighborList`.
        cutoff: Truncation radius r_cut (Å). Defaults to the list's own
            cutoff, and must not exceed it — pairs between the two radii would
            simply be absent from the buffers, silently truncating the PES
            harder than asked.
        shift: Shift pair energies by ``E_lj(cutoff)`` (see above).

    Reference:
        Lennard-Jones, "On the Determination of Molecular Fields", Proc. R.
        Soc. Lond. A 106 (1924) 463. https://doi.org/10.1098/rspa.1924.0082
        Truncated-and-shifted convention: Allen & Tildesley, "Computer
        Simulation of Liquids", 2nd ed. (2017), §5.2.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        sigma: float,
        neighbors: NeighborStrategy,
        cutoff: float | None = None,
        shift: bool = True,
    ) -> None:
        super().__init__()
        # The interaction cutoff, not the list's r_build: a skinned list holds
        # pairs further out, but only guarantees them *complete* to .cutoff
        # between rebuilds, so anything beyond is the silent truncation this
        # check exists to prevent.
        list_cutoff = float(neighbors.cutoff)
        if cutoff is None:
            cutoff = list_cutoff
        elif float(cutoff) > list_cutoff:
            raise ValueError(
                f"cutoff {cutoff} A exceeds the neighbour list's horizon {list_cutoff} A; "
                "pairs between the two radii would be silently missing from the PES"
            )
        self.neighbors = neighbors
        self.shift = bool(shift)
        eps, sig, r_cut = float(epsilon), float(sigma), float(cutoff)
        sr6_cut = (sig / r_cut) ** 6
        self.register_buffer("epsilon", torch.as_tensor(eps))
        self.register_buffer("sigma", torch.as_tensor(sig))
        self.register_buffer("cutoff_sq", torch.as_tensor(r_cut * r_cut))
        self.register_buffer(
            "energy_shift",
            torch.as_tensor(4.0 * eps * (sr6_cut * sr6_cut - sr6_cut) if shift else 0.0),
        )

    def _apply(self, fn, recurse: bool = True):
        """Extend ``nn.Module._apply`` to the neighbour list's buffers.

        ``forward`` reads ``neighbors.edge_index`` / ``shifts`` live; a
        ``.to(device/dtype)`` that skipped the list would leave the pair
        geometry behind — a cross-device indexing error at best.
        """
        module = super()._apply(fn, recurse)
        neighbors = getattr(self, "neighbors", None)
        if neighbors is not None:
            ref = self.epsilon
            neighbors.to(ref.device, ref.dtype)
        return module

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Refresh the cutoff graph in place (shapes unchanged)."""
        self.neighbors.rebuild(pos)

    def forward(self, pos: torch.Tensor) -> ForceOutput:
        """Truncated-LJ energy + forces at ``pos`` ``(N, 3)`` from the live buffers."""
        edge_index = self.neighbors.edge_index  # (capacity, 2)
        source, target = edge_index[:, 0], edge_index[:, 1]
        # Minimum-image displacement: live positions + the list's periodic remainder.
        diff = pos[target] - pos[source] + self.neighbors.shifts  # (capacity, 3)
        r2 = (diff * diff).sum(-1)
        inside = r2 < self.cutoff_sq  # dead edges: |shift| ≫ cutoff → excluded here
        safe_r2 = torch.where(inside, r2, torch.ones_like(r2))
        inv_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6
        pair_e = torch.where(
            inside, 4.0 * self.epsilon * (inv_r12 - inv_r6) - self.energy_shift, 0.0
        )
        energy = 0.5 * pair_e.sum()  # bidirectional edges visit each pair twice
        # Per-edge force on the *source*: 24ε/r²·(2(σ/r)¹² − (σ/r)⁶)·(-diff).
        # The reverse edge delivers the Newton pair to the other atom, so the
        # forces need no ½ — only the energy double-counts.
        coef = torch.where(inside, 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2, 0.0)
        force = torch.zeros_like(pos)
        force.index_add_(0, source, coef.unsqueeze(-1) * (-diff))
        return ForceOutput(energy, force)
