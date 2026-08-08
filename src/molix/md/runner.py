"""Hook-driven MD runner: drive an :class:`Integrator` through an MD lifecycle.

:class:`MDRunner` owns the observation loop the way
:class:`molix.core.trainer.Trainer` owns the training loop, but it speaks its
own, deliberately narrow protocol: :class:`MDHook`. MD hooks receive the step
count and typed physics (:class:`~molix.md.types.MDState` /
:class:`~molix.md.types.MDObservables`) — they are **not** Trainer hooks, and
Trainer hooks (which dereference ``trainer.model`` / ``trainer.optimizer``)
are not accepted. One runner, one honest contract.

Hook dispatch is **static**: hooks subclass :class:`MDHook` (all lifecycle
methods have no-op defaults), so the runner calls the typed methods directly —
no ``getattr`` name lookup. A hook that acts on a step cadence declares it via
:attr:`MDHook.cadence` so :meth:`MDRunner.run` can refuse a ``chunk`` that
would silently skip firings.

:class:`TrajectoryHook` captures strided frames to host buffers, spilling to
on-disk shards every ``flush_every`` frames so host memory stays bounded, and
writes one ``.pt`` (+ optional extended-XYZ via
:func:`molix.datasets._extxyz.write_extxyz_frames`) at :meth:`MDHook.on_run_end`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from molix.md.forcefield import ForceField
from molix.md.integrators import Integrator, _as_mass_col
from molix.md.types import MDObservables, MDState
from molix.units import KB_AMU_A_FS


class MDHook:
    """Lifecycle observer for an MD run — the MD-specific hook contract.

    Subclass and override only what you need; every method is a no-op by
    default. Hooks that act on a step cadence (every N-th step) must declare
    it in :attr:`cadence` so the runner can validate ``chunk`` against it.
    """

    #: Steps between the firings this hook acts on (``None``: every
    #: observation). :meth:`MDRunner.run` rejects a ``chunk`` that is not a
    #: divisor of a declared cadence — chunking must never silently skip a
    #: hook's step.
    cadence: int | None = None

    def on_run_start(self, runner: "MDRunner") -> None:
        """Called once before the first step (the entry force is already cached)."""

    def on_step_start(self, runner: "MDRunner", step: int, state: MDState) -> None:
        """Called before each hook-visible advance.

        Args:
            runner: The driving runner.
            step: Steps completed so far (``0`` on the first call).
            state: The live state the upcoming advance will consume — the
                place to refresh position-derived caches (neighbour lists).
        """

    def on_step_end(self, runner: "MDRunner", step: int, obs: MDObservables) -> None:
        """Called after each hook-visible advance with the step's physics.

        Args:
            runner: The driving runner.
            step: Steps completed including this advance.
            obs: Typed thermodynamic snapshot at ``step``.
        """

    def on_run_end(self, runner: "MDRunner") -> None:
        """Called once after the last step (persist buffered results here)."""


def _normalize_hooks(
    hooks: Sequence[MDHook | tuple[MDHook, int]] | None,
) -> list[MDHook]:
    """Priority-sort hooks (lower priority first; ties keep registration order)."""
    if not hooks:
        return []
    normalized = []
    for idx, item in enumerate(hooks):
        if isinstance(item, tuple):
            hook, priority = item
            normalized.append((hook, priority, idx))
        else:
            normalized.append((item, 100, idx))
    normalized.sort(key=lambda x: (x[1], x[2]))
    return [hook for hook, _, _ in normalized]


class MDRunner:
    """Drive an :class:`~molix.md.integrators.Integrator` through the MD hook lifecycle.

    Args:
        integrator: The integrator advancing the typed ``MDState``.
        mass: Per-atom mass ``(N,)`` or scalar (integrator's mass unit). Used to
            report kinetic energy / temperature.
        hooks: :class:`MDHook` instances or ``(hook, priority)`` tuples; lower
            priority fires earlier, ties keep registration order.
        kb: Boltzmann constant in the integrator's energy unit (default: the
            (amu, Å, fs) value, so temperature comes out in kelvin).
        dof: Degrees of freedom for the temperature estimator; defaults to
            ``3 N - integrator.removed_dof`` (``3 N`` under Langevin — the O
            step thermostats the COM too — and ``3 N - 3`` under NVE with
            centre-of-mass momentum removed).
    """

    def __init__(
        self,
        integrator: Integrator,
        *,
        mass: float | torch.Tensor,
        hooks: Sequence[MDHook | tuple[MDHook, int]] | None = None,
        kb: float = KB_AMU_A_FS,
        dof: int | None = None,
    ) -> None:
        self.integrator = integrator
        self.hooks: list[MDHook] = _normalize_hooks(hooks)
        self._mass = mass
        self._kb = float(kb)
        self._dof = dof

    def run(self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int, *, chunk: int = 1) -> MDState:
        """Integrate ``n_steps`` steps, firing the hook lifecycle per chunk.

        ``chunk > 1`` advances the integrator ``chunk`` steps between hook
        firings (``Integrator.advance_n`` — no per-step Python, no per-step
        thermodynamics). The dynamics are bit-identical to ``chunk=1``; only
        the observation cadence changes, so every declared hook cadence
        (``TrajectoryHook.stride``, ``NeighborListHook.every``,
        ``MDCheckpointHook.every``) must be a multiple of ``chunk`` — enforced
        via :attr:`MDHook.cadence`.

        Args:
            pos: Initial positions ``(N, 3)``.
            vel: Initial velocities ``(N, 3)``.
            n_steps: Number of MD steps.
            chunk: Steps advanced between hook firings.

        Returns:
            The final typed :class:`~molix.md.types.MDState`. Trajectory
            capture is the job of hooks (see :class:`TrajectoryHook`).
        """
        mass = _as_mass_col(self._mass, pos)
        if self._dof is not None:
            dof = self._dof
        else:
            dof = max(1, 3 * int(pos.shape[0]) - self.integrator.removed_dof)

        chunk = max(1, int(chunk))
        for hook in self.hooks:
            if hook.cadence is not None and hook.cadence % chunk:
                raise ValueError(
                    f"{type(hook).__name__} fires every {hook.cadence} steps, which chunk="
                    f"{chunk} would silently skip; make it a multiple of chunk"
                )
        md = self.integrator.initial(pos, vel)
        for hook in self.hooks:
            hook.on_run_start(self)
        done = 0
        while done < n_steps:
            n = min(chunk, n_steps - done)
            # Fired *before* the advance so a hook can refresh position-derived
            # state (the neighbour list) while it still precedes the force
            # evaluation inside.
            for hook in self.hooks:
                hook.on_step_start(self, done, md)
            md = self.integrator.advance_n(md, n)
            done += n
            kinetic = 0.5 * (mass * md.vel * md.vel).sum()
            potential = md.energy.reshape(())
            temperature = 2.0 * kinetic / (dof * self._kb)
            obs = MDObservables(
                pos=md.pos,
                vel=md.vel,
                forces=md.forces,
                potential=potential,
                kinetic=kinetic,
                total=potential + kinetic,
                temperature=temperature,
            )
            for hook in self.hooks:
                hook.on_step_end(self, done, obs)
        for hook in self.hooks:
            hook.on_run_end(self)
        return md


class TrajectoryHook(MDHook):
    """Capture an MD trajectory to host buffers; persist at run end.

    Strided frames are copied to CPU and accumulated on the host; every
    ``flush_every`` kept frames the buffer is spilled to an on-disk shard and
    cleared, so host memory stays O(``flush_every`` · N) regardless of run
    length. At :meth:`on_run_end` the shards (if any) are concatenated into one
    ``.pt`` (+ optional extended-XYZ) and removed. Runs whose kept frames fit one
    buffer skip sharding entirely (identical output to a single write).

    Copies are plain (synchronous) ``.to("cpu")``: ``non_blocking=True`` into
    pageable host memory is silently synchronous anyway, so capture incurs a
    per-frame D2H copy (mitigate with ``stride``).

    Args:
        out: Output ``.pt`` path. A sibling ``.xyz`` is written when ``write_xyz``
            and ``numbers`` are given.
        stride: Keep every ``stride``-th frame.
        numbers: Atomic numbers ``(N,)`` for the optional extended-XYZ dump.
        write_xyz: Whether to also emit an extended-XYZ trajectory.
        with_forces: Whether to also capture per-frame forces ``(T, N, 3)``.
        flush_every: Kept-frame budget before spilling a shard to disk. Default 10000.
    """

    _FIELDS = ("pos", "vel", "f", "pe", "ke", "etot", "temp")

    def __init__(
        self,
        out: str | Path,
        *,
        stride: int = 1,
        numbers: torch.Tensor | None = None,
        write_xyz: bool = True,
        with_forces: bool = True,
        flush_every: int = 10000,
    ) -> None:
        self._out = Path(out)
        self._stride = max(1, int(stride))
        self._numbers = numbers
        self._write_xyz = write_xyz
        self._with_forces = with_forces
        self._flush_every = max(1, int(flush_every))
        self.cadence = self._stride
        self._buf: dict[str, list[torch.Tensor]] = {k: [] for k in self._FIELDS}
        self._n_buffered = 0
        self._shards: list[Path] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        if step % self._stride:
            return
        b = self._buf
        b["pos"].append(obs.pos.detach().to("cpu"))
        b["vel"].append(obs.vel.detach().to("cpu"))
        if self._with_forces:
            b["f"].append(obs.forces.detach().to("cpu"))
        b["pe"].append(obs.potential.detach().to("cpu"))
        b["ke"].append(obs.kinetic.detach().to("cpu"))
        b["etot"].append(obs.total.detach().to("cpu"))
        b["temp"].append(obs.temperature.detach().to("cpu"))
        self._n_buffered += 1
        if self._n_buffered >= self._flush_every:
            self._flush_shard()

    def _stack_buffer(self) -> dict[str, torch.Tensor] | None:
        """Stack and clear the in-memory buffer; ``None`` if empty."""
        if not self._buf["pos"]:
            return None
        stacked = {k: torch.stack(v) for k, v in self._buf.items() if v}
        for v in self._buf.values():
            v.clear()
        self._n_buffered = 0
        return stacked

    def _flush_shard(self) -> None:
        chunk = self._stack_buffer()
        if chunk is None:
            return
        self._out.parent.mkdir(parents=True, exist_ok=True)
        path = self._out.with_suffix(f".part{len(self._shards)}.pt")
        torch.save(chunk, path)
        self._shards.append(path)

    def _combine_shards(self) -> dict[str, torch.Tensor] | None:
        if not self._shards:
            return None
        loaded = [torch.load(p, weights_only=True) for p in self._shards]
        fields = {k: torch.cat([d[k] for d in loaded]) for k in loaded[0]}
        for p in self._shards:
            p.unlink()
        self._shards.clear()
        return fields

    def on_run_end(self, runner: MDRunner) -> None:
        if self._shards:
            self._flush_shard()  # spill the trailing partial buffer
            fields = self._combine_shards()
        else:
            fields = self._stack_buffer()
        if fields is None:
            return
        payload: dict[str, Any] = {
            "pos": fields["pos"].to(torch.float32),
            "vel": fields["vel"].to(torch.float32),
            "pe": fields["pe"],
            "ke": fields["ke"],
            "etot": fields["etot"],
            "temp": fields["temp"],
            "stride": self._stride,
        }
        if "f" in fields:
            payload["forces"] = fields["f"].to(torch.float32)
        if self._numbers is not None:
            payload["Z"] = self._numbers.detach().cpu()
        self._out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self._out)
        if self._write_xyz and self._numbers is not None:
            self._dump_xyz(payload["pos"], payload["etot"], payload["temp"])

    def _dump_xyz(self, pos: torch.Tensor, etot: torch.Tensor, temp: torch.Tensor) -> None:
        from molpy import Element

        from molix.datasets._extxyz import write_extxyz_frames

        species = [Element(int(z)).symbol for z in self._numbers.detach().cpu()]  # type: ignore[union-attr]
        write_extxyz_frames(
            self._out.with_suffix(".xyz"),
            species=species,
            positions=pos.to(torch.float64).numpy(),
            energies=etot.to(torch.float64).numpy(),
            tags=[f"temperature={float(t):.2f}" for t in temp],
        )


class NeighborListHook(MDHook):
    """Legacy step-start neighbour rebuild — prefer ``MD(rebuild_every=)``.

    .. warning::

        Rebuilding on :meth:`MDHook.on_step_start` refreshes the list at the
        *start-of-step* positions, while velocity-Verlet evaluates forces at
        the *end-of-step* positions. That one-step lag makes ``F`` not equal
        to ``-∇E`` of the energy surface the list defines, and produces a
        systematic NVE energy drift (measured ~30× worse at ``every=5`` than
        at ``every=1`` on MACE-MatPES water). ``MD(rebuild_every=)`` now
        rebuilds inside :meth:`~molix.md.integrators.Integrator.eval_force`
        at the force-evaluation positions instead; this hook is kept for
        callers that explicitly want step-start semantics (e.g. tests).

    Args:
        force: The force field to refresh.
        every: Step interval. ``1`` rebuilds before every step.
    """

    def __init__(self, force: ForceField, *, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self._force = force
        self._every = int(every)
        self.cadence = self._every

    def on_step_start(self, runner: MDRunner, step: int, state: MDState) -> None:
        """Rebuild on cadence, using the positions the upcoming step will use."""
        if step % self._every == 0:
            self._force.rebuild_neighbors(state.pos)


class MDCheckpointHook(MDHook):
    """Persist a restartable NVE state (pos, vel, absolute step) every N steps.

    Named ``MDCheckpointHook`` — :class:`molix.hooks.CheckpointHook` is the
    training-side checkpointer with an unrelated constructor; the two must not
    collide in a ``from molix... import *`` namespace.

    Multi-hour trajectories die to walltime and node failures; without this,
    everything after the last :class:`TrajectoryHook` shard is gone. The write
    is atomic (temp file + ``rename``) so a kill mid-write leaves the previous
    checkpoint intact. Restart is exact for γ=0 — an NVE state is fully
    determined by ``(pos, vel)`` — and approximate for γ>0 (the Langevin noise
    stream restarts, which changes the realisation but not the ensemble).

    Doubles as the run's heartbeat: each checkpoint prints one line, so a
    day-long job's log shows progress instead of silence.

    Args:
        path: Checkpoint file, overwritten in place.
        every: Step interval between checkpoints.
        step_offset: Absolute step count this run resumed from, added to the
            in-run step so a chain of resumed segments keeps one monotonic
            step axis.
    """

    def __init__(self, path: str | Path, *, every: int, step_offset: int = 0) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self._path = Path(path)
        self._every = int(every)
        self._offset = int(step_offset)
        self.cadence = self._every

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        if step % self._every == 0:
            self._save(step, obs)

    def _save(self, step: int, obs: MDObservables) -> None:
        absolute = self._offset + int(step)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        torch.save(
            {
                "pos": obs.pos.detach().cpu(),
                "vel": obs.vel.detach().cpu(),
                "step": absolute,
            },
            tmp,
        )
        tmp.replace(self._path)
        print(
            f"[checkpoint] step {absolute}  E_tot={float(obs.total):.6f}  "
            f"T={float(obs.temperature):.1f} K",
            flush=True,
        )
