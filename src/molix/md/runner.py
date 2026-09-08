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
:func:`molix.io.extxyz.write_extxyz_frames`) at :meth:`MDHook.on_run_end`.
:class:`EnergyDriftHook` accumulates ``(t, E_tot)`` — sample time and total
energy — and reports the energy-conservation gate for an NVE run (constant
atom number ``N``, volume ``V`` and total energy ``E``: no thermostat and no
barostat, so a correct integration must not gain or lose energy) as an
:class:`EnergyDrift` record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from tensordict import TensorDict

from molix.md.integrators import Integrator, _as_mass_col
from molix.md.types import MDObservables, MDState
from molix.schema import MASS_KEY, POS_KEY, VEL_KEY

if TYPE_CHECKING:
    from molix.md.barostat import BerendsenBarostat
from molix.units import units

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


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

    def on_init(self, runner: "MDRunner", state: MDState) -> None:
        """Called once after the entry force is cached, before any step.

        Mutate ``state`` here (LAMMPS-style ``velocity create``). Default no-op.
        """

    def on_run_start(self, runner: "MDRunner") -> None:
        """Called once before the first step (the entry force is already cached)."""

    def on_step_start(self, runner: "MDRunner", step: int, state: MDState) -> None:
        """Called before each hook-visible advance.

        Not the place to refresh a neighbour list: velocity-Verlet evaluates
        ``F`` at the *end-of-step* positions, so a step-start rebuild leaves the
        connectivity one displacement behind. The list's policy runs inside
        :meth:`~molix.md.integrators.Integrator.eval_force` instead.

        Args:
            runner: The driving runner.
            step: Steps completed so far (``0`` on the first call).
            state: The live state the upcoming advance will consume.
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


class VelocityInitHook(MDHook):
    """Write initial velocities from a distribution (LAMMPS ``velocity create``).

    The distribution is a component with ``velocities(pos, mass) -> (N, 3)``.
    This hook only decides *when* (``on_init``)::

        VelocityInitHook(MaxwellBoltzmann(300, seed=0))
    """

    def __init__(self, distribution: object) -> None:
        self.distribution = distribution

    def on_init(self, runner: "MDRunner", state: MDState) -> None:
        vel = self.distribution.velocities(state.pos, runner.integrator.mass)
        state.vel.copy_(vel.to(dtype=state.vel.dtype, device=state.vel.device))


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
        hooks: Sequence[MDHook | tuple[MDHook, int]] | None = None,
        kb: float = _KB_MD,
        dof: int | None = None,
        barostat: BerendsenBarostat | None = None,
        sync_state: Callable[[MDState], None] | None = None,
    ) -> None:
        self.integrator = integrator
        self.hooks: list[MDHook] = _normalize_hooks(hooks)
        self._kb = float(kb)
        self._dof = dof
        self.barostat = barostat
        self.sync_state = sync_state

    def run(self, system: TensorDict, n_steps: int, *, chunk: int = 1) -> MDState:
        """Integrate ``n_steps`` on ``system``, firing hooks per chunk.

        Reads ``atoms.pos`` / ``atoms.vel`` / ``atoms.mass``, binds mass onto
        the integrator, writes positions and velocities back at the end.

        Args:
            system: TensorDict with ``atoms.pos``, ``atoms.vel``, ``atoms.mass``.
            n_steps: Number of MD steps.
            chunk: Steps advanced between hook firings.

        Returns:
            The final typed :class:`~molix.md.types.MDState`.
        """
        pos = system[POS_KEY]
        vel = system[VEL_KEY]
        self.integrator.bind_mass(system[MASS_KEY])
        mass = _as_mass_col(system[MASS_KEY], pos)
        cell = None
        if ("graphs", "cell") in system.keys(include_nested=True):
            cell = system["graphs", "cell"]
            if cell.dim() == 3:
                cell = cell[0]
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
        md = self.integrator.initial(pos, vel, cell=cell)
        for hook in self.hooks:
            hook.on_init(self, md)
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
            if self.barostat is not None:
                md = self.barostat.scale(md, mass=system[MASS_KEY], dt=float(self.integrator.dt))
                if self.sync_state is not None:
                    self.sync_state(md)
            done += n
            kinetic = 0.5 * (mass * md.vel * md.vel).sum()
            potential = md.energy.reshape(())
            temperature = 2.0 * kinetic / (dof * self._kb)
            volume = None
            pressure = None
            if md.cell is not None:
                volume = torch.linalg.det(md.cell).abs()
                if md.virial is not None:
                    pressure = (2.0 * kinetic - md.virial.trace()) / (3.0 * volume)
            obs = MDObservables(
                pos=md.pos,
                vel=md.vel,
                forces=md.forces,
                potential=potential,
                kinetic=kinetic,
                total=potential + kinetic,
                temperature=temperature,
                cell=md.cell,
                virial=md.virial,
                volume=volume,
                pressure=pressure,
            )
            for hook in self.hooks:
                hook.on_step_end(self, done, obs)
        for hook in self.hooks:
            hook.on_run_end(self)
        system[POS_KEY] = md.pos
        system[VEL_KEY] = md.vel
        if md.cell is not None and "graphs" in system.keys() and "cell" in system["graphs"].keys():
            dst = system["graphs", "cell"]
            (dst[0] if dst.dim() == 3 else dst).copy_(md.cell)
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
        dtype: Optional fp32/fp64 storage dtype for positions, velocities and
            forces. ``None`` preserves their input dtype. Energy/temperature
            precision is always preserved. The final shard merge materializes
            the complete trajectory; ``flush_every`` bounds only capture buffers.
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
        dtype: torch.dtype | None = None,
    ) -> None:
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("trajectory dtype must be torch.float32, torch.float64 or None")
        self._out = Path(out)
        self._stride = max(1, int(stride))
        self._numbers = numbers
        self._write_xyz = write_xyz
        self._with_forces = with_forces
        self._flush_every = max(1, int(flush_every))
        self._dtype = dtype
        self.cadence = self._stride
        self._buf: dict[str, list[torch.Tensor]] = {k: [] for k in self._FIELDS}
        self._n_buffered = 0
        self._shards: list[Path] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        if step % self._stride:
            return
        b = self._buf
        # CPU .to('cpu') can alias the live MD buffer. Every frame must be a
        # snapshot, on CPU as well as CUDA. Choose storage precision once.
        b["pos"].append(obs.pos.detach().to("cpu", dtype=self._dtype, copy=True))
        b["vel"].append(obs.vel.detach().to("cpu", dtype=self._dtype, copy=True))
        if self._with_forces:
            b["f"].append(obs.forces.detach().to("cpu", dtype=self._dtype, copy=True))
        b["pe"].append(obs.potential.detach().to("cpu", copy=True))
        b["ke"].append(obs.kinetic.detach().to("cpu", copy=True))
        b["etot"].append(obs.total.detach().to("cpu", copy=True))
        b["temp"].append(obs.temperature.detach().to("cpu", copy=True))
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
            raise RuntimeError(
                f"trajectory {self._out} has no frames (stride={self._stride}); nothing to write"
            )
        payload: dict[str, torch.Tensor | int] = {
            "pos": fields["pos"],
            "vel": fields["vel"],
            "pe": fields["pe"],
            "ke": fields["ke"],
            "etot": fields["etot"],
            "temp": fields["temp"],
            "stride": self._stride,
        }
        if "f" in fields:
            payload["forces"] = fields["f"]
        if self._numbers is not None:
            payload["Z"] = self._numbers.detach().cpu()
        self._out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self._out)
        if self._write_xyz and self._numbers is not None:
            self._dump_xyz(payload["pos"], payload["etot"], payload["temp"])

    def _dump_xyz(self, pos: torch.Tensor, etot: torch.Tensor, temp: torch.Tensor) -> None:
        from molpy import Element

        from molix.io.extxyz import write_extxyz_frames

        species = [Element(int(z)).symbol for z in self._numbers.detach().cpu()]
        write_extxyz_frames(
            self._out.with_suffix(".xyz"),
            species=species,
            positions=pos.to(torch.float64).numpy(),
            energies=etot.to(torch.float64).numpy(),
            tags=[f"temperature={float(t):.2f}" for t in temp],
        )


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


@dataclass(frozen=True)
class EnergyDrift:
    """Energy-conservation record of one MD run — the fit, not the samples.

    Produced by :meth:`EnergyDriftHook.drift`. Energies are in the MD energy
    unit (amu·Å²/fs², the (amu, Å, fs) unit set every :class:`MDState` is
    integrated in), times in fs; ``rel_drift`` and ``rms_rel`` are
    dimensionless because both divide by ``|e0|``.

    Every field describes the *sampled* series, not the run: the window opens
    at the first sample recorded, not at ``t = 0``. Driven through
    :class:`MDRunner` with a timestep ``dt`` (fs) the first observation lands
    at ``chunk · dt``, so ``n_steps`` at that ``chunk`` sample a window of
    ``(n_steps − chunk) · dt``. Nothing here depends on where the time axis
    starts: adding the same constant to every sample time moves no two samples
    relative to each other, so the fitted ``slope``, the window
    ``duration_fs`` and both ratios come out identical.

    Attributes:
        e0: Total energy of the **first** sample, ``E_tot(t_first)``
            (amu·Å²/fs²) — the reference every ratio below divides by.
        duration_fs: Length of the sampled window, ``t_last − t_first`` (fs).
        slope: Least-squares ``dE_tot/dt`` over the window, i.e. amu·Å²/fs²
            per fs (amu·Å²/fs³).
        rel_drift: ``|slope · duration_fs| / |e0|`` — the fitted secular
            energy change over the whole window, relative to ``|e0|``.
        rms_rel: ``rms(E_tot − mean E_tot) / |e0|`` — the bounded fluctuation
            about the mean, relative to ``|e0|``.
    """

    e0: float
    duration_fs: float
    slope: float
    rel_drift: float
    rms_rel: float


class EnergyDriftHook(MDHook):
    """Measure NVE energy conservation: secular drift vs bounded fluctuation.

    A symplectic integrator (velocity-Verlet) conserves a *shadow* Hamiltonian
    rather than ``E_tot``, so the total energy of a correct NVE run oscillates
    with a bounded amplitude — that amplitude is :attr:`EnergyDrift.rms_rel`,
    it is set by ``dt``, and it is not an error. What *is* an error is a
    **secular** term underneath: energy leaking in or out step after step,
    which is never the integrator but the thing being integrated — a
    discontinuous cutoff, a neighbour list rebuilt too late (missed pairs), an
    energy and a force that are not each other's derivative, or precision
    loss. Fitting a straight line through ``E_tot(t)`` separates the two:
    :attr:`EnergyDrift.slope` is the secular part, ``rms_rel`` the bounded
    part. Extrapolated over the sampled window and made dimensionless, the
    slope is the gate: pass when ``rel_drift = |slope · duration_fs| / |e0| <
    1e-3``, where ``e0`` is the **first sampled** total energy and
    ``duration_fs = t_last − t_first`` the span it was fitted over (both as
    defined on :class:`EnergyDrift`, which also explains why the choice of
    time origin cannot change either number).

    The hook fires on every observation (no :attr:`MDHook.cadence`), so the
    sample spacing is the runner's ``chunk`` and the first sample sits at
    ``chunk · dt``, not at ``t = 0``. That costs **one** host read per
    observation (the step's ``E_tot``); ``dt`` is constant, so it is read once
    at :meth:`on_run_start` and reused. Even so this is an observer fired per
    chunk and never a per-step term on the force path.

    Drive it through a runner, or feed it a series directly with
    :meth:`record`::

        hook = EnergyDriftHook()
        MDRunner(integrator, hooks=[hook]).run(system, 200, chunk=10)
        assert hook.result.rel_drift < 1e-3

    Attributes:
        times_fs: Sample times in fs (``step · dt``) of the run in progress —
            :meth:`on_run_start` drops the previous run's.
        totals: Sampled total energies in amu·Å²/fs², drained alongside
            ``times_fs``.
        result: The fit published at :meth:`on_run_end`; ``None`` before the
            run ends, and still ``None`` after a run that recorded fewer than
            two observations.

    Reference:
        The physics — a symplectic (structure-preserving) integrator such as
        velocity-Verlet conserving a nearby *shadow* Hamiltonian, hence
        bounded ``E_tot`` oscillation with no secular term — is
        M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids*,
        2nd ed., Oxford University Press (2017), ch. 3,
        https://doi.org/10.1093/oso/9780198803195.001.0001.

        The numeric gate ``rel_drift < 1e-3`` is this repo's, lifted from
        ``benchmarks/verify_md_ljcut_nve.py`` (lj/cut argon at the LAMMPS
        ``melt`` state point), which now measures it through this hook.
    """

    def __init__(self) -> None:
        self.times_fs: list[float] = []
        self.totals: list[float] = []
        self.result: EnergyDrift | None = None
        self._dt_fs: float | None = None

    def on_run_start(self, runner: MDRunner) -> None:
        """Start a fresh series, and read the timestep once.

        **One run, one series.** A hook mounted for several runs — a warm-up
        and then the timed run, the usual benchmark shape — drops the previous
        run's samples and its published :attr:`result` here, the way
        :class:`TrajectoryHook` drains its buffers per run. Fitting one line
        across two runs would price a discontinuity in ``E_tot`` (a re-seeded
        velocity draw, a re-phased neighbour list) as drift, and the caller
        would have no way to see it. A caller that wants both runs in one fit
        feeds :meth:`record` itself and never mounts the hook.

        ``dt`` is a registered buffer and constant for the run, so it is read
        back from the device once here rather than per observation.
        """
        self.times_fs.clear()
        self.totals.clear()
        self.result = None
        # A registered buffer widens to ``Tensor | Module`` on attribute
        # access; the cast keeps the read honest for the type checker.
        self._dt_fs = float(cast(torch.Tensor, runner.integrator.dt))

    def record(self, time_fs: float, total: float) -> None:
        """Append one ``(t, E_tot)`` sample.

        Args:
            time_fs: Sample time in fs.
            total: Total energy at ``time_fs`` in amu·Å²/fs².
        """
        self.times_fs.append(float(time_fs))
        self.totals.append(float(total))

    def drift(self) -> EnergyDrift:
        """Fit the recorded series and report the conservation record.

        The slope is the ordinary least-squares estimate over the centred
        times, ``Σ (t−t̄)(E−Ē) / Σ (t−t̄)²``, where ``t`` and ``E`` run over
        the recorded ``(times_fs, totals)`` pairs and ``t̄`` / ``Ē`` are their
        means — the straight line through the samples that minimises the
        summed squared vertical distance to them. Both sums are accumulated in
        float64 whatever dtype the MD state runs at, so the fit does not
        inherit an fp32 trajectory's rounding. Centring the times is also what
        makes the fit independent of where the time axis starts.

        Returns:
            The :class:`EnergyDrift` record for the samples recorded so far —
            ``slope`` in amu·Å²/fs³, ``duration_fs`` in fs, ``e0`` in
            amu·Å²/fs², the two ratios dimensionless.

        Raises:
            ValueError: If fewer than two samples were recorded — a slope
                needs two points. Through a runner this means the run produced
                fewer than two observations (``n_steps < 2 · chunk``).
        """
        n = len(self.totals)
        if n < 2:
            raise ValueError(f"energy drift needs at least 2 samples, got {n}")
        times = torch.tensor(self.times_fs, dtype=torch.float64)
        energies = torch.tensor(self.totals, dtype=torch.float64)
        centred = times - times.mean()
        residual = energies - energies.mean()
        slope = float((centred * residual).sum() / (centred * centred).sum())
        e0 = self.totals[0]
        duration_fs = self.times_fs[-1] - self.times_fs[0]
        return EnergyDrift(
            e0=e0,
            duration_fs=duration_fs,
            slope=slope,
            rel_drift=abs(slope * duration_fs) / abs(e0),
            rms_rel=float(residual.pow(2).mean().sqrt()) / abs(e0),
        )

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        """Record ``(step · dt, E_tot)`` for this observation.

        ``step`` counts completed MD steps and the integrator's ``dt`` is in
        fs, so the product is the sample time in fs; ``obs.total`` is the
        potential plus kinetic energy in amu·Å²/fs². ``dt`` comes from
        :meth:`on_run_start`, and is read here only when the hook is driven
        method-by-method without that call.
        """
        dt_fs = self._dt_fs
        if dt_fs is None:
            dt_fs = float(cast(torch.Tensor, runner.integrator.dt))
        self.record(step * dt_fs, float(obs.total))

    def on_run_end(self, runner: MDRunner) -> None:
        """Publish the fit on :attr:`result`, or ``None`` if it cannot be fitted.

        :attr:`result` stays ``None`` when fewer than two observations were
        recorded: a warm-up segment, or a run shorter than one ``chunk``, is
        not an error of the hook, and ending such a run must not take the
        runner down. :meth:`drift` itself still raises — a caller that needs
        the fit asks for it and gets the reason it cannot have one.
        """
        self.result = self.drift() if len(self.totals) >= 2 else None
