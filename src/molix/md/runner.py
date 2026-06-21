"""Hook-driven MD runner: drive an integrator through the molix hook lifecycle.

:class:`MDRunner` plays the role :class:`molix.core.trainer.Trainer` plays for
training — it owns a :class:`~molix.core.state.TrainState`, a priority-sorted
hook list, and a fail-loud ``_call_hooks`` dispatcher — but its loop is a
molecular-dynamics integration rather than a gradient step. This lets the
existing hook ecosystem (``StepSpeedHook``, ``GPUMemoryHook``, ``JournalHook``,
``CheckpointHook``, ...) observe an MD run unchanged: the runner fires
``on_train_start`` once, ``on_train_batch_end`` per step, and ``on_train_end`` at
the close, advancing ``state["global_step"]`` each step.

Per-step physics (positions, velocities, potential/kinetic/total energy,
instantaneous temperature) is handed to hooks via the ``outputs`` argument of
``on_train_batch_end`` — the same channel the Trainer uses for step outputs — so
the runner never writes physics scalars into the reserved ``TrainState``
namespaces (``train`` / ``eval`` / ``performance`` / ``gpu``), keeping the state
namespace-ownership contract intact. A hook that wants to persist a scalar into a
namespace is free to do so from ``on_train_batch_end``.

:class:`TrajectoryHook` is the reference consumer: it captures strided frames
into pinned-CPU buffers with non-blocking copies (no per-step device sync) and
writes a ``.pt`` (+ optional extended-XYZ) trajectory at ``on_train_end``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from molix.core.hook import BaseHook, Hook
from molix.core.state import Stage, TrainState
from molix.md.integrators import LangevinVerletIntegrator

# Boltzmann constant in the (amu, Å, fs) MD unit system used by the PiNet MD
# (energy unit 1 amu·Å²/fs² = 103.642 689 eV). k_B = 8.617333e-5 eV/K / that.
KB_AMU_A_FS = 8.617333262e-5 / 103.6426965638


def _normalize_hooks(hooks: Sequence[Hook | tuple[Hook, int]] | None) -> list[Hook]:
    """Priority-sort hooks (lower priority first; ties keep registration order).

    Mirrors :class:`molix.core.trainer.Trainer`'s normalization so a hook list
    behaves identically whether driven by training or by MD.
    """
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
    """Drive a :class:`LangevinVerletIntegrator` through the hook lifecycle.

    Args:
        integrator: The integrator providing the force-cached step primitive
            (:meth:`~LangevinVerletIntegrator.step_cached`). Build it with
            ``compile=True`` to integrate under :func:`torch.compile`.
        mass: Per-atom mass ``(N,)`` or a scalar, in the integrator's mass unit
            (amu by default). Used to report kinetic energy / temperature.
        hooks: Hooks or ``(hook, priority)`` tuples; same protocol as the Trainer.
        kb: Boltzmann constant in the integrator's energy unit (default: the
            (amu, Å, fs) value, so temperature comes out in kelvin).
        dof: Degrees of freedom for the temperature estimator; defaults to
            ``3 N - 3`` (centre-of-mass momentum removed).
    """

    def __init__(
        self,
        integrator: LangevinVerletIntegrator,
        *,
        mass: float | torch.Tensor,
        hooks: Sequence[Hook | tuple[Hook, int]] | None = None,
        kb: float = KB_AMU_A_FS,
        dof: int | None = None,
    ) -> None:
        self.integrator = integrator
        self.hooks: list[Hook] = _normalize_hooks(hooks)
        self._mass = mass
        self._kb = float(kb)
        self._dof = dof
        self.state = TrainState()

    def _call_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Call ``hook_name`` on every hook; fail loud (mirror Trainer semantics)."""
        for hook in self.hooks:
            method = getattr(hook, hook_name, None)
            if method is not None and callable(method):
                method(*args, **kwargs)

    def _mass_col(self, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(self._mass, torch.Tensor):
            m = self._mass.to(dtype=ref.dtype, device=ref.device)
            return m.reshape(-1, 1) if m.dim() == 1 else m
        return torch.as_tensor(self._mass, dtype=ref.dtype, device=ref.device)

    def run(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        n_steps: int,
    ) -> dict[str, Any]:
        """Integrate ``n_steps`` steps, firing the hook lifecycle each step.

        Args:
            pos: Initial positions ``(N, 3)``.
            vel: Initial velocities ``(N, 3)``.
            n_steps: Number of MD steps.

        Returns:
            Dict with the final ``pos`` / ``vel`` / ``force`` tensors and the
            terminal :class:`~molix.core.state.TrainState`. Trajectory capture is
            the job of hooks (see :class:`TrajectoryHook`).
        """
        state = self.state
        state["stage"] = Stage.TRAIN
        state["global_step"] = 0
        mass = self._mass_col(pos)
        dof = self._dof if self._dof is not None else max(1, 3 * int(pos.shape[0]) - 3)

        self._call_hooks("on_train_start", self, state)
        force = self.integrator.initial_force(pos)
        for i in range(n_steps):
            pos, vel, energy, force = self.integrator.step_cached(pos, vel, force)
            kinetic = 0.5 * (mass * vel * vel).sum()
            potential = energy.reshape(())
            temperature = 2.0 * kinetic / (dof * self._kb)
            state["global_step"] = i + 1
            # Physics rides the ``outputs`` channel (like Trainer step outputs),
            # NOT the reserved state namespaces — keeps the state contract clean.
            outputs = {
                "pos": pos,
                "vel": vel,
                "forces": force,
                "potential": potential,
                "kinetic": kinetic,
                "total": potential + kinetic,
                "temperature": temperature,
            }
            self._call_hooks(
                "on_train_batch_end", self, state, {"pos": pos, "vel": vel}, outputs
            )
        self._call_hooks("on_train_end", self, state)
        return {"pos": pos, "vel": vel, "force": force, "state": state}


class TrajectoryHook(BaseHook):
    """Capture an MD trajectory into pinned-CPU buffers; persist at run end.

    Frames are copied off the device with ``non_blocking=True`` (no per-step
    ``.item()`` / sync), accumulated on the host, and written once at
    ``on_train_end`` — so trajectory I/O never stalls the integration hot loop.

    Args:
        out: Output ``.pt`` path. A sibling ``.xyz`` is written too when
            ``write_xyz`` and ``numbers`` are given.
        stride: Keep every ``stride``-th frame.
        numbers: Atomic numbers ``(N,)`` for the optional extended-XYZ dump.
        write_xyz: Whether to also emit an extended-XYZ trajectory.
        with_forces: Whether to also capture the per-frame forces ``(T, N, 3)``
            (the integrator's ``outputs["forces"]``) for downstream per-atom
            force diagnostics. Cheap; on by default.
    """

    _SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}

    def __init__(
        self,
        out: str | Path,
        *,
        stride: int = 1,
        numbers: torch.Tensor | None = None,
        write_xyz: bool = True,
        with_forces: bool = True,
    ) -> None:
        self._out = Path(out)
        self._stride = max(1, int(stride))
        self._numbers = numbers
        self._write_xyz = write_xyz
        self._with_forces = with_forces
        self._pos: list[torch.Tensor] = []
        self._vel: list[torch.Tensor] = []
        self._f: list[torch.Tensor] = []
        self._pe: list[torch.Tensor] = []
        self._ke: list[torch.Tensor] = []
        self._etot: list[torch.Tensor] = []
        self._temp: list[torch.Tensor] = []

    def on_train_batch_end(
        self, trainer: Any, state: TrainState, batch: Any, outputs: Any
    ) -> None:
        if state["global_step"] % self._stride:
            return
        self._pos.append(outputs["pos"].detach().to("cpu", non_blocking=True))
        self._vel.append(outputs["vel"].detach().to("cpu", non_blocking=True))
        if self._with_forces and outputs.get("forces") is not None:
            self._f.append(outputs["forces"].detach().to("cpu", non_blocking=True))
        self._pe.append(outputs["potential"].detach().to("cpu", non_blocking=True))
        self._ke.append(outputs["kinetic"].detach().to("cpu", non_blocking=True))
        self._etot.append(outputs["total"].detach().to("cpu", non_blocking=True))
        self._temp.append(outputs["temperature"].detach().to("cpu", non_blocking=True))

    def on_train_end(self, trainer: Any, state: TrainState) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # single sync: flush the non-blocking copies
        if not self._pos:
            return
        pos = torch.stack(self._pos)
        vel = torch.stack(self._vel)
        pe = torch.stack(self._pe)
        ke = torch.stack(self._ke)
        etot = torch.stack(self._etot)
        temp = torch.stack(self._temp)
        payload: dict[str, Any] = {
            "pos": pos.to(torch.float32),
            "vel": vel.to(torch.float32),
            "pe": pe,
            "ke": ke,
            "etot": etot,
            "temp": temp,
            "stride": self._stride,
        }
        if self._f:
            payload["forces"] = torch.stack(self._f).to(torch.float32)
        if self._numbers is not None:
            payload["Z"] = self._numbers.detach().cpu()
        self._out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self._out)
        if self._write_xyz and self._numbers is not None:
            self._dump_xyz(pos, etot, temp)

    def _dump_xyz(self, pos: torch.Tensor, etot: torch.Tensor, temp: torch.Tensor) -> None:
        zs = [int(z) for z in self._numbers.detach().cpu()]  # type: ignore[union-attr]
        syms = [self._SYMBOLS.get(z, "X") for z in zs]
        n = len(zs)
        with self._out.with_suffix(".xyz").open("w") as fh:
            for t in range(pos.shape[0]):
                coords = pos[t].to(torch.float64).numpy()
                fh.write(f"{n}\n")
                fh.write(f"Etot={float(etot[t]):.6f} T={float(temp[t]):.2f} frame={t}\n")
                for sy, (x, y, z) in zip(syms, coords):
                    fh.write(f"{sy} {x:.6f} {y:.6f} {z:.6f}\n")
