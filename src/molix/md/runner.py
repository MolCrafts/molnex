"""Hook-driven MD runner: drive an :class:`Integrator` through the hook lifecycle.

:class:`MDRunner` plays the role :class:`molix.core.trainer.Trainer` plays for
training — it owns a :class:`~molix.core.state.TrainState` and a priority-sorted
hook list — but its loop is a molecular-dynamics integration. It fires
``on_train_start`` once, ``on_train_batch_end`` per step, ``on_train_end`` at the
close, advancing ``state["global_step"]`` each step, so the existing hook
ecosystem observes an MD run unchanged.

Hook dispatch is **static**: hooks are :class:`~molix.core.hook.BaseHook`
instances (all lifecycle methods have no-op defaults), so the runner calls the
typed methods directly — no ``getattr`` name lookup. The integrator advances a
typed :class:`~molix.md.types.MDState`; per-step physics is unpacked into the
``outputs`` dict (the same channel the Trainer uses) so the runner never writes
physics into the reserved ``TrainState`` namespaces.

:class:`TrajectoryHook` captures strided frames to host buffers, spilling to
on-disk shards every ``flush_every`` frames so host memory stays bounded, and
writes one ``.pt`` (+ optional extended-XYZ) at ``on_train_end``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from molix.core.hook import BaseHook
from molix.core.state import Stage, TrainState
from molix.md.integrators import EV_PER_AMU_A2_FS2, LangevinVerletIntegrator, as_mass_col

#: Boltzmann constant in eV/K. (``molix.quant`` keeps its own copy for the
#: quantization subsystem; this is the MD package's single named source.)
KB_EV_PER_K = 8.617333262e-5
#: k_B in the integrator's (amu, Å, fs) energy unit (amu·Å²/fs²), so temperature
#: comes out in kelvin: k_B[eV/K] / (1 amu·Å²/fs² in eV).
KB_AMU_A_FS = KB_EV_PER_K / EV_PER_AMU_A2_FS2


def _normalize_hooks(
    hooks: Sequence[BaseHook | tuple[BaseHook, int]] | None,
) -> list[BaseHook]:
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
    """Drive a :class:`LangevinVerletIntegrator` through the hook lifecycle.

    Args:
        integrator: The integrator advancing the typed ``MDState``.
        mass: Per-atom mass ``(N,)`` or scalar (integrator's mass unit). Used to
            report kinetic energy / temperature.
        hooks: :class:`~molix.core.hook.BaseHook` instances or ``(hook, priority)``
            tuples; same protocol as the Trainer.
        kb: Boltzmann constant in the integrator's energy unit (default: the
            (amu, Å, fs) value, so temperature comes out in kelvin).
        dof: Degrees of freedom for the temperature estimator; defaults to ``3 N``
            under Langevin (γ>0, the O step thermostats the COM too) and ``3 N - 3``
            under NVE (centre-of-mass momentum removed).
    """

    def __init__(
        self,
        integrator: LangevinVerletIntegrator,
        *,
        mass: float | torch.Tensor,
        hooks: Sequence[BaseHook | tuple[BaseHook, int]] | None = None,
        kb: float = KB_AMU_A_FS,
        dof: int | None = None,
    ) -> None:
        self.integrator = integrator
        self.hooks: list[BaseHook] = _normalize_hooks(hooks)
        self._mass = mass
        self._kb = float(kb)
        self._dof = dof
        self.state = TrainState()

    def run(self, pos: torch.Tensor, vel: torch.Tensor, n_steps: int) -> dict[str, Any]:
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
        mass = as_mass_col(self._mass, pos)
        if self._dof is not None:
            dof = self._dof
        else:
            # Langevin (γ>0) thermostats all 3N DoF including the COM; NVE with
            # COM momentum removed leaves 3N-3. 3N-3 under Langevin would
            # over-report T by 3N/(3N-3).
            n = int(pos.shape[0])
            dof = max(1, 3 * n - (0 if self.integrator.gamma > 0.0 else 3))

        for hook in self.hooks:
            hook.on_train_start(self, state)
        md = self.integrator.initial(pos, vel)
        for i in range(n_steps):
            md = self.integrator.advance(md)
            kinetic = 0.5 * (mass * md.vel * md.vel).sum()
            potential = md.energy.reshape(())
            temperature = 2.0 * kinetic / (dof * self._kb)
            state["global_step"] = i + 1
            # Physics rides the ``outputs`` channel (like Trainer step outputs),
            # NOT the reserved state namespaces — keeps the state contract clean.
            outputs = {
                "pos": md.pos,
                "vel": md.vel,
                "forces": md.force,
                "potential": potential,
                "kinetic": kinetic,
                "total": potential + kinetic,
                "temperature": temperature,
            }
            batch = {"pos": md.pos, "vel": md.vel}
            for hook in self.hooks:
                hook.on_train_batch_end(self, state, batch, outputs)
        for hook in self.hooks:
            hook.on_train_end(self, state)
        return {"pos": md.pos, "vel": md.vel, "force": md.force, "state": state}


class TrajectoryHook(BaseHook):
    """Capture an MD trajectory to host buffers; persist at run end.

    Strided frames are copied to CPU and accumulated on the host; every
    ``flush_every`` kept frames the buffer is spilled to an on-disk shard and
    cleared, so host memory stays O(``flush_every`` · N) regardless of run
    length. At ``on_train_end`` the shards (if any) are concatenated into one
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

    _SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
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
        self._buf: dict[str, list[torch.Tensor]] = {k: [] for k in self._FIELDS}
        self._n_buffered = 0
        self._shards: list[Path] = []

    def on_train_batch_end(self, trainer: Any, state: TrainState, batch: Any, outputs: Any) -> None:
        if state["global_step"] % self._stride:
            return
        b = self._buf
        b["pos"].append(outputs["pos"].detach().to("cpu"))
        b["vel"].append(outputs["vel"].detach().to("cpu"))
        if self._with_forces and outputs.get("forces") is not None:
            b["f"].append(outputs["forces"].detach().to("cpu"))
        b["pe"].append(outputs["potential"].detach().to("cpu"))
        b["ke"].append(outputs["kinetic"].detach().to("cpu"))
        b["etot"].append(outputs["total"].detach().to("cpu"))
        b["temp"].append(outputs["temperature"].detach().to("cpu"))
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

    def on_train_end(self, trainer: Any, state: TrainState) -> None:
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
