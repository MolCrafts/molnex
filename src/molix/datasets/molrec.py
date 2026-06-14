"""Read a labeled-configuration dataset from a molrs ``MolRec`` zarr record.

A *labeled-configuration dataset* (e.g. the output of distilling a teacher
potential over a set of conformers) is stored as an ordinary, **unchanged**
:class:`molrs.MolRec` record:

* **Conformers** live in the record's ``trajectory`` (one frame each, an
  ``atoms`` block of ``element``/``x``/``y``/``z`` plus a ``box``). Frames may
  have different atom counts (e.g. QM9's distinct molecules).
* **Teacher labels** live in the flat ``observables`` map, keyed by the molnex
  naming convention ``"<teacher_id>.<name>"`` (e.g. ``teacherA.energy``,
  ``teacherA.forces``). Scalars are ``ScalarObservable``; per-atom forces are a
  ``VectorObservable`` with ``axes=["timestep","atom","component"]`` and shape
  ``(n_frames, n_atoms, 3)``. Labels are arbitrary — there is no fixed schema.
* **Teacher provenance** lives in the record's free-form ``method`` JSON tree
  (e.g. ``method["teacherA"] = {"theory_level": ..., "units": ...}``).

This convention needs **no molrs change**: it builds entirely on ``MolRec``'s
generic primitives (arbitrary observable keys/axes; arbitrary ``method`` JSON).

Pinned ``molrs.MolRec`` API (verified against the installed wheel):
    write  rec = molrs.MolRec(); rec.set_trajectory(molrs.Trajectory.from_frames(frames));
           rec.observables.add_scalar/add_vector(name, data, unit=, axes=,
           time_dependent=, domain=); rec.method = {...}; rec.write_zarr(path)
    read   L = molrs.MolRec.read_zarr(path); L.count_frames();
           L.trajectory.frames[i]["atoms"].view("element"|"x"|"y"|"z");
           L.trajectory.frames[i].box.matrix; L.observables.keys();
           obs = L.observables.get(key); obs.kind ("scalar"|"vector"); obs.data
    Note   molrs zarr storage returns float64 on read-back. This source casts
           positions/targets to ``float32`` (matching the other molix sources,
           e.g. :class:`~molix.datasets.RevMD17Source`); values are lossless
           because float32 is exactly representable in float64.

Unlike :class:`~molix.datasets.QM9Source` / :class:`~molix.datasets.RevMD17Source`
this source ships **no** frozen class-level ``TARGET_SCHEMA``; it builds a
per-instance :class:`~molix.data.collate.TargetSchema` from whichever observables
the selected teacher exposes (scalar -> ``graph_level``, vector -> ``atom_level``).

References:
    QM9 — Ramakrishnan et al. "Quantum chemistry structures and properties of
    134 kilo molecules." Scientific Data 1, 140022 (2014).
    https://doi.org/10.1038/sdata.2014.22

    rMD17 — Christensen & von Lilienfeld. "On the role of gradients for machine
    learning of molecular energies and forces." Mach. Learn.: Sci. Technol. 1
    (2020). https://doi.org/10.1088/2632-2153/abba6f
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from molpy.core.element import Element

from molix.data.collate import TargetSchema
from molix.data.source import Sample


class MolRecSource:
    """Lazy-free ``DataSource`` over a labeled-configuration ``MolRec`` record.

    Conforms structurally to :class:`molix.data.source.DataSource` (it is *not*
    a subclass — molix data sources are duck-typed). The record is eagerly read
    in :meth:`__init__`; each ``__getitem__`` returns one conformer as a flat
    sample dict.

    Args:
        record_path: Path to the ``.zarr`` record written by
            :meth:`molrs.MolRec.write_zarr`.
        teacher_id: Which teacher's labels to expose. Selects the
            ``"<teacher_id>."``-prefixed observables; the prefix is stripped to
            form ``targets`` names.
        total: Optional first-N subset (for smokes / quick tests). Folded into
            :attr:`source_id` so subset and full sources cache separately.

    Raises:
        FileNotFoundError: If ``record_path`` does not exist.
        ValueError: If ``teacher_id`` matches no observable key; the message
            lists the available teacher prefixes.
    """

    def __init__(
        self, record_path: str | Path, teacher_id: str, *, total: int | None = None
    ) -> None:
        self.record_path = Path(record_path)
        self.teacher_id = teacher_id
        self.total = total
        if not self.record_path.exists():
            raise FileNotFoundError(f"MolRec record not found: {self.record_path}")

        # Imported lazily: molrs is a native (Rust/pyo3) dependency with no wheel
        # for every platform (e.g. linux-aarch64). Only MolRecSource needs it, so
        # keeping the import here lets the rest of ``molix.datasets`` (QM9Source,
        # RevMD17Source, …) load on platforms where molrs is unavailable.
        import molrs

        record = molrs.MolRec.read_zarr(str(self.record_path))

        prefix = f"{teacher_id}."
        all_keys = list(record.observables.keys())
        my_keys = [k for k in all_keys if k.startswith(prefix)]
        if not my_keys:
            available = sorted({k.split(".", 1)[0] for k in all_keys if "." in k})
            raise ValueError(
                f"teacher_id {teacher_id!r} matches no observables in {self.record_path}. "
                f"Available teacher prefixes: {available}"
            )

        graph_level: set[str] = set()
        atom_level: set[str] = set()
        scalar_targets: dict[str, np.ndarray] = {}
        vector_targets: dict[str, np.ndarray] = {}
        for key in my_keys:
            name = key[len(prefix) :]
            obs = record.observables.get(key)
            data = np.asarray(obs.data)
            if obs.kind == "scalar":
                graph_level.add(name)
                scalar_targets[name] = data
            elif obs.kind == "vector":
                atom_level.add(name)
                vector_targets[name] = data
            else:
                raise ValueError(
                    f"unsupported observable kind {obs.kind!r} for key {key!r}; "
                    "MolRecSource handles 'scalar' and 'vector' observables"
                )

        # Per-conformer geometry: Z (mapped from element symbol), pos, box.
        n_frames = record.count_frames()
        frames = record.trajectory.frames
        z_per_frame: list[torch.Tensor] = []
        pos_per_frame: list[torch.Tensor] = []
        box_per_frame: list[torch.Tensor | None] = []
        for i in range(n_frames):
            atoms = frames[i]["atoms"]
            elements = [str(e) for e in atoms.view("element")]
            z_per_frame.append(
                torch.tensor([Element.get_atomic_number(e) for e in elements], dtype=torch.long)
            )
            xyz = np.stack(
                [
                    np.asarray(atoms.view("x"), dtype=np.float64),
                    np.asarray(atoms.view("y"), dtype=np.float64),
                    np.asarray(atoms.view("z"), dtype=np.float64),
                ],
                axis=1,
            )
            pos_per_frame.append(torch.from_numpy(xyz).float())
            box = getattr(frames[i], "box", None)
            box_per_frame.append(
                torch.from_numpy(np.asarray(box.matrix, dtype=np.float64)).float()
                if box is not None
                else None
            )

        if total is not None and total < n_frames:
            sl = slice(0, total)
            z_per_frame = z_per_frame[sl]
            pos_per_frame = pos_per_frame[sl]
            box_per_frame = box_per_frame[sl]
            scalar_targets = {k: v[sl] for k, v in scalar_targets.items()}
            vector_targets = {k: v[sl] for k, v in vector_targets.items()}

        self._z = z_per_frame
        self._pos = pos_per_frame
        self._box = box_per_frame
        self._scalar_targets = scalar_targets
        self._vector_targets = vector_targets
        self._n = len(z_per_frame)
        #: Per-instance target schema discovered from the selected teacher.
        self.target_schema = TargetSchema(
            graph_level=frozenset(graph_level), atom_level=frozenset(atom_level)
        )

    @property
    def source_id(self) -> str:
        """Cache-key identity ``molrec:<stem>:size=<bytes>:n=<n>:teacher=<id>``.

        Appends ``:total=<n>`` when a subset was requested. The ``:teacher=``
        component is the only lever needed for per-teacher
        :class:`~molix.data.cache.PackedCache` invalidation: ``source_id`` is
        the root of the pipeline cache-key chain.
        """
        size = sum(f.stat().st_size for f in self.record_path.rglob("*") if f.is_file())
        sid = f"molrec:{self.record_path.stem}:size={size}:n={len(self)}:teacher={self.teacher_id}"
        if self.total is not None:
            sid += f":total={self.total}"
        return sid

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Sample:
        """Return the ``idx``-th conformer as a flat sample dict.

        Returns:
            ``{"Z": (N,) long, "pos": (N, 3) float32, "box": (3, 3) float32
            (when present), "targets": {<name>: tensor}}``. Scalar targets have
            shape ``(1,)``; per-atom (vector) targets have shape ``(N, 3)``.
            Target names are the teacher's observable keys with the
            ``"<teacher_id>."`` prefix stripped.
        """
        targets: dict[str, torch.Tensor] = {}
        for name, arr in self._scalar_targets.items():
            targets[name] = torch.tensor([float(arr[idx])], dtype=torch.float32)
        for name, arr in self._vector_targets.items():
            targets[name] = torch.from_numpy(np.asarray(arr[idx], dtype=np.float64)).float()
        sample: Sample = {"Z": self._z[idx], "pos": self._pos[idx], "targets": targets}
        if self._box[idx] is not None:
            sample["box"] = self._box[idx]
        return sample


__all__ = ["MolRecSource"]
