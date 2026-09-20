"""Class-I fit, composed from molix primitives.

Predicted energy is Class-I plus atomic dressing
``E = E_MM + Σ_i ε[Z_i]``. ε is fitted by MolHub ``AtomicDress`` and looked
up by ``molpot.heads.energy.AtomicReferenceEnergy`` (MACE
``AtomicEnergiesBlock``). The energy loss is residualized: Huber of the undressed ``E_MM`` against
``E_DFT − Σ ε[Z]`` (target built in float64). Fitting the ~O(10) kcal/mol
residual instead of the ~1e6 kcal/mol dressed total keeps the loss
well-conditioned in float32 and removes the frozen dressing from the
learnable gradient. Huber (``beta=20`` kcal/mol) saturates the
``(σ/r)¹²`` spikes that otherwise dominate the energy gradient once the
LJ head leaves zero-energy init. Forces are Huber (``beta=20`` kcal/mol/Å)
on atoms whose ``|F_ref|`` is at most 200 kcal/mol/Å (the inspect outlier
threshold); Class-I MM cannot represent those DFT tails, and chasing them
grows σ into the wall. Unit weight on the two terms.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from molhub.config import config as molhub_config
from molhub.dataset.protocol import SubsetDataset
from molhub.dataset.statistic import Field
from molhub.loader import DataModule
from molpy import Block, Frame
from tensordict import TensorDict

from molix.core.hook import BaseHook
from molix.core.losses import energy_loss
from molix.core.steps import DefaultEvalStep, DefaultTrainStep
from molix.core.trainer import Trainer
from molix.data import (
    DataModuleProtocol,
    FrameCollator,
    MoleculeCollator,
    MoleculePadder,
    TotalCharge,
)
from molix.data.frame_dataset import FrameDataset
from molix.hooks import CheckpointHook, GradClipHook, Log, MolRecMetricsHook, StepSpeedHook
from molix.units import units
from molpot.heads.energy import AtomicReferenceEnergy
from molzoo.rondo import Rondo, RondoPotential, RondoSpec

TASK_ENERGY = "energy"
TASK_CODE_ENERGY = 0

# OMol25 publishes eV / eV·Å⁻¹; Class-I training is kcal/mol (LAMMPS real).
_EV_TO_KCAL = float((1.0 * units.eV).to(units.kilocalorie_per_mole).magnitude)
_EV_PER_A_TO_KCAL_PER_A = _EV_TO_KCAL

# Saturates (σ/r)^12 energy spikes; kcal/mol. Quadratic below, linear above.
ENERGY_HUBER_BETA = 20.0
# Saturates exploded pred-forces when σ walks into the wall; kcal/mol/Å.
FORCE_HUBER_BETA = 20.0
# Same threshold as scale_audit inspect. Atoms above this |F_ref| are dropped
# from the force term — Class-I cannot represent 200–800 kcal/mol/Å DFT tails.
FORCE_OUTLIER_KCAL_A = 200.0


def inlier_weight(ref: torch.Tensor) -> torch.Tensor:
    """Per-atom ``1.0``/``0.0`` weight: is ``|F_ref|`` within the outlier bound?

    Args:
        ref: Reference Cartesian forces ``(N, 3)`` (kcal mol⁻¹ Å⁻¹).

    Returns:
        Weights ``(N, 1)`` in ``ref``'s dtype, broadcastable over components.
    """
    keep = ref.detach().norm(dim=-1) <= FORCE_OUTLIER_KCAL_A
    return keep.to(ref.dtype).unsqueeze(-1)


def force_huber(
    pred: torch.Tensor, ref: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Huber of Cartesian forces over DFT inliers (``beta=FORCE_HUBER_BETA``).

    The outlier bound is applied as a **weight**, not as a boolean gather. The
    gather it replaces produced a data-dependent output shape and a host sync
    (``bool(keep.any())``) — a graph break on the training hot path, forbidden
    by the compile law — and made the Huber denominator the number of surviving
    atoms, so the loss scale moved with how many outliers a batch happened to
    draw. Weighting divides by that same count without either.

    A batch whose atoms are *all* outliers contributes ``0`` rather than
    silently falling back to fitting the outliers, which is what the boolean
    form did.

    Args:
        pred: Predicted forces ``(N, 3)``.
        ref: Reference forces ``(N, 3)``.
        weight: Optional per-atom ``(N,)`` weight — ``atoms.mask`` on a padded
            batch, so ghost rows neither score nor count.

    Returns:
        Scalar loss (kcal mol⁻¹ Å⁻¹).
    """
    kept_weight = inlier_weight(ref)
    if weight is not None:
        kept_weight = kept_weight * weight.reshape(-1, 1).to(kept_weight.dtype)
    per_component = torch.nn.functional.smooth_l1_loss(
        pred, ref, beta=FORCE_HUBER_BETA, reduction="none"
    )
    kept = kept_weight.sum() * pred.shape[-1]
    return (per_component * kept_weight).sum() / kept.clamp(min=1.0)


def force_mae(
    pred: torch.Tensor, ref: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """MAE of Cartesian forces, over ``weight``-ed atoms when a weight is given."""
    per_component = (pred - ref).abs()
    if weight is None:
        return per_component.mean()
    w = weight.reshape(-1, 1).to(per_component.dtype)
    return (per_component * w).sum() / (w.sum() * pred.shape[-1]).clamp(min=1.0)


def _atom_mask(batch: TensorDict) -> torch.Tensor | None:
    """``atoms.mask`` of a padded batch, or ``None`` when every atom is real."""
    atoms = batch.get("atoms")
    if atoms is None or "mask" not in atoms.keys():
        return None
    return atoms["mask"]


def _graph_mask(batch: TensorDict) -> torch.Tensor | None:
    """``graphs.mask`` of a padded batch, or ``None`` when every graph is live."""
    graphs = batch.get("graphs")
    if graphs is None or "mask" not in graphs.keys():
        return None
    return graphs["mask"]


class _LabeledFrames:
    """FrameDataset as pre-collate dicts with task/group graph labels."""

    def __init__(self, dataset: FrameDataset) -> None:
        self.dataset = dataset
        codes = {gid: i for i, gid in enumerate(dict.fromkeys(dataset.group_ids))}
        self._codes = tuple(codes[gid] for gid in dataset.group_ids)
        self.task_labels = dataset.task_labels
        self.group_ids = dataset.group_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], ...]:
        return self._samples_at(idx)

    def _samples_at(self, idx: int) -> tuple[dict[str, Any], ...]:
        code = float(self._codes[idx])
        return tuple(label_sample(frame, group=code) for frame in self.dataset[idx])


def _valence_blocks(frame: Any) -> dict[str, dict[str, torch.Tensor]]:
    """Angle / proper / improper columns of *frame*'s bond graph.

    The enumeration runs in the molrs Rust kernel, reached through
    :meth:`molpy.Atomistic.get_topo`; molnex keeps no copy of it. Only bonds
    go in — an ``Atomistic`` perceives valence terms from the graph and never
    from a coordinate, which is what makes these terms a property of the
    molecule rather than of the conformer.

    molpy says ``dihedrals``; the batch schema says ``propers``. The rename is
    this function's one job beyond the call.

    Args:
        frame: A molpy ``Frame`` with ``atoms`` and, when bonded, ``bonds``.

    Returns:
        ``{"angles": {...}, "propers": {...}, "impropers": {...}}``, each an
        empty ``(0,)`` column set when the graph has no such term.
    """
    from molpy import Atomistic

    perceived = (
        Atomistic.from_frame(frame)
        .get_topo(gen_angle=True, gen_dihe=True, gen_improper=True)
        .to_frame()
    )
    empty = torch.zeros(0, dtype=torch.long)
    blocks: dict[str, dict[str, torch.Tensor]] = {}
    for out_name, in_name, columns in (
        ("angles", "angles", ("atomi", "atomj", "atomk")),
        ("propers", "dihedrals", ("atomi", "atomj", "atomk", "atoml")),
        ("impropers", "impropers", ("atomi", "atomj", "atomk", "atoml")),
    ):
        if in_name in perceived:
            block = perceived[in_name]
            blocks[out_name] = {
                column: torch.as_tensor(np.asarray(block[column], dtype=np.int64))
                for column in columns
            }
        else:
            blocks[out_name] = {column: empty for column in columns}
    return blocks


def _with_meta(frame: Any, blocks: Mapping[str, Any] | None = None, **extra: Any) -> Frame:
    """Copy *frame* with *blocks* added and ``extra`` merged into its meta.

    ``Frame.meta`` returns a **copy**, so ``frame.meta["task"] = 0`` is a silent
    no-op — the only way to add a meta scalar is to build a new frame. Blocks
    are carried over by reference to their column arrays, so this costs one
    dict walk, not a data copy.

    *blocks* exists so a caller that has new blocks to add writes each column
    **once**. Writing them into *frame* first and copying afterwards writes
    every column twice, and a column write is not free: it goes through the
    schema (``schema.column`` was 31 % of the whole per-sample prepare, and
    the prepare is what bounds loader throughput).

    Args:
        frame: The source molpy ``Frame``.
        blocks: Extra blocks to write, by name. They override same-named
            blocks of *frame*.
        **extra: Meta scalars to add or override. An ``int`` stays integral
            through collation, which is what keeps a task code routable.

    Returns:
        A new ``Frame`` with *frame*'s blocks plus *blocks*, and merged meta.
    """
    meta = {k: getattr(v, "value", v) for k, v in (getattr(frame, "meta", None) or {}).items()}
    meta.update(extra)
    out = Frame(meta=meta)
    added = dict(blocks or {})
    for name in frame.keys():
        if name in added:
            continue
        block = frame[name]
        out[name] = {column: block[column] for column in block.keys()}
    for name, block in added.items():
        out[name] = block
    return out


def label_sample(frame: Any, *, group: float) -> Frame:
    """Label one frame in place with task/group and its valence families.

    Angles/propers/impropers are always written (empty when there are no bonds)
    so a batch never mixes missing and present valence families — a molecule
    with no angles must contribute zero angle rows, not absent ones.

    Args:
        frame: A molpy ``Frame`` carrying ``atoms`` and, when bonded, ``bonds``.
        group: Parameter-group code, recorded as a per-frame meta scalar.

    Returns:
        The same frame, with the valence blocks and the ``task`` / ``group``
        meta scalars added.
    """
    blocks = _valence_blocks(frame)
    # Handed to ``_with_meta`` rather than written into *frame* first: the new
    # frame is built once, so each valence column is written once.
    # No ``total_charge`` here: the charge model's fail-closed net-charge column
    # is declared once at the batch tasks (``TotalCharge()``), not per frame. A
    # frame that publishes its own charge keeps it — the task only fills a
    # column that is absent.
    return _with_meta(
        frame,
        blocks=blocks,
        task=TASK_CODE_ENERGY,
        group=float(group),
    )


#: ``graphs`` is not a padded axis like the others: it is how many graphs one
#: dataset row contributes to a batch, which is 1 for a per-molecule dataset and
#: 24 for a THEMol torsion-scan row. `MoleculePadder.from_counts` sizes the graph
#: slot count from it; without it a row is taken to be one graph and the slot
#: count comes out 24x short.
_EXTENT_FIELDS = (
    "graph.natoms",
    "pairs",
    "bonds",
    "angles",
    "propers",
    "impropers",
    "graphs",
)
#: ``MoleculeExtent.counts()`` key of each scanned field — the namespace the
#: padder and the packer budget under.
_EXTENT_TABLE = {
    "graph.natoms": "atoms",
    "pairs": "pairs",
    "bonds": "bonds",
    "angles": "angles",
    "propers": "propers",
    "impropers": "impropers",
    "graphs": "graphs",
}
#: Largest dataset :func:`class_i_data_module` scans in-process when no frozen
#: extent exists; above it the scan is a job (``scan_class_i_extent.py``).
INLINE_SCAN_MAX = 20_000


def _extent_path(source_id: str, cutoff: float) -> Path:
    digest = hashlib.sha1(f"{source_id}:class-i-topology:cutoff={cutoff:.6g}".encode()).hexdigest()[
        :16
    ]
    return Path(molhub_config.home_dir) / "derived" / "class-i-extent" / f"{digest}.npz"


def _edge_count(frame: Frame, cutoff: float) -> int:
    """Bidirectional cutoff-list rows of one frame: ``#{(i, j), i != j : 0 < r_ij <= cutoff}``.

    The liveness rule of the :class:`molix.data.NeighborList` the collate
    mounts — ``ForceField.neighbor_list()``: open boundary, both
    orientations, and the kernel's own comparison ``0 < r <= cutoff``
    (``get_neighbor_pairs.cpp``: ``(distances <= cutoff) & (distances >
    0)``) — spelled as one ``(N, N)`` distance matrix. A molecule is a few
    hundred atoms at most, so that costs microseconds; routing each frame
    through a one-frame collate and the O(N²) kernel cost **248 ms**, which
    made a full 4m scan days of CPU (measured 2026-09-14). Exact, not an
    estimate: this number is the per-molecule edge cost the packer budgets.
    """
    atoms = frame["atoms"]
    xyz = np.stack(
        [
            np.asarray(atoms["x"], dtype=np.float64),
            np.asarray(atoms["y"], dtype=np.float64),
            np.asarray(atoms["z"], dtype=np.float64),
        ],
        axis=-1,
    )
    n = int(xyz.shape[0])
    if n < 2:
        return 0
    diff = xyz[:, None, :] - xyz[None, :, :]
    r2 = (diff * diff).sum(-1)
    return int(((r2 <= float(cutoff) ** 2) & (r2 > 0.0)).sum())


def _scan_extent_slice(payload: tuple[Any, int, int, float]) -> tuple[int, dict[str, np.ndarray]]:
    """Worker: topology-scan ``dataset[start:stop]``. Picklable for a process pool.

    One extent row is what one **dataset row** costs a batch, summed over every
    frame that row contributes — not the first frame's cost.

    The difference is a factor of 24 on THEMol torsion-scan, where a row is a
    whole constrained profile and the collate flattens all 24 conformers into
    the same batch (``PrepareCollator`` prepares a tuple per sample and the
    packer budgets per row, so a row is indivisible). Reading only ``sample[0]``
    sized the padder as if a row were 33 atoms when it is 792, which makes every
    cap 24x too small and every batch overflow it. OMol25 hid this: there a
    record is exactly one frame, so the first frame *was* the row.

    ``pairs`` is summed per frame rather than multiplied, because the cutoff
    edge count is a function of the geometry and a torsion scan is 24 different
    geometries of the same topology.
    """
    dataset, start, stop, cutoff = payload
    prepare = ClassIPrepare()
    readers = tuple(Field(field) for field in _EXTENT_FIELDS if field not in ("pairs", "graphs"))
    n = int(stop - start)
    columns = {field: np.empty(n, dtype=np.int32) for field in _EXTENT_FIELDS}
    for offset, idx in enumerate(range(start, stop)):
        sample = dataset[idx]
        frames: Sequence[Any] = sample if isinstance(sample, Sequence) else (sample,)
        totals = {field: 0 for field in _EXTENT_FIELDS}
        for frame in frames:
            prepared = prepare.topology(frame)
            for reader in readers:
                totals[reader.key] += int(reader.read(prepared))
            totals["pairs"] += int(_edge_count(prepared, cutoff))
        # Not a Frame field: how many graphs this row puts in a batch.
        totals["graphs"] = len(frames)
        for field in _EXTENT_FIELDS:
            columns[field][offset] = totals[field]
    return start, columns


class MoleculeExtent:
    """Per-molecule topology sizes from one full scan, then frozen.

    Arrays are aligned with the source row index. ``get_max_*_per_molecule``
    returns those constants; it does not walk the dataset. A
    :class:`~molhub.dataset.protocol.SubsetDataset` view :meth:`take`s the
    parent arrays. :meth:`counts` is the table
    :meth:`molix.data.MoleculePadder.from_counts` sizes the static shape
    from and the packer budgets against.

    The edge column is a function of the list cutoff, so an extent is keyed
    by ``(source_id, cutoff)``; a field whose cutoff changes needs a new scan.
    """

    FIELDS = _EXTENT_FIELDS

    def __init__(self, source_id: str, columns: Mapping[str, np.ndarray], *, cutoff: float) -> None:
        self.source_id = source_id
        self.cutoff = float(cutoff)
        self._columns = {field: np.asarray(columns[field], dtype=np.int32) for field in self.FIELDS}
        n = next(iter(self._columns.values())).size
        if any(col.size != n for col in self._columns.values()):
            raise ValueError("MoleculeExtent columns disagree on length")
        self._max = {
            field: int(col.max()) if col.size else 0 for field, col in self._columns.items()
        }

    def column(self, field: str) -> np.ndarray:
        """Stored per-row vector for *field* (source order)."""
        return self._columns[field]

    def counts(self) -> dict[str, torch.Tensor]:
        """Per-molecule counts keyed by batch namespace (``atoms``, ``pairs``, …)."""
        return {
            _EXTENT_TABLE[field]: torch.as_tensor(col, dtype=torch.long)
            for field, col in self._columns.items()
        }

    def take(self, indices: Sequence[int]) -> "MoleculeExtent":
        """The same constants restricted to *indices* into the source."""
        idx = np.asarray(list(indices), dtype=np.int64)
        return MoleculeExtent(
            self.source_id,
            {field: col[idx] for field, col in self._columns.items()},
            cutoff=self.cutoff,
        )

    def get_max_atoms_per_molecule(self) -> int:
        return self._max["graph.natoms"]

    def get_max_edges_per_molecule(self) -> int:
        return self._max["pairs"]

    def get_max_bonds_per_molecule(self) -> int:
        return self._max["bonds"]

    def get_max_angles_per_molecule(self) -> int:
        return self._max["angles"]

    def get_max_propers_per_molecule(self) -> int:
        return self._max["propers"]

    def get_max_impropers_per_molecule(self) -> int:
        return self._max["impropers"]

    def save(self, path: Path | None = None) -> Path:
        """Write the constant arrays. Default path is keyed by ``(source_id, cutoff)``."""
        out = Path(path) if path is not None else _extent_path(self.source_id, self.cutoff)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            _source_id=np.asarray(self.source_id),
            _cutoff=np.asarray(self.cutoff),
            **self._columns,
        )
        return out

    @classmethod
    def load(cls, source_id: str, *, cutoff: float, path: Path | None = None) -> "MoleculeExtent":
        """Read constants written by :meth:`save`. Does not scan."""
        stored = Path(path) if path is not None else _extent_path(source_id, cutoff)
        if not stored.is_file():
            raise FileNotFoundError(
                f"Class-I topology constants missing for {source_id!r} at cutoff {cutoff} "
                f"({stored}). Scan the full source once: "
                f"ClassIPrepare().scan_extent(dataset, cutoff={cutoff}, workers=16).save()"
            )
        data = np.load(stored)
        columns = {field: cls._stored_column(data, field, stored) for field in cls.FIELDS}
        raw_id = np.asarray(data["_source_id"]).reshape(-1)
        loaded_id = str(raw_id[0]) if raw_id.size else source_id
        stored_cutoff = float(np.asarray(data["_cutoff"]).reshape(-1)[0])
        if abs(stored_cutoff - float(cutoff)) > 1e-9:
            raise ValueError(
                f"extent at {stored} was scanned at cutoff {stored_cutoff}, not {cutoff}"
            )
        return cls(loaded_id, columns, cutoff=stored_cutoff)

    #: Column names earlier scans wrote, by the name this class reads today.
    #: The cutoff-list column was ``edges`` until the chemistry layer settled on
    #: ``pairs`` (2026-09-15). A frozen scan is hours of LMDB decoding, so it is
    #: read under its old spelling rather than rescanned — the reader
    #: normalises, which is what every other reader in this tree does.
    LEGACY_FIELDS: dict[str, tuple[str, ...]] = {"pairs": ("edges",)}

    @classmethod
    def _stored_column(cls, data: Any, field: str, stored: Path) -> np.ndarray:
        """One column of a stored scan, accepting the names earlier scans used."""
        for name in (field, *cls.LEGACY_FIELDS.get(field, ())):
            if name in data:
                return np.asarray(data[name])
        raise KeyError(
            f"{stored} has no column {field!r} "
            f"(nor {cls.LEGACY_FIELDS.get(field, ())}); it holds {sorted(data.keys())}"
        )

    @classmethod
    def for_dataset(cls, dataset: Any, *, cutoff: float) -> "MoleculeExtent":
        """Constants for *dataset*, slicing a parent scan when it is a subset."""
        source = getattr(dataset, "source", None)
        if source is not None and hasattr(dataset, "indices"):
            return cls.load(source.source_id, cutoff=cutoff).take(dataset.indices)
        return cls.load(dataset.source_id, cutoff=cutoff)

    @classmethod
    def try_for_dataset(cls, dataset: Any, *, cutoff: float) -> "MoleculeExtent | None":
        """The frozen scan of *dataset*, or ``None`` when none was ever written.

        ``None`` means *nothing was ever frozen for this dataset*: either no
        file (``FileNotFoundError``) or no scan identity at all — an in-memory
        record list has no ``source_id`` and never had a scan
        (``AttributeError``). The caller answers both by scanning inline.

        A scan that **exists but cannot be read** is a different fact and
        raises. Swallowing ``KeyError`` here is what turned a frozen 4m scan
        whose cutoff column still spelled ``edges`` into "no frozen extent …
        run the scan first", sending the operator to redo hours of work that
        was already on disk.
        """
        try:
            return cls.for_dataset(dataset, cutoff=cutoff)
        except (FileNotFoundError, AttributeError):
            return None


#: MolHub publishes energies in the chemistry spelling; molpy's registry names
#: the same quantity differently, and for ``kcal/mol`` the two are not even
#: dimensionally equal: ``units.parse("kcal/mol")`` is energy **per amount**
#: (``L² M T⁻² N⁻¹``) while ``units.kilocalorie_per_mole`` is the per-particle
#: energy chemistry means by it (``L² M T⁻²``), so converting between them
#: raises ``UnitsError: dimension mismatch`` rather than scaling. The two
#: vocabularies are bridged here, once, instead of at every call site.
#:
#: These two spellings are all of MolHub as of 2026-09-18 (``eV`` and
#: ``kcal/mol``); a dataset publishing a third is refused by name, not guessed.
MOLHUB_ENERGY_SPELLING: dict[str, str] = {
    "eV": "eV",
    "kcal/mol": "kilocalorie_per_mole",
}


class ClassIPrepare:
    """Bonds, unit conversion, and labels. Not a Dataset.

    OMol25 Frames stay in published eV with no bonds. This type is the
    Class-I sample transform applied at collate. Override :meth:`bond`,
    :meth:`convert`, or :meth:`label` to change one concern.

    Args:
        energy_unit: Pint unit the labels are converted **to**.
        source_unit: Pint unit the frames already carry, or ``None`` when it
            has not been declared — then :meth:`convert` refuses rather than
            guesses, and only the unit-free concerns (:meth:`bond`,
            :meth:`label`, :meth:`topology`) are available. Prefer
            :meth:`from_dataset`, which reads the unit the dataset publishes
            instead of having it restated here.

    Note:
        This used to default to eV, which is OMol25's unit and was right only
        where this transform started. THEMol publishes kcal mol⁻¹, so that
        default multiplied every label by 23.06 — silently, the result still
        being a plausible energy. A unit is a fact about the dataset in hand,
        so it is read from the dataset, never defaulted here.
    """

    def __init__(
        self,
        *,
        energy_unit: Any = units.kilocalorie_per_mole,
        source_unit: Any = None,
    ) -> None:
        self.energy_unit = energy_unit
        self.source_unit = source_unit
        self.scale = (
            None if source_unit is None else float((1.0 * source_unit).to(energy_unit).magnitude)
        )

    @classmethod
    def from_dataset(
        cls, dataset: Any, *, energy_unit: Any = units.kilocalorie_per_mole
    ) -> "ClassIPrepare":
        """Build one whose source unit is the dataset's own declaration.

        A MolHub dataset states what it publishes —
        ``@dataset(units={"energy": "eV"})`` — and that declaration is the
        authority. Reading it here is what keeps the unit out of the model and
        out of every call site: the transform follows the data rather than
        being told about it.

        Args:
            dataset: A MolHub dataset (or its class) carrying
                ``manifest.units["energy"]``.
            energy_unit: Pint unit the labels are converted **to**.

        Returns:
            A :class:`ClassIPrepare` whose :attr:`source_unit` is the declared one.

        Raises:
            ValueError: *dataset* declares no energy unit — there is nothing to
                read, and guessing is the bug this constructor exists to prevent.
        """
        manifest = getattr(dataset, "manifest", None) or getattr(type(dataset), "manifest", None)
        declared = dict(getattr(manifest, "units", None) or {})
        if "energy" not in declared:
            raise ValueError(
                f"{type(dataset).__name__} declares no energy unit "
                f"(manifest.units = {declared or '{}'}), so the unit its frames "
                "carry cannot be read. Declare it on the dataset with "
                "@dataset(units={'energy': ...}), or pass source_unit= to "
                "ClassIPrepare directly if you know it."
            )
        published = str(declared["energy"])
        spelling = MOLHUB_ENERGY_SPELLING.get(published)
        if spelling is None:
            raise ValueError(
                f"{type(dataset).__name__} publishes energy in {published!r}, which has "
                "no molpy registry spelling here. Add it to MOLHUB_ENERGY_SPELLING "
                f"(known: {sorted(MOLHUB_ENERGY_SPELLING)}) — after checking whether "
                "molpy treats it as a per-particle energy or carries the amount "
                "dimension — or pass source_unit= directly."
            )
        return cls(energy_unit=energy_unit, source_unit=units.parse(spelling))

    def execute(self, frame: Frame) -> Frame:
        """Return a bonded, converted, labeled copy of *frame*."""
        return self.label(self.convert(self.bond(_copy_frame(frame))))

    def bond(self, frame: Frame) -> Frame:
        """Infer covalent bonds when the Frame has none. Leave existing bonds."""
        if "bonds" in frame:
            return frame
        # The dataset-neutral entry point, not the OMol25 adapter: molnex must
        # never import a concrete MolHub dataset module (architecture note,
        # "Library and experiment boundaries").
        from molhub.dataset import infer_bonds

        atoms = frame["atoms"]
        numbers = np.asarray(atoms["number"], dtype=np.int64).reshape(-1)
        xyz = np.stack(
            [
                np.asarray(atoms["x"], dtype=np.float64),
                np.asarray(atoms["y"], dtype=np.float64),
                np.asarray(atoms["z"], dtype=np.float64),
            ],
            axis=-1,
        )
        atomi, atomj = infer_bonds(numbers, xyz)
        if int(atomi.shape[0]) > 0:
            frame["bonds"] = Block({"atomi": atomi, "atomj": atomj})
        return frame

    def convert(self, frame: Frame) -> Frame:
        """Rescale energy and forces from :attr:`source_unit` into :attr:`energy_unit`.

        Raises:
            ValueError: No source unit was declared. Converting on a guess is
                what turned every THEMol label into 23.06x itself; this refuses
                instead.
        """
        if self.scale is None:
            raise ValueError(
                f"{type(self).__name__} has no source unit, so it cannot convert "
                "into "
                f"{self.energy_unit}. Build it with ClassIPrepare.from_dataset("
                "dataset), which reads what the dataset declares, or pass "
                "source_unit= if you know it. (bond/label/topology need no unit "
                "and work without one.)"
            )
        if self.scale == 1.0:
            return frame
        atoms = frame["atoms"]
        if "fx" in atoms:
            atoms["fx"] = np.asarray(atoms["fx"], dtype=np.float64) * self.scale
            atoms["fy"] = np.asarray(atoms["fy"], dtype=np.float64) * self.scale
            atoms["fz"] = np.asarray(atoms["fz"], dtype=np.float64) * self.scale
        raw_meta = getattr(frame, "meta", None) or {}
        meta = {k: getattr(v, "value", v) for k, v in raw_meta.items()}
        if "energy" in meta:
            meta["energy"] = float(meta["energy"]) * self.scale
        frame.meta = meta
        return frame

    def label(self, frame: Frame) -> Frame:
        """Write task/group meta and valence families."""
        return label_sample(frame, group=0.0)

    def topology(self, frame: Frame) -> Frame:
        """Bonds and valence families only — the Frame stats read."""
        return self.label(self.bond(_copy_frame(frame)))

    def scan_extent(self, dataset: Any, *, cutoff: float, workers: int = 1) -> MoleculeExtent:
        """Scan every row of *dataset* once and freeze a :class:`MoleculeExtent`.

        Reads the pad axes off each prepared Frame (``graph.natoms``,
        ``bonds``, ``angles``, ``propers``, ``impropers``) and counts the
        cutoff list at *cutoff* the way the collate builds it. *workers* > 1
        uses a process pool. Nothing is written: the caller decides whether
        to :meth:`MoleculeExtent.save` (the full-role scan does; an in-process
        scan of a small view does not).
        """
        n = len(dataset)
        source_id = str(getattr(dataset, "source_id", ""))
        workers = max(1, int(workers))
        if n == 0:
            empty = {field: np.empty(0, dtype=np.int32) for field in MoleculeExtent.FIELDS}
            return MoleculeExtent(source_id, empty, cutoff=cutoff)
        if workers == 1:
            _start, columns = _scan_extent_slice((dataset, 0, n, float(cutoff)))
            return MoleculeExtent(source_id, columns, cutoff=cutoff)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        chunk = max(int(np.ceil(n / workers)), 1)
        spans = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
        parts: dict[int, dict[str, np.ndarray]] = {}
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_scan_extent_slice, (dataset, start, stop, float(cutoff))): (
                    start,
                    stop,
                )
                for start, stop in spans
            }
            for fut in as_completed(futs):
                start, stop = futs[fut]
                origin, columns = fut.result()
                parts[origin] = columns
                done += stop - start
                print(f"extent {done}/{n} source_id={source_id}", flush=True)
        columns = {
            field: np.concatenate([parts[start][field] for start, _ in spans])
            for field in MoleculeExtent.FIELDS
        }
        return MoleculeExtent(source_id, columns, cutoff=cutoff)


class PrepareCollator:
    """Run a per-frame prepare, then the Frame collator."""

    def __init__(self, prepare: ClassIPrepare, collator: FrameCollator) -> None:
        self.prepare = prepare
        self.collator = collator
        self.frame_collator = collator

    def __call__(self, samples: Sequence[Any]) -> Any:
        prepared: list[tuple[Frame, ...]] = []
        for sample in samples:
            if hasattr(sample, "keys") and "atoms" in sample:
                frames: Sequence[Any] = (sample,)
            else:
                frames = sample
            prepared.append(tuple(self.prepare.execute(frame) for frame in frames))
        return self.collator(prepared)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.collator, name)


def _copy_frame(frame: Frame) -> Frame:
    """Shallow block copy so a prepare can add bonds without mutating *frame*."""
    meta = {k: getattr(v, "value", v) for k, v in (getattr(frame, "meta", None) or {}).items()}
    out = Frame(meta=meta)
    for name in frame.keys():
        block = frame[name]
        out[name] = {column: block[column] for column in block.keys()}
    return out


def _frame_atomic_numbers(frame: Any) -> np.ndarray:
    atoms = frame["atoms"]
    if "number" not in atoms:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(atoms["number"], dtype=np.int64)


def fit_atomic_reference(records: Sequence) -> AtomicReferenceEnergy:
    """Least-squares ``E0[Z]`` via MolHub ``AtomicDress``; lookup is molpot's table.

    Solves ``counts @ ε ≈ E_DFT`` on train frames. The table is frozen
    (MACE ``AtomicEnergiesBlock``). Class-I does not grow a second ε module.
    """
    from molhub.dataset.transforms import AtomicDress

    frames: list[Any] = []
    elements: set[int] = set()
    for record in records:
        for frame in record:
            numbers = _frame_atomic_numbers(frame)
            elements.update(int(z) for z in numbers if int(z) > 0)
            meta = getattr(frame, "meta", None) or {}
            if "energy" in meta and numbers.size:
                frames.append(frame)
    max_z = max(elements) if elements else 0
    max_z = max(max_z, 94)
    if not frames or not elements:
        return _historical_are(torch.zeros(max_z + 1))
    dress = AtomicDress(elements=sorted(elements), target_key="energy")
    dress.fit(frames)
    table = [float(dress.atomic_energies.get(z, 0.0)) for z in range(max_z + 1)]
    return _historical_are(table)


def _historical_are(values: torch.Tensor | Sequence[float]) -> AtomicReferenceEnergy:
    """lstsq kcal table: explicit support, provenance units, empirical scheme."""
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    return AtomicReferenceEnergy(
        atomic_energies=tensor,
        supported=torch.ones(tensor.numel(), dtype=torch.bool),
        scheme="empirical_fit",
        units="kilocalorie_per_mole",
    )


class _FitTrainStep(DefaultTrainStep):
    """Default train step that also publishes ``train/loss_energy``.

    Class-I is one energy task. :class:`~molix.core.steps.MultiTaskTrainStep`
    refuses ``Trainer.compile(step=True)`` because the router host-syncs on
    the task label. The alias is written *after* the compiled region, so it
    is not a graph break. Historical jsonl / Log / CheckpointHook keys stay
    ``loss_energy``.
    """

    scalar_keys = (("train", "loss"), ("train", "loss_energy"))

    def on_train_batch(self, trainer: Any, state: Any, batch: Any) -> dict[str, Any]:
        outputs = super().on_train_batch(trainer, state, batch)
        state["train"]["loss_energy"] = state["train"]["loss"]
        return outputs


class _FitEvalStep(DefaultEvalStep):
    """Eval step with its own objective (MAE), not the train Huber.

    ``compute_loss`` is inside the compiled region and must stay pure tensor
    ops — :func:`val_omol_mae` is. The ``loss_energy`` alias is written after.
    """

    scalar_keys = (("eval", "loss"), ("eval", "loss_energy"))

    def __init__(self, objective: Callable[..., torch.Tensor], *, no_grad: bool = True) -> None:
        super().__init__(no_grad=no_grad)
        self.objective = objective

    def compute_loss(
        self,
        trainer: Any,
        state: Any,
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        return self.objective(predictions, batch)

    def on_eval_batch(self, trainer: Any, state: Any, batch: Any) -> dict[str, Any]:
        outputs = super().on_eval_batch(trainer, state, batch)
        state["eval"]["loss_energy"] = state["eval"]["loss"]
        return outputs


def _graph_tensor(batch: TensorDict, name: str) -> torch.Tensor | None:
    graphs = batch.get("graphs")
    if graphs is None or name not in graphs.keys():
        return None
    return graphs[name]


def energy_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MSE of graph energies (kcal/mol)."""
    return (pred.reshape(-1) - ref.reshape(-1)).square().mean()


def _graph_mean(per_graph: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Mean over graphs, or the ``weight``-ed mean (``graphs.mask`` drops dead slots)."""
    if weight is None:
        return per_graph.mean()
    w = weight.reshape(-1).to(per_graph.dtype)
    return (per_graph * w).sum() / w.sum().clamp(min=1.0)


def energy_huber(
    pred: torch.Tensor, ref: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Huber (smooth L1) of graph energies, ``beta=ENERGY_HUBER_BETA`` kcal/mol."""
    per_graph = torch.nn.functional.smooth_l1_loss(
        pred.reshape(-1), ref.reshape(-1), beta=ENERGY_HUBER_BETA, reduction="none"
    )
    return _graph_mean(per_graph, weight)


def energy_mae(
    pred: torch.Tensor, ref: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """MAE of graph energies (kcal/mol)."""
    return _graph_mean((pred.reshape(-1) - ref.reshape(-1)).abs(), weight)


def energy_force_terms(
    predictions: Mapping[str, torch.Tensor], batch: TensorDict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residualized energy Huber + inlier force MSE. Unit weights, no molecule mean.

    When the model exposes ``energy_mm`` (undressed) and ``e0`` (the frozen
    per-element dressing), the energy target is ``E_DFT − e0`` built in
    float64. This keeps the energy term on the ~O(10) kcal/mol MM scale
    instead of the ~1e6 kcal/mol dressed scale, where a single float32 ULP
    (~0.03 kcal/mol) would otherwise dominate packed-vs-single gradient
    parity. The fallback compares dressed energy vs the raw target.
    """
    energy_term = predictions["energy"].new_zeros(())
    energy = _graph_tensor(batch, "energy")
    if energy is not None:
        energy_mm = predictions.get("energy_mm")
        e0 = predictions.get("e0")
        if energy_mm is not None and e0 is not None:
            target = (energy.reshape(-1).double() - e0.reshape(-1).double()).float()
            energy_term = energy_huber(energy_mm, target, _graph_mask(batch))
        else:
            energy_term = energy_huber(predictions["energy"], energy, _graph_mask(batch))
    force_term = predictions["energy"].new_zeros(())
    atoms = batch.get("atoms")
    if atoms is not None and "forces" in atoms.keys() and "forces" in predictions:
        force_term = force_huber(predictions["forces"], atoms["forces"], _atom_mask(batch))
    return energy_term, force_term


def train_objective(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """Residualized energy Huber + inlier force MSE. Unit weights."""
    energy_term, force_term = energy_force_terms(predictions, batch)
    return energy_term + force_term


def val_omol_mae(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """Dressed energy MAE (kcal/mol) + force MAE (kcal mol⁻¹ Å⁻¹)."""
    energy_term = predictions["energy"].new_zeros(())
    energy = _graph_tensor(batch, "energy")
    if energy is not None:
        energy_term = energy_mae(predictions["energy"], energy, _graph_mask(batch))
    force_term = predictions["energy"].new_zeros(())
    atoms = batch.get("atoms")
    if atoms is not None and "forces" in atoms.keys() and "forces" in predictions:
        force_term = force_mae(predictions["forces"], atoms["forces"], _atom_mask(batch))
    return energy_term + force_term


_PER_ATOM_MAE = energy_loss(
    target_key="energy_residual",
    pred_key="energy",
    per_atom=True,
    elementwise=torch.nn.L1Loss(reduction="none"),
)


def per_atom_mae_energy(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """Per-atom L1 on ``graphs.energy_residual``, divided by protocol ``s_E``."""
    from scripts.mm_param_learning.configs import OMOL25_FAIRCHEM_4M

    scale = float(OMOL25_FAIRCHEM_4M["energy_rms"])
    # Reference arithmetic is float64 (protocol refs_dtype); float32 ULP misses 1e-10.
    return _PER_ATOM_MAE({"energy": predictions["energy"].double()}, batch) / scale


def per_structure_l2_force(
    predictions: Mapping[str, torch.Tensor], batch: TensorDict
) -> torch.Tensor:
    """Mean over graphs of mean-atom Euclidean force error, divided by ``s_F``.

    Ghost rows contribute 0 through ``atoms.mask`` (multiply, never a gather)
    and dead graph slots drop out of the graph mean through ``graphs.mask``.
    Missing masks treat every atom / graph as real. Zero-error structures
    stay in the graph mean.
    """
    from scripts.mm_param_learning.configs import OMOL25_FAIRCHEM_4M

    scale = float(OMOL25_FAIRCHEM_4M["force_rms"])
    delta = predictions["forces"].double() - batch["atoms", "forces"].double()
    atoms = batch["atoms"]
    if "mask" in atoms.keys():
        real = atoms["mask"].to(dtype=delta.dtype)
        delta = delta * real.unsqueeze(-1)
    else:
        real = torch.ones(delta.shape[0], dtype=delta.dtype, device=delta.device)
    atom_l2 = delta.norm(dim=-1)
    atom_batch = atoms["batch"].to(dtype=torch.long)
    n_graphs = batch["graphs", "num_atoms"].shape[0]
    per_sum = atom_l2.new_zeros(n_graphs)
    per_n = atom_l2.new_zeros(n_graphs)
    per_sum.index_add_(0, atom_batch, atom_l2)
    per_n.index_add_(0, atom_batch, real)
    return _graph_mean(per_sum / per_n.clamp(min=1.0) / scale, _graph_mask(batch))


def fairchem_4m_objective(
    predictions: Mapping[str, torch.Tensor], batch: TensorDict
) -> torch.Tensor:
    """Named protocol loss ``10 L_E + 10 L_F`` on residual energy and forces."""
    from scripts.mm_param_learning.configs import OMOL25_FAIRCHEM_4M

    energy_w = float(OMOL25_FAIRCHEM_4M["energy_coefficient"])
    force_w = float(OMOL25_FAIRCHEM_4M["force_coefficient"])
    return energy_w * per_atom_mae_energy(predictions, batch) + force_w * per_structure_l2_force(
        predictions, batch
    )


def _list_capacity(atom_counts: Any) -> int:
    """``max_num_pairs`` for a list built over these molecules.

    :class:`molix.data.NeighborList` runs its O(N²) kernel **per molecule**
    and grows the buffer to that molecule's own complete half graph when the
    configured floor is smaller, so this number is a *per-molecule* bound —
    the largest molecule's ``n(n-1)/2`` — never the batch's edge capacity.
    Handing it the batch capacity instead makes every molecule allocate the
    whole pack's worth of pair buffers: at 65536 atom slots that is 4.95 M
    pairs per molecule per worker, which took the host out of memory rather
    than the GPU (GH200 sweep round 2, 11.7 GiB RSS cap).
    """
    n = int(torch.as_tensor(atom_counts).max())
    return max(512, n * (n - 1) // 2)


def _extent_for(fit: "ClassIFit", dataset: Any, *, cutoff: float) -> MoleculeExtent:
    """The frozen scan of *dataset* at *cutoff*, or an in-process scan of a small view.

    A dataset above :data:`INLINE_SCAN_MAX` rows without a frozen scan is
    refused: walking OMol25 4m inside a loader constructor is hours of
    surprise, and the scan is a job (``scan_class_i_extent.py``).
    """
    extent = MoleculeExtent.try_for_dataset(dataset, cutoff=cutoff)
    if extent is not None:
        return extent
    if len(dataset) > INLINE_SCAN_MAX:
        raise FileNotFoundError(
            f"no frozen Class-I extent for {getattr(dataset, 'source_id', dataset)!r} at "
            f"cutoff {cutoff} and the dataset ({len(dataset)} rows) is too large to scan "
            "here; run scripts/mm_param_learning/scan_class_i_extent.py first."
        )
    # scan_extent reads topology only, so no source unit is needed here; a
    # ClassIPrepare built without one refuses to convert rather than guessing.
    prepare = fit.prepare if fit.prepare is not None else ClassIPrepare(energy_unit=fit.energy_unit)
    return prepare.scan_extent(dataset, cutoff=cutoff, workers=1)


def class_i_data_module(
    fit: "ClassIFit",
    *,
    atoms: int | None = None,
    slack: float = 1.10,
    train_counts: Mapping[str, torch.Tensor] | None = None,
    val_counts: Mapping[str, torch.Tensor] | None = None,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = "spawn",
) -> DataModule:
    """Compose the molhub DataModule this fit's field needs.

    One static shape for the whole run: the padder is sized from the train
    split's per-molecule counts at atom capacity *atoms*
    (:meth:`molix.data.MoleculePadder.from_counts` — every cap is the
    dataset density times the atom budget, never below the largest
    molecule), and the DataModule packs train and val under that padder's
    own budgets (``batch_counts`` / ``val_batch_counts``), so no batch can
    exceed any cap and the compiled step never sees a second shape. The
    molecule count per batch floats; the atom capacity is the knob.

    Args:
        fit: Fit whose ``train_ds`` / ``val_ds`` / ``forcefield`` /
            ``batch_tasks`` / ``seed`` / ``device`` are read.
        atoms: Padded atom capacity (ghost slots included; a power of two
            picked by the HBM sweep). Default ``fit.max_atoms``.
        slack: Head-room factor on every density-derived cap.
        train_counts: Per-molecule count tables of the train split
            (``MoleculeExtent.counts()`` shape). Default: the frozen scan of
            the split at the field's cutoff, or an in-process scan of a small
            view (:func:`_extent_for`) — pass them to skip both.
        val_counts: The same for the val split.
        num_workers: DataLoader workers.
        prefetch_factor: Batches prefetched per worker.
        multiprocessing_context: Worker start method.

    Returns:
        A molhub :class:`~molhub.loader.DataModule` satisfying molix
        ``DataModuleProtocol``.
    """
    on_cuda = fit.device.type == "cuda"
    cutoff = float(fit.forcefield.neighbor_list().cutoff)
    if train_counts is None:
        train_counts = _extent_for(fit, fit.train_ds, cutoff=cutoff).counts()
    if val_counts is None:
        val_counts = _extent_for(fit, fit.val_ds, cutoff=cutoff).counts()
    capacity = int(fit.max_atoms if atoms is None else atoms)
    padder = MoleculePadder.from_counts(train_counts, atoms=capacity, cutoff=cutoff, slack=slack)
    nlist = fit.forcefield.neighbor_list(max_num_pairs=_list_capacity(train_counts["atoms"]))
    print(
        "class_i_pad "
        + json.dumps({"caps": padder.caps(), "budgets": padder.budgets(), "cutoff": cutoff}),
        flush=True,
    )
    collate_fn: Any = FrameCollator(
        collator=MoleculeCollator(),
        padder=padder,
        prepare=fit.prepare,
    )
    return DataModule(
        fit.train_ds,
        fit.val_ds,
        collate_fn=collate_fn,
        batch_tasks=[
            nlist,
            TotalCharge(),
            *fit.batch_tasks,
        ],
        batch_counts=dict(train_counts),
        val_batch_counts=dict(val_counts),
        num_workers=num_workers,
        pin_memory=on_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        seed=fit.seed,
    )


_GROUP_META = ("scan_id", "molecule_id", "source_record_id")


def _require_group(frame: Any) -> str:
    """Molecule identity for the train/val split. Refuses a silent ``ungrouped`` merge."""
    meta = getattr(frame, "meta", None) or {}
    for key in _GROUP_META:
        if key in meta:
            value = meta[key]
            return str(getattr(value, "value", value))  # molpy 0.14 MetaValue unwrap
    raise ValueError(
        "Class-I frame has no scan_id/molecule_id/source_record_id; "
        "refusing to merge distinct molecules into one split group"
    )


class ClassIFit:
    """One Class-I train/eval step over already-loaded FrameDatasets."""

    def __init__(
        self,
        train_records: Sequence,
        val_records: Sequence,
        *,
        train_task: str = TASK_ENERGY,
        val_task: str = TASK_ENERGY,
        hidden: int = 64,
        n_blocks: int = 4,
        n_torsion_terms: int = 8,
        max_atoms: int = 4096,
        epochs: int = 2,
        lr: float = 1e-4,
        seed: int = 2666,
        device: str = "cpu",
        record_root: Path | None = None,
        energy_scope: str = "full",
        eval_every_epochs: int = 1,
        eval_every_n_steps: int | None = None,
        eval_loss: Callable[..., torch.Tensor] | None = None,
        loss: Callable[..., torch.Tensor] = train_objective,
        optimizer_factory: Callable[..., Any] | None = None,
        lr_scheduler_factory: Callable[..., Any] | None = None,
        batch_tasks: Sequence[Any] = (),
        dressing: AtomicReferenceEnergy | None = None,
        energy_unit: Any = units.kilocalorie_per_mole,
        clip_max_norm: float = 1.0e3,
        log_every_n_steps: int = 500,
        metrics_every_n_steps: int = 100,
    ) -> None:
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.n_torsion_terms = n_torsion_terms
        #: Padded atom capacity (ghost slots included) the loader packs to —
        #: :func:`class_i_data_module`'s default ``atoms``.
        self.max_atoms = max_atoms
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.record_root = record_root
        self.energy_scope = energy_scope
        self.eval_every_epochs = max(1, int(eval_every_epochs))
        # An epoch is the stop unit only while an epoch is short. On the 4M
        # OMol25 role one epoch is ~1.2e5 steps, so an epoch-boundary eval
        # fires twice in a 24 h run — too few points to see a trend or to pick
        # a checkpoint by. ``eval_every_n_steps`` states the stride directly.
        # Exactly one of the two knobs may be non-default: two spellings of the
        # same cadence silently disagreeing is the failure this refuses.
        if eval_every_n_steps is not None:
            if int(eval_every_n_steps) < 1:
                raise ValueError(f"eval_every_n_steps must be >= 1, got {eval_every_n_steps}")
            if int(eval_every_epochs) != 1:
                raise ValueError(
                    "pass eval_every_epochs or eval_every_n_steps, not both "
                    f"(got {eval_every_epochs} epochs and {eval_every_n_steps} steps)"
                )
        self.eval_every_n_steps = None if eval_every_n_steps is None else int(eval_every_n_steps)
        self.eval_loss = eval_loss if eval_loss is not None else val_omol_mae
        self.loss = loss
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.Adam(params, lr=self.lr)
        )
        self.lr_scheduler_factory = lr_scheduler_factory
        self.batch_tasks = tuple(batch_tasks)
        self.energy_unit = energy_unit
        self.clip_max_norm = float(clip_max_norm)
        # Both sinks materialise device scalars (``grad_norm`` is a 0-d CUDA
        # tensor kept undetached on purpose), so each sampled step costs one
        # GPU->CPU sync. Sampling every step buys no resolution a 3e5-step run
        # can use and pays that sync 3e5 times.
        self.log_every_n_steps = int(log_every_n_steps)
        self.metrics_every_n_steps = int(metrics_every_n_steps)
        #: The Rondo model this fit trains. Built once, in the constructor,
        #: because :attr:`forcefield` — the field whose pair list
        #: :func:`class_i_data_module` mounts — is *its* field. One object
        #: owns both the parameter tables and the nonbonded radius, so the
        #: graph the batches carry and the radius the model prices cannot
        #: drift apart (ledger finding **F21**).
        self.spec = RondoSpec(
            hidden=hidden,
            n_blocks=n_blocks,
            n_torsion_terms=n_torsion_terms,
        )
        self.parameterizer = Rondo(self.spec)
        if isinstance(train_records, (list, tuple)):
            self.train_ds = _LabeledFrames(
                FrameDataset.from_records(train_records, task=train_task, group_of=_require_group)
            )
            self.val_ds = _LabeledFrames(
                FrameDataset.from_records(val_records, task=val_task, group_of=_require_group)
            )
            self.prepare = None
            self.dressing = (
                dressing if dressing is not None else fit_atomic_reference(train_records)
            )
        else:
            self.train_ds = train_records
            self.val_ds = val_records
            # This branch is the one where *train_records* is a dataset rather
            # than a list of frames, so its declared unit is right here to read.
            self.prepare = ClassIPrepare.from_dataset(train_records, energy_unit=energy_unit)
            self.dressing = dressing

    @property
    def forcefield(self) -> Any:
        """The one field of this fit — the model's own (ledger finding **F21**).

        Reading it off the model is what makes a pair list built here and the
        pairs the model prices the same fact, rather than two equal-looking
        objects kept in step by hand.
        """
        return self.parameterizer.template

    def model(self) -> RondoPotential:
        """Wrap this fit's Rondo parameterizer as an energy/force model.

        The dressing table is cloned so the fit's own frozen reference
        energies are never mutated by the model that reads them.
        """
        cloned = None
        if self.dressing is not None:
            cloned = AtomicReferenceEnergy(
                atomic_energies=self.dressing.atomic_energies.detach().cpu().clone(),
                supported=self.dressing.supported.detach().cpu().clone(),
                scheme=self.dressing.scheme,
                units=self.dressing.units,
            )
        return RondoPotential(self.parameterizer, dressing=cloned, energy_unit=self.energy_unit).to(
            self.device
        )

    def trainer(self, model: nn.Module) -> Trainer:
        """Build the historical Trainer primitive. Official cosine is run()-only."""
        if self.loss is fairchem_4m_objective and self.lr_scheduler_factory is None:
            raise ValueError(
                "fairchem_4m_objective requires a cosine lr_scheduler_factory; "
                "compose it in omol25_fairchem_4m.run"
            )
        if self.record_root is not None:
            checkpoint_dir = self.record_root / "checkpoints"
        else:
            checkpoint_dir = Path("checkpoints")
        hooks: list[BaseHook] = [
            GradClipHook(max_norm=self.clip_max_norm),
            StepSpeedHook(window_size=4),
            Log(
                every_n_steps=self.log_every_n_steps,
                keys=(
                    ("train", "loss"),
                    ("train", "grad_norm"),
                    ("eval", "loss"),
                    ("performance", "step_per_second"),
                ),
            ),
            CheckpointHook(
                checkpoint_dir=str(checkpoint_dir),
                save_best=True,
                save_last=True,
                best_metric_name=("eval", "loss"),
                best_metric_mode="min",
            ),
        ]
        if self.record_root is not None:
            hooks.append(
                MolRecMetricsHook(
                    record_root=self.record_root,
                    every_n_steps=self.metrics_every_n_steps,
                )
            )
        return Trainer(
            model,
            loss_fn=self.loss,
            optimizer_factory=self.optimizer_factory,
            lr_scheduler_factory=self.lr_scheduler_factory,
            train_step=_FitTrainStep(),
            eval_step=_FitEvalStep(self.eval_loss, no_grad=True),
            hooks=hooks,
            eval_every_n_steps=self.eval_every_n_steps,
            device=str(self.device),
        )

    def run(
        self,
        datamodule: DataModuleProtocol,
        *,
        resume_from: Path | str | None = None,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        """Train on OMol25 labels; report the configured eval loss.

        Args:
            datamodule: Loader composed by the caller (typically
                :func:`class_i_data_module`).
        """
        if self.loss is fairchem_4m_objective:
            raise ValueError(
                "ClassIFit.run cannot train fairchem_4m_objective; "
                "compose attach_omol_elem_refs and extra-role eval in "
                "omol25_fairchem_4m.run"
            )
        model = self.model()
        trainer = self.trainer(model)
        if self.eval_every_n_steps is None and self.eval_every_epochs > 1:
            steps_per_epoch = max(1, len(datamodule.train_dataloader()))
            trainer.eval_every_n_steps = steps_per_epoch * self.eval_every_epochs
        # Loader already has the one padder. Compile the train body only —
        # eval compile leaks encoder ``grad_fn``s (GH200 2363952). Do not
        # fall back to eager.
        from molix.compile import Compiler

        trainer.train_step._traced_forward_loss = Compiler(fullgraph=True)(
            trainer.train_step._traced_forward_loss
        )
        trainer._resume_from_checkpoint = None if resume_from is None else str(resume_from)
        trainer.smoke(datamodule, steps=3)
        trainer.train(datamodule, max_epochs=self.epochs, max_steps=max_steps)
        val_mae = trainer.state.get(("eval", "loss"))
        if val_mae is None:
            val_mae = trainer.state.get(("eval", "loss_energy"))
        train_loss = trainer.state.get(("train", "loss"))
        if train_loss is None:
            train_loss = trainer.state.get(("train", "loss_energy"))
        step_per_second = trainer.state.get(("performance", "step_per_second"))
        return {
            # Where the run stopped. A resume is only demonstrated by a step
            # counter that moved past the checkpoint's, and ``max_steps`` is an
            # absolute target: ``max_steps=1`` on a run resumed at step 16 is
            # already satisfied and trains nothing.
            "global_step": float(trainer.state.global_step),
            "train_loss": float(train_loss) if train_loss is not None else float("nan"),
            "val_mae_kcal": float(val_mae) if val_mae is not None else float("nan"),
            "step_per_second": (
                float(step_per_second) if step_per_second is not None else float("nan")
            ),
            "epochs": float(self.epochs),
            "n_train": float(len(self.train_ds)),
            "n_val": float(len(self.val_ds)),
        }

    def probe_finite(
        self, datamodule: DataModuleProtocol, *, max_steps: int = 33
    ) -> dict[str, float]:
        """Walk the packed train loader; count batches whose force Hessian is non-finite."""
        model = self.model()
        model.train()
        n_bad = 0
        n_steps = 0
        for step, batch in enumerate(datamodule.train_dataloader(), start=1):
            if step > max_steps:
                break
            batch = batch.to(self.device)
            model.zero_grad(set_to_none=True)
            out = model(batch)
            loss = train_objective(out, batch)
            energy_ok = bool(torch.isfinite(out["energy"]).all())
            forces_ok = bool(torch.isfinite(out["forces"]).all())
            loss_ok = bool(torch.isfinite(loss).all())
            if loss_ok:
                loss.backward()
                grads_ok = all(
                    param.grad is None or bool(torch.isfinite(param.grad).all())
                    for param in model.parameters()
                )
            else:
                grads_ok = False
            n_atoms = int(batch["atoms", "Z"].shape[0])
            graphs = batch.get("graphs")
            n_mols = int(graphs["num_atoms"].shape[0]) if graphs is not None else 0
            print(
                f"step={step} n_mols={n_mols} n_atoms={n_atoms} "
                f"E={int(energy_ok)} F={int(forces_ok)} L={int(loss_ok)} G={int(grads_ok)} "
                f"loss={float(loss.detach()) if loss_ok else float('nan')}",
                flush=True,
            )
            if not (energy_ok and forces_ok and loss_ok and grads_ok):
                n_bad += 1
            n_steps += 1
        return {"n_steps": float(n_steps), "n_bad": float(n_bad)}


def split_records(
    records: Sequence,
    *,
    train_frac: float = 0.8,
    seed: int = 2666,
) -> tuple[list, list]:
    """Molecule-level shuffle then ``train_frac`` / remainder split.

    All frames of one ``molecule_id`` stay on the same side.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not records:
        raise ValueError("split_records needs at least one record")
    by: dict[str, list] = {}
    order: list[str] = []
    for record in records:
        mid = _require_group(record[0])
        if mid not in by:
            by[mid] = []
            order.append(mid)
        by[mid].append(record)
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = int(round(len(order) * train_frac))
    n_train = min(max(n_train, 1), len(order) - 1)
    train_ids = order[:n_train]
    val_ids = order[n_train:]
    train = [item for mid in train_ids for item in by[mid]]
    val = [item for mid in val_ids for item in by[mid]]
    return train, val


def _record_n_atoms(record: Sequence) -> int:
    atoms = record[0]["atoms"]
    if "number" in atoms:
        return int(len(atoms["number"]))
    return int(len(atoms["x"]))


def _record_z(record: Sequence) -> set[int]:
    atoms = record[0]["atoms"]
    if "number" not in atoms:
        return set()
    return {int(z) for z in atoms["number"]}


def split_coverage(train: Sequence, val: Sequence) -> dict[str, Any]:
    """Molecule-level coverage. Refuses an overlapping split."""
    train_ids = [_require_group(rec[0]) for rec in train]
    val_ids = [_require_group(rec[0]) for rec in val]
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val molecule_id overlap: {sorted(overlap)[:8]}")
    train_n = [_record_n_atoms(rec) for rec in train]
    val_n = [_record_n_atoms(rec) for rec in val]
    train_z: set[int] = set()
    val_z: set[int] = set()
    for rec in train:
        train_z |= _record_z(rec)
    for rec in val:
        val_z |= _record_z(rec)
    return {
        "n_train_frames": len(train),
        "n_val_frames": len(val),
        "n_train_molecules": len(set(train_ids)),
        "n_val_molecules": len(set(val_ids)),
        "train_n_atoms_min": min(train_n) if train_n else 0,
        "train_n_atoms_max": max(train_n) if train_n else 0,
        "val_n_atoms_min": min(val_n) if val_n else 0,
        "val_n_atoms_max": max(val_n) if val_n else 0,
        "train_Z": sorted(train_z),
        "val_Z": sorted(val_z),
        "overlap": 0,
    }


def write_split_manifest(path: Path, train: Sequence, val: Sequence) -> Path:
    """Write molecule ids for each split. Coverage is included."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "train_ids": [_require_group(rec[0]) for rec in train],
        "val_ids": [_require_group(rec[0]) for rec in val],
        "coverage": split_coverage(train, val),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_omol25_records(
    *,
    packed: str | Path | None = None,
    role: str = "val-neutral",
    max_atoms: int = 32,
    n_keep: int = 0,
) -> list:
    """Load OMol25-small records. ``packed`` is the offline test path."""
    from molhub.dataset.omol25 import OMol25, OMol25Dataset, OMol25Small

    if packed:
        source = OMol25Dataset(packed)
    else:
        # Scan extra rows so the small-molecule filter can still fill *n_keep*.
        scan_cap = None if n_keep <= 0 else max(int(n_keep) * 16, 64)
        source = OMol25(role=role, max_rows=scan_cap).load()
    small = OMol25Small(source, max_atoms=max_atoms, limit=n_keep if n_keep > 0 else None)
    small.load()
    records = [small[i] for i in range(len(small))]
    if not records:
        raise ValueError("OMol25-small kept no frames")
    return records


def load_omol25_split(
    *,
    packed: str | Path | None = None,
    role: str = "val-neutral",
    max_atoms: int = 32,
    n_keep: int = 0,
    train_frac: float = 0.8,
    seed: int = 2666,
) -> tuple[list, list]:
    """OMol25-small, molecule-level ``train_frac`` split."""
    raw = load_omol25_records(packed=packed, role=role, max_atoms=max_atoms, n_keep=n_keep)
    return split_records(raw, train_frac=train_frac, seed=seed)


def _raw_omol_record(source: Any, idx: int) -> dict[str, Any]:
    """LMDB dict via public ``record()``; Frame-shaped fallback for tests."""
    record_fn = getattr(source, "record", None)
    if callable(record_fn):
        rec = dict(record_fn(idx))
        rec.setdefault("energy_is_ev", True)
        return rec
    frame = source[idx][0]
    atoms = frame["atoms"]
    meta = getattr(frame, "meta", None) or {}
    energy = meta.get("energy", 0.0)
    return {
        "numbers": np.asarray(atoms["number"], dtype=np.int64),
        "energy": float(getattr(energy, "value", energy)),
        "energy_is_ev": True,
    }


def _counts_cache_path(source: Any) -> Path:
    """On-disk path for *source*'s per-row atom-count vector.

    Keyed like :func:`_e0_cache_path` — identity plus row count — and written
    under the same ``derived/omol25-index`` directory. The same caveat applies:
    a source that changes content without changing length or identity would
    read a stale vector.

    Args:
        source: Loaded MolHub view.

    Returns:
        ``home_dir / "derived" / "omol25-index" / {digest}-n_atoms.npy``.
    """
    home = Path(molhub_config.home_dir)
    root = str(getattr(source, "root", getattr(source, "source_id", type(source).__name__)))
    digest = hashlib.sha1(f"{root}:{len(source)}".encode()).hexdigest()[:16]
    path = home / "derived" / "omol25-index" / f"{digest}-n_atoms.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _official_vector_counts(source: Any) -> np.ndarray:
    """Per-row atom counts from ``source.get_distrib("graph.natoms")``."""
    return np.asarray(source.get_distrib("graph.natoms").raw)


def _measured_counts(source: Any) -> np.ndarray:
    """:func:`_official_vector_counts`, measured once and kept on disk.

    ``get_distrib`` decodes **every** row into a molpy ``Frame`` to read one
    scalar from it. On official 4m that is 3.99 M decodes: measured at 1632 s
    on a GPU node with the ε[Z] cache already warm (Slurm 2528891), and paid
    again at every process start, before step 1 — 27 minutes of a 24 h run
    spent re-measuring a constant. Cached, the same phase takes 0.3 s.

    The measurement itself stays in :func:`_official_vector_counts`, which is
    deliberately a bare call: this wrapper adds storage, never discovery, and
    never a fabricated count.
    """
    identity = str(getattr(source, "source_id", ""))
    if not identity.startswith("dataset:"):
        # Only a versioned MolHub coordinate pins content, and the key is
        # identity plus row count. Anything else — a local view, a test
        # double — can hold different rows under one name, and a cache keyed
        # on it would hand back another source's counts.
        return _official_vector_counts(source)
    path = _counts_cache_path(source)
    n_rows = len(source)
    cached = _stored_counts(path, n_rows)
    if cached is not None:
        return cached
    counts = _official_vector_counts(source)
    np.save(path, counts)
    return counts


def _stored_counts(path: Path, n_rows: int) -> np.ndarray | None:
    """A cached count vector, or ``None`` when the file cannot be trusted.

    Refuses anything that is not a readable integer vector of the right
    length. The 2026-09-11 files under this name are a raw ``int32`` dump of
    the constant 512 — a fabricated fallback from a deleted code path, with no
    ``.npy`` header at all — so "a file is there" is not evidence that the
    counts in it were ever measured. A rejected file is re-measured and
    overwritten.
    """
    if not path.is_file():
        return None
    try:
        cached = np.load(path)
    except (ValueError, OSError, EOFError):
        return None
    if cached.shape != (n_rows,) or not np.issubdtype(cached.dtype, np.integer):
        return None
    return cached


def _e0_cache_path(source: Any, *, max_atoms: int, n_fit: int, seed: int) -> Path:
    """On-disk ε[Z] cache path under ``molhub_config.home_dir``.

    Args:
        source: Loaded MolHub view. Identity is ``root`` (or ``source_id``)
            plus ``len(source)``.
        max_atoms: Keep bound used when the subsample was chosen (atoms).
        n_fit: Number of rows in the least-squares subsample (dimensionless).
        seed: RNG seed of that subsample (dimensionless).

    Returns:
        ``home_dir / "derived" / "omol25-index" / {digest}-e0.npy`` where
        *digest* is the sha1[:16] of ``root:len:max_atoms:n_fit:seed``.
        *home_dir* is a pathlib Path, not a physical unit.
    """
    home = Path(molhub_config.home_dir)
    root = str(getattr(source, "root", getattr(source, "source_id", type(source).__name__)))
    digest = hashlib.sha1(f"{root}:{len(source)}:{max_atoms}:{n_fit}:{seed}".encode()).hexdigest()[
        :16
    ]
    path = home / "derived" / "omol25-index" / f"{digest}-e0.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fit_e0_subsample(
    source: Any,
    indices: Sequence[int],
    *,
    n_fit: int = 8192,
    seed: int = 2666,
    max_atoms: int = 0,
) -> np.ndarray:
    """Least-squares ε[Z] on a small random subset, cached on disk.

    8192 samples are ample for a ~10-element least-squares table; the old
    32768-sample pass spent most of ``load_omol25_official`` re-reading
    random LMDB records (≈18 ms/record cold). The result is cached per
    source + ``max_atoms`` so a resubmitted smoke/prod run skips the reads.
    """
    cache_path = _e0_cache_path(source, max_atoms=int(max_atoms), n_fit=n_fit, seed=seed)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape == (95,):
            return cached.astype(np.float64, copy=False)

    factor = _EV_PER_A_TO_KCAL_PER_A
    pick = list(indices)
    if len(pick) > n_fit:
        rng = random.Random(seed)
        pick = rng.sample(pick, n_fit)
    size = 95
    ata = np.zeros((size, size), dtype=np.float64)
    atb = np.zeros(size, dtype=np.float64)
    for idx in pick:
        rec = _raw_omol_record(source, int(idx))
        numbers = np.asarray(rec["numbers"], dtype=np.int64).reshape(-1)
        energy = float(rec["energy"])
        if rec.get("energy_is_ev", True):
            energy *= factor
        hist = np.bincount(np.clip(numbers, 0, size - 1), minlength=size).astype(np.float64)[:size]
        ata += np.outer(hist, hist)
        atb += hist * energy
    e0 = np.linalg.lstsq(ata, atb, rcond=None)[0]
    np.save(cache_path, e0)
    return e0


def _index_omol_source(
    source: Any, *, max_atoms: int, fit_e0: bool = True
) -> tuple[list[int], np.ndarray]:
    """Keep rows with ``1 <= n_atoms <= max_atoms``; optionally fit ε[Z].

    The count vector is fully known, so the keep is exact. A measured
    zero is dropped (``n >= 1``); negatives are dropped by the same
    predicate.

    Args:
        source: Loaded MolHub view with ``get_distrib("graph.natoms")``.
        max_atoms: Inclusive keep bound (atoms).
        fit_e0: When True, least-squares ε[Z] on a subsample of the keep.

    Returns:
        ``(keep, e0)`` — kept source-row indices and the 95-element ε
        table in kcal mol⁻¹ (zeros when *fit_e0* is False or keep is empty).
    """
    n_atoms = _measured_counts(source)
    keep = np.flatnonzero((n_atoms >= 1) & (n_atoms <= int(max_atoms)))
    if keep.size == 0:
        return [], np.zeros(95, dtype=np.float64)
    e0 = (
        _fit_e0_subsample(source, keep.tolist(), max_atoms=int(max_atoms))
        if fit_e0
        else np.zeros(95, dtype=np.float64)
    )
    return keep.tolist(), e0


class MoleculeSplit:
    """Train/val index views of one source, plus the optional ε[Z] table.

    The datasets are :class:`~molhub.dataset.protocol.SubsetDataset` over
    whatever the loader opened — OMol25's LMDB, a THEMol shard. Dressing is a
    model table, not a dataset attribute, which is why it rides along here
    rather than on either dataset.
    """

    def __init__(
        self,
        train: SubsetDataset,
        val: SubsetDataset,
        *,
        dressing: AtomicReferenceEnergy | None = None,
    ) -> None:
        self.train = train
        self.val = val
        self.dressing = dressing


def load_omol25_official(
    *,
    train_role: str = "4m",
    val_role: str = "val-neutral",
    max_atoms: int = 16384,
) -> MoleculeSplit:
    """Official OMol25: ``4m`` train, ``val-neutral`` val. Frames stay eV, no bonds."""
    t0 = perf_counter()
    train_src = _official_source(train_role)
    print(
        f"omol train_role={train_role} n={len(train_src)} load_s={perf_counter() - t0:.1f}",
        flush=True,
    )
    t0 = perf_counter()
    val_src = _official_source(val_role)
    print(
        f"omol val_role={val_role} n={len(val_src)} load_s={perf_counter() - t0:.1f}",
        flush=True,
    )
    t0 = perf_counter()
    train_keep, e0 = _index_omol_source(train_src, max_atoms=max_atoms)
    print(f"omol train index_s={perf_counter() - t0:.1f}", flush=True)
    t0 = perf_counter()
    val_keep, _ = _index_omol_source(val_src, max_atoms=max_atoms, fit_e0=False)
    print(f"omol val index_s={perf_counter() - t0:.1f}", flush=True)
    if not train_keep or not val_keep:
        raise ValueError("OMol25 official split kept no frames")
    train = SubsetDataset(train_src, train_keep)
    val = SubsetDataset(val_src, val_keep)
    print(f"omol kept train={len(train)} val={len(val)}", flush=True)
    return MoleculeSplit(train, val, dressing=_historical_are(e0.tolist()))


_OFFICIAL_SOURCE_CACHE: dict[str, Any] = {}


def _official_source(role: str) -> Any:
    """Loaded ``OMol25`` per role, cached so one process opens each LMDB once.

    ``OMol25.load`` is idempotent per instance but opens a new LMDB env per
    instance; the calibrate gate calls the tiny loader several times in one
    process, and LMDB refuses a second env on the same path.
    """
    from molhub.dataset.omol25 import OMol25

    cached = _OFFICIAL_SOURCE_CACHE.get(role)
    if cached is None:
        cached = OMol25(role=role).load()
        _OFFICIAL_SOURCE_CACHE[role] = cached
    return cached


def _official_prefix(n_atoms: np.ndarray, n: int, max_atoms: int, scan_cap: int) -> list[int]:
    """First *n* source rows with ``1 <= n_atoms <= max_atoms``.

    Operates on an already-loaded numpy count vector; does not decode
    LMDB records.

    Args:
        n_atoms: Per-row atom counts (dimensionless, atoms per row).
        n: Maximum number of kept rows (dimensionless).
        max_atoms: Inclusive keep bound (atoms).
        scan_cap: Maximum prefix of *n_atoms* to scan (dimensionless).

    Returns:
        Source row indices of the first *n* rows that pass the predicate.
    """
    limit = min(int(n_atoms.shape[0]), int(scan_cap))
    keep: list[int] = []
    for i in range(limit):
        if len(keep) >= n:
            break
        size = int(n_atoms[i])
        if 1 <= size <= int(max_atoms):
            keep.append(i)
    return keep


def load_omol25_official_tiny(
    *,
    n_train: int = 48,
    n_val: int = 12,
    max_atoms: int = 32,
    seed: int = 2666,
    scan_cap: int = 8192,
) -> MoleculeSplit:
    """Official ``4m`` / ``val-neutral`` coordinates on a bounded real subset.

    For the align / inspect / overfit calibration gates: same coordinate
    identity as production, but only the first ``n_train`` / ``n_val`` rows
    with ``1 ≤ n_atoms ≤ max_atoms`` atoms are kept. ε[Z] is fitted on that
    tiny subset. The molecule count is an explicitly enumerated
    calibration-only difference.
    """
    train_src = _official_source("4m")
    val_src = _official_source("val-neutral")
    train_counts = _measured_counts(train_src)
    val_counts = _measured_counts(val_src)
    train_keep = _official_prefix(train_counts, n_train, max_atoms, scan_cap)
    val_keep = _official_prefix(val_counts, n_val, max_atoms, scan_cap)
    if not train_keep or not val_keep:
        raise ValueError(
            "official tiny subset kept no frames "
            f"(n_train={len(train_keep)}, n_val={len(val_keep)})"
        )
    e0 = _fit_e0_subsample(
        train_src, train_keep, n_fit=len(train_keep), max_atoms=max_atoms, seed=seed
    )
    return MoleculeSplit(
        SubsetDataset(train_src, train_keep),
        SubsetDataset(val_src, val_keep),
        dressing=_historical_are(e0.tolist()),
    )


_FAIRCHEM_EVAL_ROLES: tuple[tuple[str, str], ...] = (
    ("general", ""),
    ("metal", "val_metal_complexes"),
    ("electrolytes", "val_electrolytes"),
    ("biomolecules", "val_biomolecules"),
    ("spice", "val_spice"),
    ("neutral_organics", "val_neutralorganics"),
)


def _omol_elem_refs_path() -> Path:
    """Hosted ``omol-elem-refs`` YAML path. Fail-closed if the artifact is absent."""
    import molhub
    from molhub.dataset.omol25 import COORDINATE

    paths = molhub.fetch(COORDINATE, roles=["omol-elem-refs"])
    path = Path(paths["omol-elem-refs"])
    if not path.is_file():
        raise FileNotFoundError(f"omol-elem-refs artifact is absent: {path}")
    return path


def load_omol25_fairchem_4m() -> tuple[Any, dict[str, Any]]:
    """Official eV 4M train plus named official-val / domain views.

    Fail-closes if ``omol-elem-refs`` is absent. Does not construct
    :class:`AtomicReferenceEnergy` and does not filter by ``batch_max_atoms``.
    Train and ``general`` val are the OMol25 sources; named domain views
    are :class:`~molhub.dataset.protocol.SubsetDataset`.
    """
    _omol_elem_refs_path()
    train_src = _official_source("4m")
    val_src = _official_source("val")
    assignment = val_src.assignment()
    roles: dict[str, Any] = {}
    for name, key in _FAIRCHEM_EVAL_ROLES:
        if name == "general":
            roles[name] = val_src
        else:
            roles[name] = SubsetDataset(val_src, list(assignment[key]))
    return train_src, roles


def load_omol25_fairchem_4m_tiny(
    *,
    n_train: int = 48,
    n_val: int = 12,
) -> tuple[SubsetDataset, dict[str, SubsetDataset]]:
    """Bounded prefix of :func:`load_omol25_fairchem_4m` roles, still eV."""
    train, roles = load_omol25_fairchem_4m()
    train_view = SubsetDataset(train, list(range(min(n_train, len(train)))))
    role_views = {
        name: SubsetDataset(view, list(range(min(n_val, len(view)))))
        for name, view in roles.items()
    }
    return train_view, role_views


def load_phalkethoh_val(root: str | Path, *, n_keep: int = 0) -> list:
    """Load the published PhAlkEthOH val split. Never used as train."""
    from molhub.dataset.phalkethoh_mm import PhalkethohMMDataset

    ds = PhalkethohMMDataset(
        root, download=False, split="val", max_frames=n_keep if n_keep > 0 else None
    )
    records = [ds[i] for i in range(len(ds))]
    if not records:
        raise ValueError("PhAlkEthOH val split is empty")
    return records


def _themol_e0_path(source_id: str, *, n_fit: int, seed: int, n_rows: int) -> Path:
    """On-disk path for a THEMol ε[Z] table.

    Keyed by the source, the number of train rows it was fitted over, and the
    subsample ``(n_fit, seed)`` that selected them — every input that changes the
    table, and nothing else.

    Why it is cached at all: the fit reads *n_fit* whole records, and a record is
    24 frames, so the default 4 096 records is ~98 000 frame decodes — measured
    160 s on a CPU node (job 2724471). The calibration gate runs one
    **subprocess per cell** (nine of them, deliberately, because
    ``error_on_recompile`` is process-global), and each one calls this loader, so
    the same table was fitted nine times: ~24 minutes of a three-hour job
    (2724066, cancelled for this). A cached read is 0.0 s. The OMol path had
    :func:`_e0_cache_path` for exactly this reason; this is its counterpart.

    Args:
        source_id: The loaded shard's provenance string.
        n_fit: Records the least-squares reads.
        seed: Subsample seed.
        n_rows: Train rows the subsample was drawn from.

    Returns:
        ``home_dir / "derived" / "themol-e0" / {digest}.npy``.
    """
    digest = hashlib.sha1(
        f"{source_id}:n_rows={n_rows}:n_fit={n_fit}:seed={seed}".encode()
    ).hexdigest()[:16]
    path = Path(molhub_config.home_dir) / "derived" / "themol-e0" / f"{digest}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_themol_split(
    definition: Any,
    *,
    role: str = "shard-000",
    val_frac: float = 0.05,
    seed: int = 2666,
    n_train: int = 0,
    n_val: int = 0,
    n_fit: int = 4096,
    dressing: bool = True,
) -> MoleculeSplit:
    """One THEMol shard, split by molecule. Frames are kcal/mol, with bonds.

    Serves every stage of the ByteFF curriculum (spec `rondo.md` §5: staged
    partial-Hessian -> torsion profiles -> off-equilibrium E/F) plus the MBIS
    charge set, because all five THEMol definitions load the same
    :class:`~molhub.dataset.themol.THEMolShardDataset` and differ only in what
    their reader puts in the Frame.

    **The split unit is the molecule, never the row.** A row is whatever that
    dataset's reader calls a record — one equilibrium geometry for
    ``hessian``, one whole 24-conformer scan for ``torsion-scan`` — and a
    molecule contributes as many rows as it has scanned dihedrals or sampled
    geometries. On torsion-scan a row-wise shuffle puts 1 619 of 83 951 scans'
    molecules on both sides (measured 2026-09-20: 82 332 distinct molecules,
    1.02 scans/molecule). Whole molecules are held out through
    :meth:`~molhub.dataset.themol.THEMolShardDataset.molecule_index`.

    Why THEMol and not OMol25: it ships the topology. OMol25 frames carry no
    bonds, so the Class-I path inferred them per molecule inside collate at
    379 us/molecule, which is what made that step CPU-bound. THEMol records
    carry a mapped SMILES, and the energies are already kcal/mol — the unit
    every Class-I term is written in.

    Args:
        definition: The THEMol definition class to load, e.g.
            ``THEMolTorsionScan`` or ``THEMolHessian``. Passed as the class, not
            a name, so a typo is an import error rather than a runtime one.
        role: Shard role.
        val_frac: Fraction of *molecules* (not rows) held out.
        seed: Permutation seed for the molecule shuffle.
        n_train: Cap on kept train rows. ``0`` keeps all.
        n_val: Cap on kept val rows. ``0`` keeps all.
        n_fit: Records the eps[Z] least-squares reads. The table has ~10 free
            parameters, so a few thousand records is ample and the fit is not
            what the loader should spend its time on.
        dressing: Fit an eps[Z] table. ``False`` for a set whose records carry
            no energy — MBIS declares an empty ``graph_level`` schema, so a fit
            there would read no energies, return an all-zero table, and then be
            refused by the element-coverage check for being all zero. A set with
            no energy needs no dressing, and saying so is clearer than a table
            of zeros.

    Returns:
        Train and val :class:`~molhub.dataset.protocol.SubsetDataset` views of
        the one shard, plus the eps[Z] table fitted on the train side only (or
        ``None`` when *dressing* is false).

    Raises:
        ValueError: *val_frac* leaves either side empty.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    name = getattr(definition, "COORDINATE", type(definition).__name__)
    t0 = perf_counter()
    source = definition().load(role)
    print(
        f"themol {name} role={role} n_rows={len(source)} load_s={perf_counter() - t0:.1f}",
        flush=True,
    )

    t0 = perf_counter()
    groups = source.molecule_index()
    n_molecules = int(groups.max()) + 1 if groups.size else 0
    print(
        f"themol molecules={n_molecules} rows_per_molecule="
        f"{len(source) / max(n_molecules, 1):.2f} index_s={perf_counter() - t0:.1f}",
        flush=True,
    )

    # Hold out whole molecules: permute the molecule ids, take a prefix for val.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_molecules)
    n_val_groups = int(round(n_molecules * val_frac))
    n_val_groups = min(max(n_val_groups, 1), n_molecules - 1)
    is_val = np.zeros(n_molecules, dtype=bool)
    is_val[order[:n_val_groups]] = True

    val_rows = np.flatnonzero(is_val[groups]).tolist()
    train_rows = np.flatnonzero(~is_val[groups]).tolist()
    if not train_rows or not val_rows:
        raise ValueError(
            f"val_frac={val_frac} over {n_molecules} molecules left "
            f"train={len(train_rows)} val={len(val_rows)} rows"
        )
    # eps[Z] is fitted from the **whole** train side, before n_train/n_val cut a
    # view out of it. Those two only bound how much a job looks at; the dressing
    # is a property of the dataset and the split, so a bounded job gets the same
    # table production does — one cached table for all of them.
    #
    # This is not a refinement. Fitting from a *truncated* train side is what
    # broke smoke 2723274: its 80 rows held no iodine, the held-out molecules
    # did, so eps[53] stayed 0 and those molecules kept their ~1e6 kcal/mol
    # offset inside what the loss reads as a residual — val_mae_kcal 231 651 on a
    # path that was working, with a median residual of 35.6 hiding under the mean.
    table = _themol_dressing(source, role, train_rows, n_fit=n_fit, seed=seed) if dressing else None

    if n_train > 0:
        train_rows = train_rows[:n_train]
    if n_val > 0:
        val_rows = val_rows[:n_val]

    train = SubsetDataset(source, train_rows)
    val = SubsetDataset(source, val_rows)
    print(
        f"themol kept train={len(train)} val={len(val)} val_molecules={n_val_groups}/{n_molecules}",
        flush=True,
    )

    if table is not None:
        _require_element_coverage(source, table, train_rows, val_rows)
    return MoleculeSplit(train, val, dressing=table)


def _themol_dressing(
    source: Any,
    role: str,
    train_rows: Sequence[int],
    *,
    n_fit: int,
    seed: int,
) -> AtomicReferenceEnergy:
    """The ε[Z] table for this split's train side, read from disk when it is there.

    THEMol energies are already kcal/mol, so there is no eV factor here — the
    OMol path needs one, this does not.

    Args:
        source: The loaded shard.
        role: Shard role, used only to name the cache when the shard reports no
            ``source_id``.
        train_rows: The **whole** train side, never a bounded view of it.
        n_fit: Records the least-squares reads.
        seed: Subsample seed.

    Returns:
        The fitted (or cached) table.
    """
    t0 = perf_counter()
    cache = _themol_e0_path(
        str(getattr(source, "source_id", role)),
        n_fit=n_fit,
        seed=seed,
        n_rows=len(train_rows),
    )
    if cache.is_file():
        print(f"themol e0 cached={cache.name} read_s={perf_counter() - t0:.1f}", flush=True)
        return _historical_are(np.load(cache).tolist())
    rows = list(train_rows)
    pick = rows if len(rows) <= n_fit else random.Random(seed).sample(rows, n_fit)
    dressing = fit_atomic_reference([source[int(row)] for row in pick])
    np.save(cache, np.asarray(dressing.atomic_energies, dtype=np.float64))
    print(
        f"themol e0 n_fit={len(pick)} fit_s={perf_counter() - t0:.1f} wrote={cache.name}",
        flush=True,
    )
    return dressing


def _require_element_coverage(
    source: Any,
    dressing: AtomicReferenceEnergy,
    train_rows: Sequence[int],
    val_rows: Sequence[int],
) -> None:
    """Refuse a dressing table with a zero where a split actually needs a value.

    :func:`fit_atomic_reference` solves ``counts @ eps = E_DFT`` over the records
    it is given and leaves ``eps = 0`` for every element none of them contained.
    That zero is not neutral: a molecule holding such an element keeps its whole
    absolute-energy offset inside what the loss reads as a residual.

    Measured 2026-09-20 (job 2723274): fitted on 80 records, the table covered
    ``[1, 5, 6, 7, 8, 9, 14, 16, 17, 35]`` while the held-out molecules carried
    iodine, and the validation side went to a **mean** ``|E_DFT - e0|`` of
    217 162 kcal/mol against a **median** of 35.6 — the median says the fit is
    fine, the mean says a handful of molecules are off by 1e6, and the smoke
    duly reported ``val_mae_kcal`` 231 651 for a path that was working. At the
    calibration sample of 4 096 records every element is covered and train and
    val agree (median 52.9 vs 56.9).

    Element sets come from
    :meth:`~molhub.dataset.themol.THEMolShardDataset.element_mask` — one cached
    integer per row — so this costs a bitwise OR, not a record decode.

    Args:
        source: The loaded shard both splits index into.
        dressing: The fitted table.
        train_rows: Source rows on the train side.
        val_rows: Source rows on the val side.

    Raises:
        ValueError: An element either split needs has no fitted value.
    """
    masks = source.element_mask()
    fitted = np.flatnonzero(np.asarray(dressing.atomic_energies, dtype=np.float64))
    fitted_mask = 0
    for z in fitted.tolist():
        fitted_mask |= 1 << int(z)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        needed = int(np.bitwise_or.reduce(masks[np.asarray(rows, dtype=np.int64)]))
        gap = needed & ~fitted_mask
        if gap:
            missing = [z for z in range(64) if gap >> z & 1]
            raise ValueError(
                f"the epsilon[Z] table has no value for element(s) {missing} that the "
                f"{name} split needs: the fit sample never saw them, so their entry is "
                f"0 and every molecule carrying one keeps its whole ~1e6 kcal/mol "
                f"absolute-energy offset in the residual. Raise n_fit (currently the "
                f"loader's default) or widen the fit sample."
            )
