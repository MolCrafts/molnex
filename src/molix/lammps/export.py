"""Export a molnex potential for the generic ``pair_style molnex`` LAMMPS interface.

:func:`export_for_lammps` wraps a model in a flat-convention :class:`LammpsForward`
(see :mod:`molix.lammps.adapter`), AOT-compiles it via :func:`molix.export.export_model`,
then augments the export's ``meta.json`` with a ``lammps`` block carrying everything
the C++ ``pair_style molnex`` needs to be model-agnostic:

* ``cutoff``        — neighbour cutoff in Å (the C++ side no longer takes it on the command line)
* ``units``         — the model's native unit system (``"real"`` = kcal/mol, ``"metal"`` = eV);
                      ``pair_style molnex`` reads LAMMPS' active ``units`` and converts.
* ``species``       — sorted atomic numbers ``Z`` the model supports
* ``model_dtype``   — ``"float32"`` / ``"float64"``; the pair style casts positions to this
* ``inputs`` / ``outputs`` — the flat tensor names/order (provenance + future schema checks)
* ``supports_pbc`` / ``supports_virial`` — capability flags (reserved; this cut is non-periodic)
* ``adapter``       — the registered adapter name used at export time

Because MD changes the edge count ``E`` (and possibly atom count ``N``) every step,
the export marks those dimensions dynamic so a single ``.so`` serves any frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod
from molix.export import export_model

from .adapter import LammpsAdapter, LammpsForward

logger = _logger_mod.getLogger(__name__)

#: meta.json ``lammps`` block schema version — bump on incompatible field changes.
LAMMPS_META_SCHEMA = 1

#: unit systems the pair style understands (LAMMPS names → energy unit label)
_UNIT_LABELS = {"real": "kcal/mol", "metal": "eV"}

_DTYPE_NAMES = {torch.float32: "float32", torch.float64: "float64"}


def _example_system(species: list[int], cutoff: float, dtype: torch.dtype, device: str):
    """A tiny but representative frame for tracing: two atoms within the cutoff.

    Uses the first two declared species (or one atom duplicated) so the traced
    graph exercises both the node and edge paths. Concrete shapes are irrelevant
    — N and E are marked dynamic at export — but the values must form a valid,
    in-cutoff graph so the forward doesn't hit an empty-edge degenerate path.
    """
    z0 = species[0]
    z1 = species[1] if len(species) > 1 else species[0]
    Z = torch.tensor([z0, z1], dtype=torch.long, device=device)
    r = min(0.5 * cutoff, max(cutoff - 0.5, 0.3))             # comfortably inside the cutoff
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype, device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    return Z, pos, edge_index


def export_for_lammps(
    model: nn.Module,
    export_dir: str | Path,
    *,
    species: list[int],
    cutoff: float,
    units: str = "real",
    adapter: LammpsAdapter | str = "molnex-tensordict",
    device: str = "auto",
    dtype: torch.dtype | None = None,
    name: str = "model",
    supports_pbc: bool = False,
    supports_virial: bool = False,
) -> Path:
    """Export ``model`` to an AOTI ``.so`` consumable by ``pair_style molnex <dir>``.

    Args:
        model: A potential. With the default ``adapter`` it must accept a nested
            ``TensorDict`` and return ``{"energy", "forces"}`` (e.g. ``PiNetPotential``).
        export_dir: Output directory (created if missing); receives ``{name}.so``,
            ``{name}.pt``, ``{name}.meta.json``.
        species: Atomic numbers the model supports. The order maps to LAMMPS atom
            types in the user's ``pair_coeff * * <Z...>``; stored sorted in meta.
        cutoff: Neighbour cutoff in Å (typically ``model.encoder.config.r_max``).
        units: The model's native unit system — ``"real"`` (kcal/mol, kcal/mol/Å)
            or ``"metal"`` (eV, eV/Å). The pair style converts to LAMMPS' units.
        adapter: Registered adapter name or instance translating the flat
            ``(Z, pos, edge_index)`` convention onto the model. Default builds the
            molnex nested ``TensorDict``.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"`` — forwarded to ``export_model``.
        dtype: Compute dtype for the exported model (default: the model's current
            parameter dtype, falling back to ``float32``).
        name: Artifact basename.
        supports_pbc: Reserved capability flag (this cut is non-periodic).
        supports_virial: Reserved capability flag (model emits a 3rd virial output).

    Returns:
        The export directory as a :class:`Path`.

    Raises:
        ValueError: If ``species`` is empty or ``units`` is not understood.
    """
    if not species:
        raise ValueError("species must be a non-empty list of atomic numbers")
    if units not in _UNIT_LABELS:
        raise ValueError(f"units must be one of {sorted(_UNIT_LABELS)}, got {units!r}")

    resolved_dtype = dtype or _infer_dtype(model)
    wrapper = LammpsForward(model, adapter)
    adapter_name = wrapper.adapter.name

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Z, pos, edge_index = _example_system(sorted(set(species)), cutoff, resolved_dtype, device)

    # N (atoms) and E (edges) vary across MD frames → mark dynamic so one .so
    # serves every step. Z and pos share the atom dim; edge_index has its own.
    n_dim = torch.export.Dim("n_atoms", min=1, max=1 << 20)
    e_dim = torch.export.Dim("n_edges", min=1, max=1 << 24)
    dynamic_shapes = (
        {0: n_dim},              # Z (N,)
        {0: n_dim},              # pos (N, 3)
        {0: e_dim},              # edge_index (E, 2)
    )

    export_model(
        wrapper,
        (Z, pos, edge_index),
        export_dir,
        device=device,
        name=name,
        dynamic_shapes=dynamic_shapes,
    )

    export_dir = Path(export_dir)
    meta_path = export_dir / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["lammps"] = {
        "schema_version": LAMMPS_META_SCHEMA,
        "cutoff": float(cutoff),
        "units": units,
        "energy_unit": _UNIT_LABELS[units],
        "species": sorted(set(int(z) for z in species)),
        "model_dtype": _DTYPE_NAMES.get(resolved_dtype, str(resolved_dtype).split(".")[-1]),
        "inputs": ["Z", "pos", "edge_index"],
        "outputs": ["energy", "forces"],
        "adapter": adapter_name,
        "supports_pbc": bool(supports_pbc),
        "supports_virial": bool(supports_virial),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    logger.info(
        f"Exported LAMMPS model to {export_dir} "
        f"(adapter={adapter_name}, units={units}, cutoff={cutoff} A, "
        f"dtype={meta['lammps']['model_dtype']}, species={meta['lammps']['species']})"
    )
    return export_dir


def _infer_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32
