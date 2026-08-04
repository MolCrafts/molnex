"""Export a molnex potential for the generic ``pair_style molnex`` LAMMPS interface.

:func:`export_for_lammps` wraps a model in a flat-convention :class:`EngineForward`
(see :mod:`molix.engine.adapter`), AOT-compiles it via :class:`molix.export.Exporter`
into a single ``{name}.pt2`` package, then writes a sibling ``{name}.meta.json`` with a
``lammps`` block carrying everything the C++ ``pair_style molnex`` needs to be
model-agnostic:

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
from molix.export import Exporter

from .adapter import EngineAdapter, EngineForward

logger = _logger_mod.getLogger(__name__)

#: meta.json ``lammps`` block schema version — bump on incompatible field changes.
LAMMPS_META_SCHEMA = 1

#: unit systems the pair style understands (LAMMPS names → energy unit label)
_UNIT_LABELS = {"real": "kcal/mol", "metal": "eV"}

_DTYPE_NAMES = {torch.float32: "float32", torch.float64: "float64"}


def _example_system(species: list[int], cutoff: float, dtype: torch.dtype, device: str):
    """A small dense cluster covering every species, for tracing.

    NOT two atoms: ``make_fx`` symbolic tracing (the force-export path) bakes a
    wrong specialization on a degenerate 2-atom graph that fails at larger N.
    A denser cluster (≥ 8 atoms, every pair within cutoff so the neighbour graph
    is non-trivial) traces a graph that ``dynamic_shapes`` then generalize to any
    N/E. Concrete sizes are placeholders — N and E are marked dynamic at export.
    """
    n_ex = max(8, 2 * len(species))
    gen = torch.Generator().manual_seed(0)
    # box small enough that the cube diagonal stays inside the cutoff → every
    # pair is an edge (a fully-connected, non-degenerate example graph).
    box = 0.45 * cutoff
    pos = ((torch.rand(n_ex, 3, generator=gen, dtype=torch.float64) - 0.5) * box).to(
        dtype=dtype, device=device
    )
    Z = torch.tensor(
        [species[i % len(species)] for i in range(n_ex)], dtype=torch.long, device=device
    )
    dist = torch.cdist(pos.double(), pos.double())
    # Contiguous row-major (E, 2) — the exact layout pair_style molnex builds
    # at runtime. AOTInductor fixes the input stride from this trace example at
    # the graph boundary, so it MUST match the deployed caller's layout (a
    # non-contiguous example silently scrambles source/target in C++).
    edge_index = ((dist < cutoff) & (dist > 0)).nonzero().to(torch.long).contiguous()
    return Z, pos, edge_index


def export_for_lammps(
    model: nn.Module,
    export_dir: str | Path,
    *,
    species: list[int],
    cutoff: float,
    units: str = "real",
    adapter: EngineAdapter | str = "molnex-tensordict",
    device: str = "auto",
    dtype: torch.dtype | None = None,
    name: str = "model",
    supports_pbc: bool = False,
    supports_virial: bool = False,
    cuda_graph: bool = False,
    n_atoms: int | None = None,
    e_max: int | None = None,
) -> Path:
    """Export ``model`` to an AOTI ``.pt2`` consumable by ``pair_style molnex <dir>``.

    Args:
        model: A potential. With the default ``adapter`` it must accept a nested
            ``TensorDict`` and return ``{"energy", "forces"}`` (e.g. ``PiNetPotential``).
        export_dir: Output directory (created if missing); receives ``{name}.pt2``
            and ``{name}.meta.json``.
        species: Atomic numbers the model supports. The order maps to LAMMPS atom
            types in the user's ``pair_coeff * * <Z...>``; stored sorted in meta.
        cutoff: Neighbour cutoff in Å (typically ``model.encoder.config.r_max``).
        units: The model's native unit system — ``"real"`` (kcal/mol, kcal/mol/Å)
            or ``"metal"`` (eV, eV/Å). The pair style converts to LAMMPS' units.
        adapter: Registered adapter name or instance translating the flat
            ``(Z, pos, edge_index)`` convention onto the model. Default builds the
            molnex nested ``TensorDict``.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"`` — selects the export target.
        dtype: Compute dtype for the exported model (default: the model's current
            parameter dtype, falling back to ``float32``).
        name: Artifact basename.
        supports_pbc: Reserved capability flag (this cut is non-periodic).
        supports_virial: Reserved capability flag (model emits a 3rd virial output).
        cuda_graph: Export a **fixed-shape** ``.pt2`` (``dynamic_shapes=None``) that
            the C++ side loads ``run_single_threaded`` and wraps in a CUDA graph
            (PyTorch #158834; ~2.9x on small systems). Requires ``n_atoms`` and
            ``e_max``; emits a 4-input model ``(Z, pos, edge_index, mask)`` with a
            ``StaticForward`` wrapper (padded edges inert via over-cutoff
            ``edge_diff``). Only the default ``molnex-tensordict`` adapter.
        n_atoms: Fixed atom count ``N`` (required when ``cuda_graph``).
        e_max: Fixed padded edge count (required when ``cuda_graph``); the C++ side
            must fail loudly when the real edge count exceeds it.

    Returns:
        The export directory as a :class:`Path`.

    Raises:
        ValueError: If ``species`` is empty, ``units`` is not understood, or
            ``cuda_graph`` is set without ``n_atoms``/``e_max``.
    """
    if not species:
        raise ValueError("species must be a non-empty list of atomic numbers")
    if units not in _UNIT_LABELS:
        raise ValueError(f"units must be one of {sorted(_UNIT_LABELS)}, got {units!r}")

    resolved_dtype = dtype or _infer_dtype(model)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if cuda_graph:
        if n_atoms is None or e_max is None:
            raise ValueError("cuda_graph=True requires n_atoms and e_max")
        from .static import StaticForward

        wrapper = StaticForward(model.to(device), n_atoms, e_max, cutoff)
        adapter_name = "molnex-tensordict-static"
        # static example: N atoms, E_max edges (all valid here is fine for tracing).
        gen = torch.Generator().manual_seed(0)
        pos = (
            (torch.rand(n_atoms, 3, generator=gen, dtype=torch.float64) - 0.5) * 0.45 * cutoff
        ).to(dtype=resolved_dtype, device=device)
        sp = sorted(set(species))
        Z = torch.tensor([sp[i % len(sp)] for i in range(n_atoms)], dtype=torch.long, device=device)
        ei = torch.randint(0, n_atoms, (e_max, 2), generator=gen, dtype=torch.long).to(device)
        mask = torch.ones(e_max, dtype=torch.bool, device=device)
        # fixed shapes → no dynamic_shapes; load run_single_threaded + CUDA graph in C++.
        Exporter(wrapper).export_pretraced(
            (Z, pos, ei, mask), export_dir / f"{name}.pt2", dynamic_shapes=None
        )
        inputs = ["Z", "pos", "edge_index", "mask"]
    else:
        wrapper = EngineForward(model.to(device), adapter)
        adapter_name = wrapper.adapter.name
        Z, pos, edge_index = _example_system(sorted(set(species)), cutoff, resolved_dtype, device)
        # N (atoms) and E (edges) vary across MD frames → mark dynamic so one .pt2
        # serves every step. min=2 dodges torch.export's 0/1-size specialization.
        n_dim = torch.export.Dim("n_atoms", min=2, max=1 << 20)
        e_dim = torch.export.Dim("n_edges", min=1, max=1 << 24)
        dynamic_shapes = ({0: n_dim}, {0: n_dim}, {0: e_dim})
        Exporter(wrapper).export_pretraced(
            (Z, pos, edge_index), export_dir / f"{name}.pt2", dynamic_shapes=dynamic_shapes
        )
        inputs = ["Z", "pos", "edge_index"]

    lammps_meta = {
        "schema_version": LAMMPS_META_SCHEMA,
        "cutoff": float(cutoff),
        "units": units,
        "energy_unit": _UNIT_LABELS[units],
        "species": sorted(set(int(z) for z in species)),
        "model_dtype": _DTYPE_NAMES.get(resolved_dtype, str(resolved_dtype).split(".")[-1]),
        "inputs": inputs,
        "outputs": ["energy", "forces"],
        "adapter": adapter_name,
        "supports_pbc": bool(supports_pbc),
        "supports_virial": bool(supports_virial),
        "cuda_graph": bool(cuda_graph),
    }
    if cuda_graph:
        lammps_meta["n_atoms"] = int(n_atoms)
        lammps_meta["e_max"] = int(e_max)
    meta = {"device": device, "model_class": model.__class__.__name__, "lammps": lammps_meta}
    (export_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

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
