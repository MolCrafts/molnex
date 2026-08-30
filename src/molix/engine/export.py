"""Export a molnex potential for the generic ``pair_style molnex`` LAMMPS interface.

LAMMPS is the C++ molecular-dynamics (MD) engine molnex deploys trained models
into. A *pair style* is its plug-in point for "given the atoms and their
neighbours, return the potential energy and the force on every atom".
``pair_style molnex`` is one such plug-in that contains no model-specific C++:
everything it needs to drive a particular network is read from the export this
module writes.

:func:`export_for_lammps` wraps a model in a flat-convention :class:`EngineForward`
(see :mod:`molix.engine.adapter`), compiles it ahead-of-time (AOT) via
:class:`molix.export.Exporter` into a single ``{name}.pt2`` package (a
self-contained AOTInductor artifact that runs without a Python interpreter),
then writes a sibling ``{name}.meta.json`` whose ``lammps`` block carries
everything the C++ side needs to stay model-agnostic. That block is schema v2 and
is assembled — as a whole and nowhere else — by :func:`_lammps_meta_block`:

* ``schema_version`` — contract version of this block, :data:`LAMMPS_META_SCHEMA`.
                      The C++ ``LammpsContract`` refuses any other value while
                      loading instead of guessing at the field set.
* ``contract``      — where the edge set comes from: ``"cutoff-graph"`` (recomputed
                      each step from ``cutoff``) or ``"fixed-graph"`` (edges given by a
                      fixed covalent topology). Both export paths here produce
                      ``"cutoff-graph"``; ``"fixed-graph"`` has no C++ implementation
                      yet and is refused explicitly, never run down the cutoff path.
* ``execution``     — execution strategy inside that contract: ``"dynamic"`` or
                      ``"cuda-graph"`` (fixed shapes). Selects the C++ executor.
* ``cutoff``        — neighbour cutoff in Å (the C++ side no longer takes it on the command line)
* ``units``         — the model's native unit system (``"real"`` = kcal/mol, ``"metal"`` = eV);
                      ``pair_style molnex`` reads LAMMPS' active ``units`` and converts.
* ``energy_unit`` / ``length_unit`` — labels for those native units. ``length_unit`` is
                      currently always ``"angstrom"``; the pair style asserts it before
                      reusing the energy factor as the force factor (``[F] = [E]/[L]``).
* ``energy_conv``   — ``{unit style: one model energy unit measured in that style}``,
                      evaluated by :func:`molix.units.preset_energy_ratio`, so no
                      conversion float is spelled out in Python or C++
* ``species``       — sorted atomic numbers ``Z`` the model supports
* ``model_dtype``   — ``"float32"`` / ``"float64"``; the pair style casts positions to this
* ``inputs`` / ``outputs`` — the flat tensor names, in call order (provenance + schema
                      checks): ``Z (N,)``, ``pos (N, 3)`` in Å and ``edge_index (E, 2)``,
                      plus ``mask (E,)`` under ``"cuda-graph"``; back come ``energy ()``
                      and ``forces (N, 3)``, plus ``virial (6,)`` when supported — all in
                      the model's native units.
* ``supports_pbc`` / ``supports_virial`` — capability flags read by ``pair_style molnex``.
  ``supports_pbc`` lets it put *ghost* atoms (LAMMPS' local copies of atoms owned by a
  periodic image or a neighbouring MPI rank) into the graph and reverse-communicate their
  forces back to the owning rank; ``supports_virial`` declares that the model itself
  returns the virial, the 6-component stress-like tensor LAMMPS turns into pressure.
* ``adapter``       — the registered adapter name used at export time (provenance only)
* ``n_atoms`` / ``e_max`` — fixed atom count and padded edge capacity. Written **only**
                      when ``execution == "cuda-graph"``; absent on the dynamic path.

The v1 ``cuda_graph`` boolean is retired in favour of ``execution``: a v1 plugin
reading a v2 export would silently default it to ``false`` and run a fixed-shape
model down the dynamic path, so plugin and exported artifacts must be upgraded as
a pair — see ``interface/lammps/README.md``.

Because MD changes the edge count ``E`` (and possibly the atom count ``N``) every
step, the default (``"dynamic"``) path marks those dimensions dynamic so one
exported package serves every frame. The ``"cuda-graph"`` path does the opposite
on purpose: it fixes both dimensions (``n_atoms`` / ``e_max``) because a CUDA
graph — a recorded, replayable sequence of GPU kernels — can only be captured
over static shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod
from molix.export import Exporter
from molix.units import preset_energy_ratio

from .adapter import EngineAdapter, EngineForward

logger = _logger_mod.getLogger(__name__)

#: meta.json ``lammps`` block schema version — bump on incompatible field changes.
#: The C++ reader pins the same number (``LammpsContract::kSchemaVersion`` in
#: ``interface/lammps/lammps_contract.h``) and refuses anything else, so the two
#: constants move together or the plugin stops loading new exports.
LAMMPS_META_SCHEMA = 2

#: unit systems the pair style understands (LAMMPS names → energy unit label).
#: Also the key set of the ``energy_conv`` table: one row per style listed here.
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


def _lammps_meta_block(
    *,
    species: list[int],
    cutoff: float,
    units: str,
    adapter_name: str,
    model_dtype: str,
    supports_pbc: bool,
    supports_virial: bool,
    cuda_graph: bool,
    n_atoms: int | None = None,
    e_max: int | None = None,
) -> dict[str, object]:
    """Assemble the schema-v2 ``lammps`` block of ``{name}.meta.json``.

    Pure dict assembly, split out from :func:`export_for_lammps` because the
    export path around it is CPU-refused (force export, see
    :class:`molix.export.Exporter`) and this contract must still be unit-tested
    on a CPU box — and because the C++ test fixtures are generated from it.

    ``contract`` and ``execution`` are the two orthogonal discriminators the C++
    side dispatches on and are *derived* here, never passed in: both current
    export paths build their edges from a cutoff, and only the fixed-shape path
    is CUDA-graph executable.

    Two fields are stamped rather than chosen. ``schema_version`` is always
    :data:`LAMMPS_META_SCHEMA`, the constant the C++ ``LammpsContract`` validates
    against at load. ``energy_conv`` holds one row per unit style molnex supports
    (:data:`_UNIT_LABELS`), each the value of one model energy unit in that style,
    so the row for the model's own ``units`` is exactly ``1.0``.

    Args:
        species: Atomic numbers the model supports; stored sorted and deduped.
        cutoff: Neighbour cutoff in Å.
        units: The model's native LAMMPS unit style (``"real"`` / ``"metal"``).
        adapter_name: Name of the :class:`~molix.engine.adapter.EngineAdapter`
            used at export time. Provenance only — must be a registered name.
        model_dtype: ``"float32"`` / ``"float64"``.
        supports_pbc: Capability flag; the pair style builds a ghost graph.
        supports_virial: Capability flag; adds ``"virial"`` to ``outputs``.
        cuda_graph: Fixed-shape export → ``execution == "cuda-graph"``, a fourth
            ``"mask"`` input, and the ``n_atoms`` / ``e_max`` fields.
        n_atoms: Fixed atom count ``N`` (required when ``cuda_graph``).
        e_max: Fixed padded edge count (required when ``cuda_graph``).

    Returns:
        The ``lammps`` block, ready to be JSON-serialised.

    Raises:
        ValueError: If ``cuda_graph`` is set without ``n_atoms`` / ``e_max``.
    """
    block: dict[str, object] = {
        "schema_version": LAMMPS_META_SCHEMA,
        "contract": "cutoff-graph",
        "execution": "cuda-graph" if cuda_graph else "dynamic",
        "cutoff": float(cutoff),
        "units": units,
        "energy_unit": _UNIT_LABELS[units],
        "length_unit": "angstrom",
        "energy_conv": {style: preset_energy_ratio(units, style) for style in _UNIT_LABELS},
        "species": sorted({int(z) for z in species}),
        "model_dtype": model_dtype,
        "inputs": ["Z", "pos", "edge_index", "mask"] if cuda_graph else ["Z", "pos", "edge_index"],
        "outputs": ["energy", "forces", "virial"] if supports_virial else ["energy", "forces"],
        "adapter": adapter_name,
        "supports_pbc": bool(supports_pbc),
        "supports_virial": bool(supports_virial),
    }
    if cuda_graph:
        if n_atoms is None or e_max is None:
            raise ValueError("execution 'cuda-graph' requires n_atoms and e_max")
        block["n_atoms"] = int(n_atoms)
        block["e_max"] = int(e_max)
    return block


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
        species: Atomic numbers the model supports. Recorded in ``meta.json``
            sorted and deduplicated, as the model's declared species set — the
            LAMMPS atom type → ``Z`` mapping itself comes from the user's
            ``pair_coeff * * <Z...>`` line, not from the order of this list.
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
        supports_pbc: If true, the pair style builds the graph over local +
            ghost atoms (copies of atoms owned by a periodic image or another
            MPI rank) and reverse-communicates ghost forces to their owners.
            Incompatible with ``cuda_graph`` — the ghost count varies, so the
            shapes cannot be fixed. That combination is *not* rejected here; the
            pair style refuses it when the model is loaded.
        supports_virial: If true, the model must emit a third ``(6,)`` virial
            tensor ``(xx, yy, zz, xy, xz, yz)`` in the model energy unit — the
            stress-like tensor LAMMPS accumulates into the pressure.
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
        ValueError: If ``cuda_graph`` is set together with a non-default
            ``adapter``. The fixed-shape wrapper inlines the ``molnex-tensordict``
            convention, so any other adapter could only be silently ignored —
            which would then be misreported as provenance in ``meta.json``.
    """
    if not species:
        raise ValueError("species must be a non-empty list of atomic numbers")
    if units not in _UNIT_LABELS:
        raise ValueError(f"units must be one of {sorted(_UNIT_LABELS)}, got {units!r}")

    adapter_name = adapter.name if isinstance(adapter, EngineAdapter) else adapter
    if cuda_graph and adapter_name != "molnex-tensordict":
        raise ValueError(
            f"the adapter argument ({adapter_name!r}) is not supported with cuda_graph=True; "
            "cuda-graph export wraps the model in StaticForward, which inlines the default "
            "'molnex-tensordict' convention and cannot honour another adapter"
        )

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
    else:
        wrapper = EngineForward(model.to(device), adapter)
        Z, pos, edge_index = _example_system(sorted(set(species)), cutoff, resolved_dtype, device)
        # N (atoms) and E (edges) vary across MD frames → mark dynamic so one .pt2
        # serves every step. min=2 dodges torch.export's 0/1-size specialization.
        n_dim = torch.export.Dim("n_atoms", min=2, max=1 << 20)
        e_dim = torch.export.Dim("n_edges", min=1, max=1 << 24)
        dynamic_shapes = ({0: n_dim}, {0: n_dim}, {0: e_dim})
        Exporter(wrapper).export_pretraced(
            (Z, pos, edge_index), export_dir / f"{name}.pt2", dynamic_shapes=dynamic_shapes
        )

    lammps_meta = _lammps_meta_block(
        species=species,
        cutoff=cutoff,
        units=units,
        adapter_name=adapter_name,
        model_dtype=_DTYPE_NAMES.get(resolved_dtype, str(resolved_dtype).split(".")[-1]),
        supports_pbc=supports_pbc,
        supports_virial=supports_virial,
        cuda_graph=cuda_graph,
        n_atoms=n_atoms,
        e_max=e_max,
    )
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
