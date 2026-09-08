"""Export a molnex potential for the generic ``pair_style molnex`` LAMMPS interface.

LAMMPS is the C++ molecular-dynamics (MD) engine molnex deploys trained models
into. A *pair style* is its plug-in point for "given the atoms and their
neighbours, return the potential energy and the force on every atom".
``pair_style molnex`` is one such plug-in that contains no model-specific C++:
everything it needs to drive a particular network is read from the export this
module writes.

:func:`export_for_lammps` wraps a model in a flat-convention :class:`EngineForward`
(see :mod:`molix.engine.adapter`), compiles it ahead-of-time (AOT) via
:class:`molix.export.Exporter` into a single ``{name}.pt2`` package — a
self-contained *AOTInductor* (AOTI) artifact, i.e. compiled machine code plus
weights that runs without a Python interpreter — then writes a sibling
``{name}.meta.json`` whose ``lammps`` block carries everything the C++ side needs
to stay model-agnostic. That block is schema v3 and is assembled — as a whole and
nowhere else — by :func:`_lammps_meta_block`:

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
                      checks): ``Z (N,) int64``, ``pos (N, 3)`` in Å, ``edge_index (E, 2)
                      int64`` and ``local_mask (N,) bool``, with ``edge_mask (E_max,) bool``
                      inserted before ``local_mask`` under ``"cuda-graph"``; back come
                      ``energy ()``, ``forces (N, 3)`` and ``atomic_energy (N,)``, plus
                      ``virial (6,)`` when supported — all in the model's native units.
                      ``local_mask`` marks the atoms the rank owns; the exported energy is
                      ``(atomic_energy * local_mask).sum()``.
* ``receptive_field_layers`` — ``T``, the number of message-passing layers, a positive int
                      with **no default**. A site's energy depends on everything within
                      ``T * cutoff``, so the C++ side checks
                      ``ghost cutoff >= T * cutoff + skin`` before running and refuses
                      otherwise. It cannot infer ``T`` from anything else in this block,
                      and guessing 1 would silently declare a 3-layer MACE strictly local.
* ``requires_sorted_edges`` — whether the model needs ``edge_index`` sorted by
                      ``(source, target)``. Default ``False``: scatter-class reductions are
                      permutation-invariant, so the C++ side skips a per-step sort. Models
                      using segmented / CSR aggregation must declare ``True``. The asymmetry
                      with ``receptive_field_layers`` is deliberate — every model *has* a
                      receptive field, so the author must state it; almost none need sorted
                      edges, so opting in is right.
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

**v2 → v3 is a hard break: every export produced before v3 is invalid and must be
re-exported.** The tensor count, names and meaning all moved — an added
``local_mask`` input, an added ``atomic_energy`` output, ``mask`` renamed
``edge_mask``, and the two new fields above. The C++ ``LammpsContract`` pins the
schema version and refuses anything else rather than guessing at the field set,
so an old artifact fails loudly at load instead of running a mis-wired graph.

An optional third artifact, ``{name}.manifest.json``, is written when
:func:`export_for_lammps` is handed a ``manifest`` mapping: the caller's
model-provenance record — which model, trained on what, at which precision —
carried verbatim for humans and tooling. In molnex today it is produced by
``ForceField.manifest(profile=...).to_dict()`` in ``molpot``, but this module
never imports that package and never inspects the mapping; it travels as plain
JSON-native data. It is invisible to the C++ side — which reads
``doc["lammps"]`` of ``meta.json`` and nothing else — and its presence changes
no byte of that file.

Two properties of a model are checked **eagerly, on the example system, before
tracing**, because neither can be expressed as a traced assertion and both are
silent physics errors if left to runtime:

1. *decomposability* — ``|energy - Σ atomic_energy|`` within rel 1e-6. Under v3
   the deployed energy is the masked site-energy sum, so a model carrying a term
   with no per-site attribution would be deployed as a *different* energy
   function than the one that was trained.
2. *force consistency* — with an all-ones mask, the wrapper's derived forces
   ``-∂(Σ E_i)/∂pos`` against the model's own forces, rel 1e-5. A model whose
   reported forces are not the gradient of its reported energy fails here; a
   consistent one passes trivially.

Because MD changes the edge count ``E`` (and possibly the atom count ``N``) every
step, the default (``"dynamic"``) path marks those dimensions dynamic so one
exported package serves every frame. The ``"cuda-graph"`` path does the opposite
on purpose: it fixes both dimensions (``n_atoms`` / ``e_max``) because a CUDA
graph — a recorded, replayable sequence of GPU kernels — can only be captured
over static shapes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from molix import logger as _logger_mod
from molix.export import Exporter
from molix.units import preset_energy_ratio

from .adapter import EngineAdapter, EngineForward
from .static import StaticForward

logger = _logger_mod.getLogger(__name__)

#: meta.json ``lammps`` block schema version — bump on incompatible field changes.
#: The C++ reader pins the same number (``LammpsContract::kSchemaVersion`` in
#: ``interface/lammps/lammps_contract.h``) and refuses anything else, so the two
#: constants move together or the plugin stops loading new exports.
LAMMPS_META_SCHEMA = 3

#: Relative tolerance of the eager decomposability check: the model's scalar
#: energy against the sum of its per-atom energies on the example system.
_DECOMPOSITION_RTOL = 1e-6
#: Relative tolerance of the eager force-consistency check: the wrapper's
#: derived forces against the model's own, all-ones mask. Looser than the energy
#: check because it compares a full backward pass against an independently
#: computed field, not two spellings of one number.
_FORCE_CONSISTENCY_RTOL = 1e-5

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


def _check_partitionable(
    wrapper: "EngineForward | StaticForward",
    model_inputs: tuple[torch.Tensor, ...],
    local_mask: torch.Tensor,
) -> None:
    """Refuse, eagerly, a model schema v3 cannot deploy correctly.

    Runs in **eager mode on the example system, before any tracing**. Neither
    property below survives being deferred: a traced graph cannot branch on a
    data-dependent comparison, and at MD runtime the C++ side has nothing to
    compare against — it would simply integrate the wrong forces.

    1. *Decomposability*: ``|energy - Σ atomic_energy|`` within
       :data:`_DECOMPOSITION_RTOL`, relative to the larger magnitude. The
       deployed energy is ``(atomic_energy * local_mask).sum()``, so a model
       carrying an energy term with no per-site attribution would be deployed as
       a different energy function than the one that was trained — silently,
       and by an amount no test downstream of the export could attribute.
    2. *Force consistency*: with an all-ones mask (where the partition is the
       whole system), the wrapper's derived forces against the model's own,
       within :data:`_FORCE_CONSISTENCY_RTOL`. The wrapper differentiates the
       site-energy sum, so this asks whether the model's reported forces really
       are the gradient of its reported energy. For a consistent model the two
       agree to round-off; a model that fails here would have shipped forces
       that do not conserve its own energy. An **energy-only** model (forces
       slot ``None`` — e.g. ``PiNetPotential(compute_forces=False)``) has
       nothing to compare: the deployed forces *are* the derivative, so only
       their finiteness is asserted.

    Args:
        wrapper: The export wrapper — :class:`~molix.engine.adapter.EngineForward`
            or :class:`~molix.engine.static.StaticForward`. Both expose
            ``model_outputs`` (the model's unmasked triple) and a partitioned
            ``forward``, which is all this needs.
        model_inputs: The wrapper's inputs *up to but excluding* ``local_mask``.
        local_mask: The all-ones mask completing the ``forward`` call.

    Raises:
        ValueError: If either check fails, naming the measured values.
    """
    energy, forces, atomic_energy = wrapper.model_outputs(*model_inputs)
    total = float(energy.detach().sum())
    site_sum = float(atomic_energy.detach().sum())
    scale = max(abs(total), abs(site_sum))
    if abs(total - site_sum) > _DECOMPOSITION_RTOL * scale:
        raise ValueError(
            f"model energy {total!r} is not the sum of its per-atom energies {site_sum!r} "
            f"(difference {abs(total - site_sum)!r}, tolerance rel {_DECOMPOSITION_RTOL}). "
            "Schema v3 deploys the masked site-energy sum, so a term with no per-site "
            "attribution would change the energy function at deployment. Attribute the "
            "missing term to sites (a constant offset belongs in the per-species E_0), or "
            "export a model whose atomic_energy is its full energy decomposition."
        )

    _, derived_forces, _ = wrapper(*model_inputs, local_mask)
    if not bool(torch.isfinite(derived_forces).all()):
        raise ValueError(
            "the wrapper's derived forces -d(sum of per-atom energies)/dpos are not finite "
            "on the example system; the export would integrate NaN/Inf forces at MD runtime"
        )
    if forces is None:
        return
    reference = forces.detach()
    deviation = float((derived_forces.detach() - reference).abs().max())
    reference_scale = float(reference.abs().max())
    if deviation > _FORCE_CONSISTENCY_RTOL * reference_scale:
        raise ValueError(
            f"model forces disagree with -d(sum of per-atom energies)/dpos by {deviation!r} "
            f"(max |F| {reference_scale!r}, tolerance rel {_FORCE_CONSISTENCY_RTOL}). The "
            "export derives forces from the site energies so that a partial local_mask "
            "yields the partition force, so the two must already agree at full mask; they "
            "do not, which means the model's forces are not the gradient of its energy."
        )


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
    receptive_field_layers: int,
    requires_sorted_edges: bool = False,
    n_atoms: int | None = None,
    e_max: int | None = None,
) -> dict[str, object]:
    """Assemble the schema-v3 ``lammps`` block of ``{name}.meta.json``.

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
            Under v3 the virial is the **fourth** output — the per-atom energy
            precedes it — and the C++ decoder resolves output positions by this
            ``outputs`` name list rather than by a baked-in index.
        cuda_graph: Fixed-shape export → ``execution == "cuda-graph"``, an
            ``"edge_mask"`` input before ``"local_mask"``, and the ``n_atoms`` /
            ``e_max`` fields.
        receptive_field_layers: ``T``, the number of message-passing layers.
            Positive int, no default; see the module docstring.
        requires_sorted_edges: Whether the C++ side must sort ``edge_index`` by
            ``(source, target)`` before the call. Default ``False``.
        n_atoms: Fixed atom count ``N`` (required when ``cuda_graph``).
        e_max: Fixed padded edge count (required when ``cuda_graph``).

    Returns:
        The ``lammps`` block, ready to be JSON-serialised.

    Raises:
        ValueError: If ``receptive_field_layers`` is not a positive int, or if
            ``cuda_graph`` is set without ``n_atoms`` / ``e_max``.
    """
    if isinstance(receptive_field_layers, bool) or not isinstance(receptive_field_layers, int):
        raise ValueError(
            "receptive_field_layers must be an int (the number of message-passing "
            f"layers T), got {receptive_field_layers!r}"
        )
    if receptive_field_layers < 1:
        raise ValueError(
            "receptive_field_layers must be positive: every model has at least one "
            f"layer, and the fat-ghost check needs T >= 1; got {receptive_field_layers!r}"
        )
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
        "inputs": (
            ["Z", "pos", "edge_index", "edge_mask", "local_mask"]
            if cuda_graph
            else ["Z", "pos", "edge_index", "local_mask"]
        ),
        "outputs": (
            ["energy", "forces", "atomic_energy", "virial"]
            if supports_virial
            else ["energy", "forces", "atomic_energy"]
        ),
        "adapter": adapter_name,
        "supports_pbc": bool(supports_pbc),
        "supports_virial": bool(supports_virial),
        "receptive_field_layers": int(receptive_field_layers),
        "requires_sorted_edges": bool(requires_sorted_edges),
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
    receptive_field_layers: int | None = None,
    requires_sorted_edges: bool = False,
    n_atoms: int | None = None,
    e_max: int | None = None,
    manifest: Mapping[str, object] | None = None,
) -> Path:
    """Export ``model`` to an AOTI ``.pt2`` consumable by ``pair_style molnex <dir>``.

    Args:
        model: A potential. With the default ``adapter`` it must accept a nested
            ``TensorDict`` and return ``{energy, atomic_energy}``, optionally
            with ``forces`` (e.g. ``PiNetPotential``). The per-atom energy is
            mandatory under schema v3 and must sum to the scalar energy — both
            are checked eagerly on the example system before anything is
            traced. Forces, when present, are checked against the derived
            gradient; energy-only is the preferred export shape (the wrapper
            derives the deployed forces itself, and a model whose own force
            pass consumes its autograd graph in eval mode — PiNet's
            ``method="grad"`` — cannot be re-differentiated at trace time).
        export_dir: Output directory (created if missing); receives ``{name}.pt2``
            and ``{name}.meta.json``, plus ``{name}.manifest.json`` whenever
            ``manifest`` is not ``None`` — an empty mapping is a manifest that
            says nothing, not the absence of one, and still writes its file.
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
            ``e_max``; emits a 5-input model
            ``(Z, pos, edge_index, edge_mask, local_mask)`` with a
            ``StaticForward`` wrapper (padded edges inert via over-cutoff
            ``edge_diff``). Only the default ``molnex-tensordict`` adapter.
        receptive_field_layers: ``T``, the number of message-passing layers —
            **required, no default**. A site energy depends on everything within
            ``T * cutoff``, and the LAMMPS side must hold a ghost shell that
            wide; it checks ``ghost cutoff >= T * cutoff + skin`` and refuses
            otherwise. Defaulting this to 1 would let a 3-layer model pass a
            check computed at a third of the radius it needs, which is worse
            than no check at all — hence the explicit refusal below rather than
            a plausible-looking default.
        requires_sorted_edges: Declare that the model needs ``edge_index``
            sorted by ``(source, target)`` — true for segmented / CSR
            aggregation, false (the default) for the scatter-class reductions
            every in-tree model uses. When false the C++ side skips a per-step
            sort; when true it guarantees the order.
        n_atoms: Fixed atom count ``N`` (required when ``cuda_graph``).
        e_max: Fixed padded edge count (required when ``cuda_graph``); the C++ side
            must fail loudly when the real edge count exceeds it.
        manifest: Optional provenance record of the exported model — the
            caller's ``ModelManifest`` already reduced to a plain mapping
            (``ModelManifest.to_dict()``). Its schema belongs to the caller and
            is opaque here: ``molix.engine`` must not import ``molpot``, so this
            travels as data, is never inspected, and is written verbatim
            (sorted keys, indented) to ``export_dir/{name}.manifest.json``
            beside the ``.pt2``. It is for humans and tooling only — the C++
            reader validates ``doc["lammps"]`` of ``meta.json`` and never opens
            this file — and ``meta.json`` is byte-identical whether or not a
            manifest is passed.

    Returns:
        The export directory as a :class:`Path`.

    Raises:
        ValueError: If ``species`` is empty, ``units`` is not understood, or
            ``cuda_graph`` is set without ``n_atoms``/``e_max``.
        ValueError: If ``receptive_field_layers`` is omitted or not a positive
            int. The message names the keyword and gives a re-export command,
            because this is normally hit *after* a finished training run.
        ValueError: If ``cuda_graph`` is set together with a non-default
            ``adapter``. The fixed-shape wrapper inlines the ``molnex-tensordict``
            convention, so any other adapter could only be silently ignored —
            which would then be misreported as provenance in ``meta.json``.
        ValueError: If the model's scalar energy is not the sum of its per-atom
            energies (rel 1e-6), or its forces are not the gradient of that
            energy (rel 1e-5), on the example system. Both are checked eagerly,
            before tracing.
    """
    if not species:
        raise ValueError("species must be a non-empty list of atomic numbers")
    if units not in _UNIT_LABELS:
        raise ValueError(f"units must be one of {sorted(_UNIT_LABELS)}, got {units!r}")
    if receptive_field_layers is None:
        raise ValueError(
            "receptive_field_layers is required by schema v3 and has no default: it is "
            "the number of message-passing layers T, which LAMMPS needs to size the "
            "ghost shell (ghost cutoff >= T * cutoff + skin) and cannot infer from the "
            "export. Re-export with it, e.g.\n"
            "    export_for_lammps(model, export_dir, species=[...], cutoff=..., "
            "receptive_field_layers=<T>)"
        )

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

    pt2_path = export_dir / f"{name}.pt2"
    if cuda_graph:
        if n_atoms is None or e_max is None:
            raise ValueError("cuda_graph=True requires n_atoms and e_max")
        _export_cuda_graph(
            model,
            pt2_path,
            species=species,
            cutoff=cutoff,
            dtype=resolved_dtype,
            device=device,
            n_atoms=n_atoms,
            e_max=e_max,
        )
    else:
        _export_dynamic(
            model,
            pt2_path,
            adapter=adapter,
            species=species,
            cutoff=cutoff,
            dtype=resolved_dtype,
            device=device,
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
        receptive_field_layers=receptive_field_layers,
        requires_sorted_edges=requires_sorted_edges,
        n_atoms=n_atoms,
        e_max=e_max,
    )
    meta = {"device": device, "model_class": model.__class__.__name__, "lammps": lammps_meta}
    (export_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if manifest is not None:
        manifest_path = export_dir / f"{name}.manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(dict(manifest), fh, indent=2, sort_keys=True)
            fh.write("\n")

    logger.info(
        f"Exported LAMMPS model to {export_dir} "
        f"(adapter={adapter_name}, units={units}, cutoff={cutoff} A, "
        f"dtype={meta['lammps']['model_dtype']}, species={meta['lammps']['species']})"
    )
    return export_dir


def _export_cuda_graph(
    model: nn.Module,
    pt2_path: Path,
    *,
    species: list[int],
    cutoff: float,
    dtype: torch.dtype,
    device: str,
    n_atoms: int,
    e_max: int,
) -> None:
    """Trace the fixed-shape 5-input ``StaticForward`` wrapper to *pt2_path*."""
    wrapper = StaticForward(model.to(device), n_atoms, e_max, cutoff)
    # static example: N atoms, E_max edges (all valid here is fine for tracing).
    gen = torch.Generator().manual_seed(0)
    pos = ((torch.rand(n_atoms, 3, generator=gen, dtype=torch.float64) - 0.5) * 0.45 * cutoff).to(
        dtype=dtype, device=device
    )
    sp = sorted(set(species))
    Z = torch.tensor([sp[i % len(sp)] for i in range(n_atoms)], dtype=torch.long, device=device)
    ei = torch.randint(0, n_atoms, (e_max, 2), generator=gen, dtype=torch.long).to(device)
    edge_mask = torch.ones(e_max, dtype=torch.bool, device=device)
    local_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    _check_partitionable(wrapper, (Z, pos, ei, edge_mask), local_mask)
    # fixed shapes → no dynamic_shapes; load run_single_threaded + CUDA graph in C++.
    Exporter(wrapper).export_pretraced(
        (Z, pos, ei, edge_mask, local_mask), pt2_path, dynamic_shapes=None
    )


def _export_dynamic(
    model: nn.Module,
    pt2_path: Path,
    *,
    adapter: EngineAdapter | str,
    species: list[int],
    cutoff: float,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Trace the dynamic-shape 4-input ``EngineForward`` wrapper to *pt2_path*."""
    wrapper = EngineForward(model.to(device), adapter)
    Z, pos, edge_index = _example_system(sorted(set(species)), cutoff, dtype, device)
    local_mask = torch.ones(Z.shape[0], dtype=torch.bool, device=device)
    _check_partitionable(wrapper, (Z, pos, edge_index), local_mask)
    # N (atoms) and E (edges) vary across MD frames → mark dynamic so one .pt2
    # serves every step. min=2 dodges torch.export's 0/1-size specialization.
    n_dim = torch.export.Dim("n_atoms", min=2, max=1 << 20)
    e_dim = torch.export.Dim("n_edges", min=1, max=1 << 24)
    # local_mask is per-atom, so it rides the same dynamic dim as Z / pos.
    dynamic_shapes = ({0: n_dim}, {0: n_dim}, {0: e_dim}, {0: n_dim})
    Exporter(wrapper).export_pretraced(
        (Z, pos, edge_index, local_mask), pt2_path, dynamic_shapes=dynamic_shapes
    )


def _infer_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32
