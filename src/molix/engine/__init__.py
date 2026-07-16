"""Export molnex potentials for external MD engines (AOTInductor ``.pt2``).

The general half of the engine bridge — model wrapping + AOT export — lives here
in ``molix``; the engine-specific C++ (currently the LAMMPS ``pair_style
molnex``) lives in the repo-root ``interface/``. A potential is wrapped to the
flat ``(Z, pos, edge_index) -> (energy, forces)`` calling convention by an
:class:`EngineForward` + :class:`EngineAdapter` (engine-neutral; the same flat
convention any atomistic MD engine speaks), then AOT-exported. ``interface/`` has
no per-model C++ — all model glue is this Python side.

:func:`export_for_lammps` is the LAMMPS preset (it stamps the ``lammps`` meta
block ``pair_style molnex`` reads):

    >>> from molix.engine import export_for_lammps
    >>> export_for_lammps(pinet_potential, "pinet_aspirin",
    ...                   species=[1, 6, 7, 8], cutoff=4.5, units="real")
    # then in LAMMPS:  pair_style molnex pinet_aspirin
    #                  pair_coeff * * 1 6 7 8

Third-party models: implement a ~15-line :class:`EngineAdapter` subclass (or use
:class:`FlatTensorAdapter` if the model already speaks the flat convention) and
pass it as ``adapter=`` to :func:`export_for_lammps`.

The export stamps a ``lammps`` block into ``meta.json`` (cutoff, units, species,
dtype, capabilities) so the C++ side reads everything it needs from the export
directory rather than the command line.
"""

from __future__ import annotations

from .adapter import (
    EngineAdapter,
    EngineForward,
    FlatTensorAdapter,
    MolnexTensorDictAdapter,
)
from .export import LAMMPS_META_SCHEMA, export_for_lammps
from .static import StaticForward

__all__ = [
    "LAMMPS_META_SCHEMA",
    "FlatTensorAdapter",
    "EngineAdapter",
    "EngineForward",
    "MolnexTensorDictAdapter",
    "StaticForward",
    "export_for_lammps",
]
