"""Generic LAMMPS interface for molnex potentials (``pair_style molnex``).

One command, any model. A potential that can be AOT-exported to the flat
``(Z, pos, edge_index) -> (energy, forces)`` calling convention is driven by a
single C++ pair style (``interface/lammps/pair_molnex.cpp``) — no per-model C++.
Model-specific glue lives entirely on the Python export side as an
:class:`LammpsAdapter`:

    >>> from molix.lammps import export_for_lammps
    >>> export_for_lammps(pinet_potential, "pinet_aspirin",
    ...                   species=[1, 6, 7, 8], cutoff=4.5, units="real")
    # then in LAMMPS:  pair_style molnex pinet_aspirin
    #                  pair_coeff * * 1 6 7 8

Third-party models: implement a ~15-line :class:`LammpsAdapter` subclass (or use
:class:`FlatTensorAdapter` if the model already speaks the flat convention) and
pass it as ``adapter=`` to :func:`export_for_lammps`.

The export stamps a ``lammps`` block into ``meta.json`` (cutoff, units, species,
dtype, capabilities) so the C++ side reads everything it needs from the export
directory rather than the command line.
"""

from __future__ import annotations

from .adapter import (
    FlatTensorAdapter,
    LammpsAdapter,
    LammpsForward,
    MolnexTensorDictAdapter,
)
from .export import LAMMPS_META_SCHEMA, export_for_lammps

__all__ = [
    "LAMMPS_META_SCHEMA",
    "FlatTensorAdapter",
    "LammpsAdapter",
    "LammpsForward",
    "MolnexTensorDictAdapter",
    "export_for_lammps",
]
