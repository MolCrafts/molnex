"""Shared fixtures for the ``molix.ff_export`` unit tests.

The one nonbonded method both backend adapters are pinned against lives here
because the LAMMPS and OpenMM cases are only comparable while they describe the
*same* model — a drifting copy on either side would turn a real disagreement
between the two decks into two tests that quietly measure different runs.
"""

from __future__ import annotations

from molpot.ir import NonbondedMethodRecord

#: PME under a cell, with a Lennard-Jones switching window.
#:
#: Radii in Å (LJ tapered from 10 to 12, Ewald real-space split at 15, Gaussian
#: smearing width 5), reciprocal-space wavelength and mesh spacing in Å, and a
#: dimensionless capacity factor. ``real_space_cutoff`` is the derived
#: ``PAIR_STYLE_DEFAULTS.real_space_cutoff_factor * smearing`` = 3 x 5, i.e.
#: what a force field declaring only ``calculator="pme"`` actually runs at.
#: ``lj_mixing`` is the ``PAIR_STYLE_DEFAULTS.lj_mixing`` fallback, so a test
#: wanting the OPLS-AA rule states it with
#: ``dataclasses.replace(..., lj_mixing="geometric")`` and the difference in the
#: exported deck is attributable to that one field.
PME_SWITCHED_METHOD = NonbondedMethodRecord(
    lj_cutoff=12.0,
    lj_switch_start=10.0,
    lj_mixing="lorentz-berthelot",
    coulomb_cutoff=None,
    coulomb_calculator="pme",
    real_space_cutoff=15.0,
    smearing=5.0,
    lr_wavelength=1.0,
    mesh_spacing=1.0,
    capacity_factor=1.0,
)
