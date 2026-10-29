"""ByteFF fixtures: the published functional form and the official weights directory."""

from __future__ import annotations

import os
from pathlib import Path

import molpy as mp
import pytest

#: ``ByteFF-26/models`` of HF ``ByteDance-Seed/byteff2`` at revision ``v1.1.0``.
WEIGHTS_DIR = Path(os.environ.get("MOLNEX_BYTEFF_WEIGHTS_DIR", "/nonexistent"))


@pytest.fixture
def forcefield() -> mp.ForceField:
    """The published functional form: harmonic valence, periodic torsions, cvff, LJ + Coulomb."""
    ff = mp.ff.forcefield.ForceField(name="byteff-form", units="real")
    ff.def_style("bond", "harmonic")
    ff.def_style("angle", "harmonic")
    ff.def_style("dihedral", "periodic")
    ff.def_style("improper", "cvff")
    ff.def_style("pair", "lj/cut", params={"cutoff": 50.0, "mixing": "arithmetic"})
    ff.def_style(
        "pair", "coul/cut", params={"cutoff": 50.0, "coulomb": 332.0637141491396, "dielectric": 1.0}
    )
    ff.set_special_bonds(lj=[0.0, 0.0, 0.5], coul=[0.0, 0.0, 0.8333333333])
    return ff
