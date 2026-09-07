"""Tests for molix.schema — the post-collate batch-schema key contract."""

import torch
from tensordict import TensorDict

from molix.schema import (
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    MASS_KEY,
    POS_KEY,
    VEL_KEY,
    has_energy,
    has_forces,
)


def _batch(*, energy: bool = False, forces: bool = False) -> TensorDict:
    atoms = {"pos": torch.zeros(4, 3)}
    if forces:
        atoms["forces"] = torch.zeros(4, 3)
    td = TensorDict({"atoms": TensorDict(atoms, batch_size=[4])}, batch_size=[])
    if energy:
        td["graphs"] = TensorDict({"energy": torch.zeros(1)}, batch_size=[1])
    return td


def test_keys_match_the_documented_schema():
    """The tuple keys are the CLAUDE.md post-collate contract, verbatim."""
    assert ENERGY_KEY == ("graphs", "energy")
    assert ATOMIC_ENERGY_KEY == ("atoms", "energy")
    assert FORCES_KEY == ("atoms", "forces")
    assert POS_KEY == ("atoms", "pos")
    assert VEL_KEY == ("atoms", "vel")
    assert MASS_KEY == ("atoms", "mass")


def test_flat_alias_of_the_nested_per_atom_energy_key():
    """The one flat spelling of the per-atom energy (spec ``lammps-iface-05-pbc``).

    ``("atoms", "energy")`` is the nested key; ``"atomic_energy"`` is its flat
    alias, used by plain-dict model outputs and by the LAMMPS export's
    ``outputs`` list. Exactly one flat spelling exists — a third one
    (``node_energies`` and friends) is what the constant is here to prevent.
    Imported inside the test so the rest of this module stays green while the
    constant is missing.
    """
    from molix.schema import ATOMIC_ENERGY_FLAT_KEY

    assert ATOMIC_ENERGY_FLAT_KEY == "atomic_energy"
    assert ATOMIC_ENERGY_KEY == ("atoms", "energy")


def test_edge_distance_clamp_is_the_conservative_value():
    """One shared small-``r`` regularisation for both flat-export paths.

    ``molix.engine.adapter`` clamped at 1e-6 and ``molix.engine.static`` at
    1e-8 while claiming equivalence; spec ``lammps-iface-05-pbc`` standardises
    on the more conservative 1e-6. The behavioural assertions live with each
    path (``tests/test_molix/test_engine/``); the value is pinned here, where
    the constant is declared.
    """
    from molix.schema import EDGE_DIST_CLAMP_MIN

    assert EDGE_DIST_CLAMP_MIN == 1e-6


def test_has_energy_and_has_forces_walk_the_nesting():
    assert not has_energy(_batch())
    assert not has_forces(_batch())
    assert has_energy(_batch(energy=True))
    assert has_forces(_batch(forces=True))
