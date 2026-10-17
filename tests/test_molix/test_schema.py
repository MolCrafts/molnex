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


def test_e0_key_is_the_graph_reference_sum():
    """Optional elemental reference total, same NestedKey shape as ENERGY_KEY."""
    from molix.schema import E0_KEY

    assert E0_KEY == ("graphs", "e0")


def test_energy_residual_key_is_the_graph_residual_label():
    """In-flight residual total, same NestedKey shape as ENERGY_KEY.

    Spec ``cache-layers-05-residual``. Address only; this module does not
    write the leaf. Distinct from ``E0_KEY`` (the reference sum, cache-layers-04).
    """
    from molix.schema import ENERGY_RESIDUAL_KEY

    assert ENERGY_RESIDUAL_KEY == ("graphs", "energy_residual")


def test_float64_label_keys_are_energy_and_e0():
    """Collate and batch_to skip this NestedKey set on the dtype axis."""
    from molix.schema import E0_KEY, ENERGY_KEY, FLOAT64_LABEL_KEYS

    assert ENERGY_KEY == ("graphs", "energy")
    assert FLOAT64_LABEL_KEYS == frozenset({ENERGY_KEY, E0_KEY})


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
    """One shared small-``r`` regularisation for the flat deployment batch.

    Two earlier flat-export paths clamped at 1e-6 and 1e-8 while claiming
    equivalence; spec ``lammps-iface-05-pbc`` standardised on the more
    conservative 1e-6. The behavioural assertion lives with the adapter
    (``tests/test_molix/test_deploy/``); the value is pinned here, where the
    constant is declared.
    """
    from molix.schema import PAIR_DIST_CLAMP_MIN

    assert PAIR_DIST_CLAMP_MIN == 1e-6


def test_has_energy_and_has_forces_walk_the_nesting():
    assert not has_energy(_batch())
    assert not has_forces(_batch())
    assert has_energy(_batch(energy=True))
    assert has_forces(_batch(forces=True))


def test_dead_edge_convention_is_one_function_of_the_cutoff():
    """The dead (padding) edge is ``[DEAD_PAIR_CUTOFF_FACTOR · r_cut, 0, 0]``, declared once.

    ``molix.md.NeighborList`` (MD capacity padding) and
    ``molix.data.MoleculePadder`` (training padding) displace a dead edge by
    the same vector, so every cutoff envelope evaluates to exactly zero on it
    and ``pair_dist`` is exactly the length. The value is pinned here, where
    it is declared.
    """
    from molix.schema import dead_pair_length, dead_pair_shift
    from molix.units import DEAD_PAIR_CUTOFF_FACTOR

    assert dead_pair_length(4.5) == DEAD_PAIR_CUTOFF_FACTOR * 4.5
    shift = dead_pair_shift(4.5, dtype=torch.float32)
    assert shift.dtype == torch.float32
    assert shift.tolist() == [DEAD_PAIR_CUTOFF_FACTOR * 4.5, 0.0, 0.0]
    assert float(shift.norm()) == dead_pair_length(4.5)
