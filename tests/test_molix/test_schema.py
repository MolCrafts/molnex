"""Tests for molix.schema — the post-collate batch-schema key contract."""

import torch
from tensordict import TensorDict

from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY, POS_KEY, has_energy, has_forces


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


def test_has_energy_and_has_forces_walk_the_nesting():
    assert not has_energy(_batch())
    assert not has_forces(_batch())
    assert has_energy(_batch(energy=True))
    assert has_forces(_batch(forces=True))


def test_molpot_protocol_reexports_the_same_objects():
    """molpot.derivation.protocol must alias, not restate, the schema keys."""
    from molpot.derivation import protocol

    assert protocol.ENERGY_KEY is ENERGY_KEY
    assert protocol.FORCES_KEY is FORCES_KEY
    assert protocol.has_forces is has_forces
