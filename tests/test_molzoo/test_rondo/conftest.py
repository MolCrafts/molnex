"""Fixtures for the Rondo recipe: a Class-I force field and molecules collated the real way.

The frames carry every column the ByteFF-aligned encoder declares (formal
charge, ring features, bond order, ring membership) and the GAFF2 reference
wells the Lennard-Jones head starts from — the columns MolHub writes for THEMol.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molix import config
from molix.data import PairRelation, TotalCharge
from molix.data.collate import collate_molecules
from molix.data.valence_columns import cyclic_impropers as molix_cyclic_impropers
from molix.neighbor import NeighborList
from molpot.forcefield.partial_hessian import PartialHessianPairs
from molzoo.rondo import EncoderSpec, HeadsSpec, RondoSpec
from tests.conftest import (
    BUTANE_BONDS,
    BUTANE_POSITIONS,
    butane_angles,
    butane_neighbors,
    butane_propers,
    frame,
)

CUTOFF = 10.0

#: GAFF 2.2.30 ``MOD4`` rows (R*, ε) → (σ = 2R*/2^{1/6}, ε) for the types the fixtures use.
GAFF2 = {
    "c3": (2 * 1.9069 / 2 ** (1 / 6), 0.1078),
    "c1": (2 * 1.9525 / 2 ** (1 / 6), 0.1596),
    "hc": (2 * 1.4593 / 2 ** (1 / 6), 0.0208),
}


def class_i_forcefield(cutoff: float = CUTOFF, *, dihedral: str = "periodic") -> mp.ForceField:
    """The Class-I force field every Rondo fixture is built on (molrs IR)."""
    ff = mp.ff.forcefield.ForceField(name="rondo-test", units="real")
    ff.def_style("bond", "harmonic")
    ff.def_style("angle", "harmonic")
    ff.def_style("dihedral", dihedral)
    ff.def_style("improper", "cvff")
    ff.def_style("pair", "lj/cut", params={"cutoff": cutoff, "mixing": "geometric"})
    ff.def_style(
        "pair", "coul/cut", params={"cutoff": cutoff, "coulomb": 332.0637, "dielectric": 1.0}
    )
    ff.set_special_bonds(lj=[0.0, 0.0, 0.5], coul=[0.0, 0.0, 0.5])
    return ff


def _with_columns(f: mp.Frame, *, atoms: dict, bonds: dict) -> mp.Frame:
    """Add the extra atom / bond columns the encoder and the LJ head read."""
    atom_block = {key: np.asarray(f["atoms"][key]) for key in f["atoms"].keys()}
    atom_block.update({k: np.asarray(v) for k, v in atoms.items()})
    f["atoms"] = atom_block
    bond_block = {key: np.asarray(f["bonds"][key]) for key in f["bonds"].keys()}
    bond_block.update({k: np.asarray(v) for k, v in bonds.items()})
    f["bonds"] = bond_block
    return f


def cyclic_impropers(centres: list[tuple[int, int, int, int]]) -> dict[str, list[int]]:
    """The batch's three rows per centre (:func:`molix.data.cyclic_impropers`), as frame columns."""
    block = {
        name: torch.tensor([centre[n] for centre in centres], dtype=torch.long)
        for n, name in enumerate(("atomi", "atomj", "atomk", "atoml"))
    }
    return {name: column.tolist() for name, column in molix_cyclic_impropers(block).items()}


SPEC = RondoSpec(
    encoder=EncoderSpec(width=8, n_blocks=1),
    heads=HeadsSpec(hidden=8, proper_periodicity=(1, 2), eem_max_atoms=32),
)


def butane_frame() -> mp.Frame:
    """One butane at rest, no lattice, with ByteFF's input columns and GAFF2 wells.

    Butane has no planar centre; the two inner carbons are given improper rows
    anyway (three cyclic rows each, legs = three of the centre's neighbours) so
    the improper path is exercised.
    """
    angles = butane_angles()
    propers = butane_propers()
    adjacency = butane_neighbors()
    centres = [(c, *adjacency[c][:3]) for c in (1, 2)]
    z = [6] * 4 + [1] * 10
    f = frame(
        Z=z,
        pos=np.asarray(BUTANE_POSITIONS, dtype=np.float64),
        bonds=([b[0] for b in BUTANE_BONDS], [b[1] for b in BUTANE_BONDS]),
        terms={
            "angles": {
                "atomi": [a[0] for a in angles],
                "atomj": [a[1] for a in angles],
                "atomk": [a[2] for a in angles],
            },
            "propers": {
                "atomi": [p[0] for p in propers],
                "atomj": [p[1] for p in propers],
                "atomk": [p[2] for p in propers],
                "atoml": [p[3] for p in propers],
            },
            "impropers": cyclic_impropers(centres),
        },
        total_charge=0.0,
        energy=-1.0,
    )
    n_bonds = len(BUTANE_BONDS)
    return _with_columns(
        f,
        atoms={
            "formal_charge": np.zeros(len(z), dtype=np.int64),
            "ring_connectivity": np.zeros(len(z), dtype=np.int64),
            "min_ring_size": np.zeros(len(z), dtype=np.int64),
            "gaff2_sigma": np.array([GAFF2["c3" if zz == 6 else "hc"][0] for zz in z]),
            "gaff2_epsilon": np.array([GAFF2["c3" if zz == 6 else "hc"][1] for zz in z]),
        },
        bonds={
            "bond_type": np.ones(n_bonds, dtype=np.int64),
            "in_ring": np.zeros(n_bonds, dtype=np.int64),
        },
    )


def butyne_frame() -> mp.Frame:
    """1-butyne, H3C–CH2–C≡CH: a defined torsion (5-0-1-2) and two through sp carbons.

    Atoms: 0 C sp3 (CH3), 1 C sp3 (CH2), 2 C sp, 3 C sp, 4 H on 3, 5-7 H on 0,
    8-9 H on 1.
    """
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.19, 1.33, 0.0],
            [2.73, 2.41, 0.0],
            [3.20, 3.36, 0.0],
            [-0.37, 1.03, 0.0],
            [-0.37, -0.51, 0.89],
            [-0.37, -0.51, -0.89],
            [1.90, -0.51, 0.89],
            [1.90, -0.51, -0.89],
        ]
    )
    z = [6, 6, 6, 6, 1, 1, 1, 1, 1, 1]
    bonds = ([0, 1, 2, 3, 0, 0, 0, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    f = frame(
        Z=z,
        pos=pos,
        bonds=bonds,
        terms={
            "angles": {
                "atomi": [0, 1, 2, 5, 5, 6, 0, 8],
                "atomj": [1, 2, 3, 0, 0, 0, 1, 1],
                "atomk": [2, 3, 4, 6, 7, 7, 8, 9],
            },
            # 5-0-1-2 is defined; 0-1-2-3 and 1-2-3-4 pass through an sp carbon.
            "propers": {
                "atomi": [5, 0, 1],
                "atomj": [0, 1, 2],
                "atomk": [1, 2, 3],
                "atoml": [2, 3, 4],
            },
            "impropers": cyclic_impropers([(0, 5, 6, 7)]),
        },
        total_charge=0.0,
        energy=0.0,
    )
    types = ["c3", "c3", "c1", "c1", "hc", "hc", "hc", "hc", "hc", "hc"]
    return _with_columns(
        f,
        atoms={
            "formal_charge": np.zeros(10, dtype=np.int64),
            "ring_connectivity": np.zeros(10, dtype=np.int64),
            "min_ring_size": np.zeros(10, dtype=np.int64),
            "gaff2_sigma": np.array([GAFF2[t][0] for t in types]),
            "gaff2_epsilon": np.array([GAFF2[t][1] for t in types]),
        },
        bonds={
            "bond_type": np.array([1, 1, 3, 1, 1, 1, 1, 1, 1], dtype=np.int64),
            "in_ring": np.zeros(9, dtype=np.int64),
        },
    )


def collate(frames: list[mp.Frame]) -> TensorDict:
    """Collate as a run does: pair list at the field's cutoff, relations, total charge."""
    batch = collate_molecules(frames)
    NeighborList(cutoff=CUTOFF).execute(batch)
    PairRelation().execute(batch)
    TotalCharge().execute(batch)
    batch = batch.apply(lambda t: t.to(config.ftype) if t.is_floating_point() else t)
    batch["atoms", "forces"] = torch.zeros_like(batch["atoms", "pos"])
    return batch


@pytest.fixture
def forcefield() -> mp.ForceField:
    return class_i_forcefield()


@pytest.fixture
def butane_batch() -> TensorDict:
    return collate([butane_frame()])


@pytest.fixture
def butyne_batch() -> TensorDict:
    return collate([butyne_frame()])


def with_hessian_pairs(batch: TensorDict) -> PartialHessianPairs:
    """Add the partial-Hessian pair list and ``hpair_*`` slot columns, in place."""
    valence = {
        "bonds": batch["bonds", "bond_index"].long(),
        "angles": torch.stack([batch["angles", c].long() for c in ("atomi", "atomj", "atomk")]),
        "propers": torch.stack(
            [batch["propers", c].long() for c in ("atomi", "atomj", "atomk", "atoml")]
        ),
        "impropers": torch.stack(
            [batch["impropers", c].long() for c in ("atomi", "atomj", "atomk", "atoml")]
        ),
    }
    layout = PartialHessianPairs(valence)
    batch["hessian_pairs"] = TensorDict(
        {"atomi": layout.pairs[:, 0], "atomj": layout.pairs[:, 1]}, batch_size=[len(layout)]
    )
    for namespace, slots in layout.slots.items():
        for (i, j), slot in slots.items():
            batch[namespace, f"hpair_{i}_{j}"] = slot
    return layout


def excite(model) -> None:
    """Give every zero-started head a random non-zero output layer, in place."""
    with torch.no_grad():
        for head in model.heads.values():
            mlps = getattr(head, "mlps", None)
            if mlps is None:
                continue
            for mlp in mlps.values():
                mlp[-1].weight.normal_(0, 0.3)
                mlp[-1].bias.normal_(0, 0.3)
