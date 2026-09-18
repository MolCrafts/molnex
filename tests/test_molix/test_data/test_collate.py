import numpy as np
import pytest
import torch

from molix.data.collate import (
    INDEX_KEYS,
    MoleculeCollator,
    collate_molecules,
    freeze_graph_columns,
    graph_schema_from,
    rebase,
)
from molix.frame import TensorFrame
from tests.conftest import frame


def test_collate_basic_fields_and_offsets():
    """Two frames become one batch: atoms concatenate, graphs stack, no edges."""
    batch = collate_molecules(
        [
            frame(Z=[1, 8], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], U0=1.5),
            frame(
                Z=[6, 1, 1],
                pos=[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
                U0=2.5,
            ),
        ]
    )

    # Collate hands out the *typed* view: the conversion belongs to the data
    # path, so a model never has to wrap a bare batch inside its forward.
    assert isinstance(batch, TensorFrame)
    assert batch["atoms", "Z"].shape == (5,)
    assert batch["atoms", "pos"].shape == (5, 3)
    assert batch["atoms", "batch"].tolist() == [0, 0, 1, 1, 1]
    assert batch["graphs"].batch_size[0] == 2
    assert batch["graphs", "num_atoms"].tolist() == [2, 3]

    # Frames carry no neighbour graph: NeighborList builds it on the batch.
    assert batch["pairs", "pair_index"].shape == (0, 2)

    # The frames are fp64 at rest and the collate keeps that precision.
    assert torch.allclose(batch["graphs", "U0"], torch.tensor([1.5, 2.5], dtype=torch.float64))


def test_molecule_collator_is_reusable_callable():
    frames = [
        frame(Z=[1], pos=[[0.0, 0.0, 0.0]], energy=1.0),
        frame(Z=[6, 1], pos=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], energy=2.0),
    ]
    batch = MoleculeCollator()(frames)
    assert batch["atoms", "batch"].tolist() == [0, 1, 1]
    assert batch["graphs", "energy"].tolist() == [1.0, 2.0]


def test_graph_schema_from_is_canonical_not_insertion_order():
    assert graph_schema_from(["spin", "task", "energy", "num_atoms"]) == (
        "num_atoms",
        "energy",
        "task",
        "spin",
    )


def test_freeze_graph_columns_rebuilds_to_the_compiled_schema():
    from tensordict import TensorDict

    graphs = TensorDict(
        {
            "spin": torch.ones(2),
            "task": torch.zeros(2, dtype=torch.long),
            "energy": torch.tensor([1.0, 2.0]),
            "num_atoms": torch.tensor([3, 4]),
        },
        batch_size=[2],
    )
    batch = TensorDict(graphs=graphs, batch_size=[])
    schema = ("num_atoms", "energy", "task")
    out = freeze_graph_columns(batch, schema)
    assert list(out["graphs"].keys()) == ["num_atoms", "energy", "task"]
    assert "spin" not in out["graphs"].keys()


def test_freeze_graph_columns_raises_when_a_compiled_name_is_missing():
    from tensordict import TensorDict

    batch = TensorDict(
        graphs=TensorDict({"num_atoms": torch.tensor([1, 2])}, batch_size=[2]),
        batch_size=[],
    )
    with pytest.raises(ValueError, match="missing"):
        freeze_graph_columns(batch, ("num_atoms", "energy"))


def test_index_keys_registry_contract():
    # Pre-declared registry that sub-specs 02/03 import as a stable contract.
    assert INDEX_KEYS["pair_index"] == (0, 1)
    assert INDEX_KEYS["bond_index"] == (1, 0)
    assert INDEX_KEYS["angle_index"] == (1, 0)
    assert INDEX_KEYS["dihedral_index"] == (1, 0)


def test_rebase_scalar_offset_edge_index():
    # collate_molecules path: scalar atom_offset added to the whole [E, 2] tensor.
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    out = rebase(pair_index, 2, "pair_index")
    assert out.tolist() == [[2, 3], [2, 4]]


def test_rebase_vector_offset_edge_index_matches_unsqueeze1():
    # packed path: per-edge segment offset broadcast over both columns.
    pair_index = torch.tensor([[0, 1], [0, 2], [1, 0]], dtype=torch.long)
    seg_offset = torch.tensor([0, 0, 5], dtype=torch.long)
    out = rebase(pair_index, seg_offset, "pair_index")
    assert torch.equal(out, pair_index + seg_offset.unsqueeze(1))
    assert out.tolist() == [[0, 1], [0, 2], [6, 5]]


def test_rebase_generalizes_to_cat_dim1_mock_bond_index():
    # Proves the differing-axis path before phase 03 produces a real bond_index:
    # a [2, N] COO tensor with a per-count (per-bond) offset must broadcast over
    # the rows (index_axis=0), i.e. offset.unsqueeze(0).
    bond_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    per_bond_offset = torch.tensor([10, 10, 20], dtype=torch.long)
    out = rebase(bond_index, per_bond_offset, "bond_index")
    assert torch.equal(out, bond_index + per_bond_offset.unsqueeze(0))
    assert out.tolist() == [[10, 11, 22], [11, 12, 20]]


def test_integer_meta_scalars_stay_integral():
    """A task code must survive collation as an integer.

    ``TaskRouter.route`` refuses a floating task label rather than truncate one
    silently, so casting every ``meta`` scalar to float would break multi-task
    routing for every frame-fed batch.
    """
    batch = collate_molecules(
        [
            frame(Z=[8, 1], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], energy=-1.5, task=3),
            frame(Z=[6, 1], pos=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], energy=-2.5, task=3),
        ]
    )
    assert not batch["graphs", "task"].is_floating_point()
    assert batch["graphs", "task"].tolist() == [3, 3]
    assert batch["graphs", "energy"].is_floating_point()


def test_a_frame_box_becomes_graphs_cell_with_the_lattice_vectors_as_rows():
    """``frame.box`` → ``graphs.cell (B, 3, 3)``, rows = lattice vectors, at the frame's precision.

    molpy's ``Box.matrix`` holds the lattice vectors as *columns*; every
    reader of ``graphs.cell`` (the MD list, the periodic pair terms) computes
    ``fractional @ cell``, so the collate transposes. A triclinic box is the
    only geometry that can tell the two apart.
    """
    from molpy import Box

    box = Box.from_lengths_angles(np.array([10.0, 12.0, 14.0]), np.array([90.0, 90.0, 60.0]))
    rows = np.asarray(box.matrix).T  # the helper takes rows and rebuilds the same Box
    batch = collate_molecules(
        [
            frame(Z=[1, 8], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], box=rows),
            frame(Z=[6], pos=[[2.0, 2.0, 2.0]], box=rows),
        ]
    )

    cell = batch["graphs", "cell"]
    assert cell.shape == (2, 3, 3)
    assert cell.dtype is torch.float64
    assert torch.equal(cell[0], torch.tensor(rows))
    assert torch.equal(cell[1], torch.tensor(rows))
    assert torch.allclose(
        cell[0].norm(dim=1), torch.tensor([10.0, 12.0, 14.0], dtype=torch.float64)
    )


def test_frames_carry_a_box_all_or_none():
    """A box on some frames and not others is refused by name; no boxes, no cell."""
    with_box = frame(Z=[1], pos=[[0.0, 0.0, 0.0]], box=np.eye(3) * 5.0)
    without = frame(Z=[1], pos=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=r"frames \[1\] do not"):
        collate_molecules([with_box, without])

    batch = collate_molecules([without, without])
    assert ("graphs", "cell") not in batch.keys(include_nested=True)


def test_float_leaves_keep_the_frames_precision_unless_a_dtype_is_asked_for():
    """Compute floats follow ``dtype=``; graph totals stay IEEE-754 binary64.

    The collate narrows nothing on its own for ``dtype=None``. When a pipeline
    asks for float32, positions / mass / forces take that dtype, but
    ``graphs.energy`` keeps the Frame's rest precision (a Python float is a
    double). Integer leaves stay long either way. The typed view shares storage.
    """
    frames = [
        frame(
            Z=[1, 8],
            pos=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            forces=[[1.0] * 3] * 2,
            energy=-1.5,
            task=2,
        ),
    ]
    frames[0]["atoms"]["mass"] = np.array([1.008, 15.999])

    kept = collate_molecules(frames)
    assert kept["atoms", "pos"].dtype is torch.float64
    assert kept["atoms", "mass"].dtype is torch.float64
    assert kept["atoms", "forces"].dtype is torch.float64
    assert kept["graphs", "energy"].dtype is torch.float64
    assert kept["graphs", "task"].dtype is torch.long
    assert kept["atoms", "pos"][1, 2].item() == 0.3

    cast = collate_molecules(frames, dtype=torch.float32)
    assert cast["atoms", "pos"].dtype is torch.float32
    assert cast["atoms", "mass"].dtype is torch.float32
    assert cast["atoms", "forces"].dtype is torch.float32
    assert cast["graphs", "energy"].dtype is torch.float64
    assert cast["graphs", "task"].dtype is torch.long
    assert cast["atoms", "Z"].dtype is torch.long

    wide = collate_molecules(frames, dtype=torch.float64)
    assert wide["atoms", "pos"].dtype is torch.float64
    assert wide["graphs", "energy"].dtype is torch.float64

    view = TensorFrame.from_tensordict(cast)
    assert view["atoms", "pos"].data_ptr() == cast["atoms", "pos"].data_ptr()
    assert view["graphs", "energy"].data_ptr() == cast["graphs", "energy"].data_ptr()

    collated = MoleculeCollator(dtype=torch.float32)(frames)
    assert collated["atoms", "pos"].dtype is torch.float32
    assert collated["graphs", "energy"].dtype is torch.float64


def test_graph_e0_stays_float64_when_a_compute_dtype_is_asked_for():
    """Meta ``e0`` is a graph total: same binary64 skip as ``graphs.energy``."""
    batch = collate_molecules(
        [
            frame(
                Z=[1, 8],
                pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                energy=-1.5,
                e0=-2.0,
            ),
        ],
        dtype=torch.float32,
    )
    assert batch["atoms", "pos"].dtype is torch.float32
    assert batch["graphs", "energy"].dtype is torch.float64
    assert batch["graphs", "e0"].dtype is torch.float64
    assert batch["graphs", "e0"].tolist() == [-2.0]


def test_unprotected_graph_floats_follow_the_asked_dtype():
    """Meta floats that are not graph totals (e.g. ``U0``) take ``dtype=``."""
    batch = collate_molecules(
        [frame(Z=[1], pos=[[0.0, 0.0, 0.0]], energy=-1.5, U0=1.5)],
        dtype=torch.float32,
    )
    assert batch["graphs", "U0"].dtype is torch.float32
    assert batch["graphs", "energy"].dtype is torch.float64


def test_atoms_energy_follows_dtype_because_it_is_not_a_graph_total():
    """An atoms extra named ``energy`` is ``ATOMIC_ENERGY_KEY``, not protected."""
    frames = [
        frame(
            Z=[1, 8],
            pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            energy=-1.5,
        ),
    ]
    frames[0]["atoms"]["energy"] = np.array([0.25, 0.75], dtype=np.float64)
    batch = collate_molecules(frames, dtype=torch.float32)
    assert batch["atoms", "energy"].dtype is torch.float32
    assert batch["graphs", "energy"].dtype is torch.float64


def test_kr_scale_energy_survives_float32_collate_as_float64():
    """Kr-scale DFT totals are not a binary32 quantity.

    IEEE-754 binary32 ulp at ``|E| = 74800 eV`` (``2^16 <= |E| < 2^17``) is
    ``2^(16-23) = 0.0078125 eV`` — several meV, larger than OMol's meV/atom
    residual. Binary64 is the label dtype.

    Provenance: IEEE Std 754-2019; Levine et al. 2025 arXiv:2505.08762
    (OMol25, errors in meV / meV atom^-1, totals in eV); FAIR fairchem
    ``DoubleTensor`` at commit ``41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9``
    (offline pin — this test does not import fairchem).
    """
    e_raw_ev = -74800.123456789
    batch = collate_molecules(
        [frame(Z=[36], pos=[[0.0, 0.0, 0.0]], energy=e_raw_ev)],
        dtype=torch.float32,
    )
    energy = batch["graphs", "energy"]
    assert energy.dtype is torch.float64
    assert torch.equal(energy, torch.tensor([e_raw_ev], dtype=torch.float64))
    assert 2**16 <= 74800.0 < 2**17
    assert 2 ** (16 - 23) == 0.0078125
    as_f32 = float(torch.tensor(e_raw_ev, dtype=torch.float32))
    assert abs(as_f32 - e_raw_ev) > 0.001
    assert batch["atoms", "pos"].dtype is torch.float32


def test_string_atom_columns_are_skipped():
    """OMol25 writes ``atoms.element`` as symbols; that is not a tensor leaf."""
    mol = frame(
        Z=[8, 1, 1],
        pos=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        energy=-76.4,
    )
    mol["atoms"]["element"] = np.array(["O", "H", "H"])
    batch = collate_molecules([mol])
    assert "element" not in batch["atoms"].keys()
    assert batch["atoms", "Z"].tolist() == [8, 1, 1]


def test_velocity_columns_stack_into_atoms_vel():
    """``vx``/``vy``/``vz`` become ``atoms.vel (N, 3)``, like ``fx``/``fy``/``fz`` → ``forces``."""
    vel = [[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]
    batch = collate_molecules([frame(Z=[1, 8], pos=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vel=vel)])

    assert torch.equal(batch["atoms", "vel"], torch.tensor(vel, dtype=torch.float64))
    for component in ("vx", "vy", "vz"):
        assert ("atoms", component) not in batch.keys(include_nested=True)
