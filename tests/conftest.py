"""Generic symmetry-test helpers for molecular ML modules.

Three physical symmetries every encoder + property head must satisfy:

    * **Translation invariance** — features / energy / scalar moments
      unchanged under a rigid shift of all positions; vector moments
      (μ, F) are also invariant *for neutral systems* because their
      origin-dependent piece (Σ q_i) vanishes.
    * **Rotation invariance / equivariance** — scalar features and
      energy are invariant; vector outputs (forces, dipoles) rotate
      with the same rotation matrix.
    * **Permutation equivariance** — relabelling atoms permutes
      per-atom outputs accordingly; per-graph scalars / vectors
      are invariant.

This module provides three things:

    1. :func:`make_graph_batch` — build a ``TensorDict`` from raw
       tensors, computing ``pair_diff`` / ``pair_dist`` from positions.
    2. :func:`translate_graph` / :func:`rotate_graph` /
       :func:`permute_graph` — the three input transforms.
    3. :func:`recompute_edge_geometry` — call inside a forward pass so
       autograd can trace ``∂E/∂pos`` for force / equivariance tests.
    4. :func:`random_rotation_matrix` / :func:`rotation_matrix_x` /
       :func:`rotation_matrix_y` / :func:`rotation_matrix_z` /
       :func:`rotate_vectors` / :func:`rotate_irreps_features_simple` —
       SO(3) helpers for equivariance tests on raw tensors and ``ir_mul``
       irreps features (scalars and vectors only).
    5. :func:`place_edges` and the **butane**
       fixture family — one molecule's geometry and topology, and the
       two-molecule periodic box (:func:`butane_pair_frame` →
       :func:`butane_pair_batch`, with
       :func:`butane_pair_class_i_forcefield`) that the assembled-energy
       symmetry gates in ``test_molpot`` and the NVE drift gate in
       ``test_molix`` both run on. The box is built as a molpy ``Frame``
       and reaches the batch only through ``collate_molecules``, which is
       the one Frame → batch conversion molnex has. They live here, at
       the nearest conftest common to both, rather than in either
       package's leaf conftest: a fixture two test packages share must
       not make one of them import the other's private test scaffolding.

The transforms are intentionally small — concrete tests live next to
the module they exercise (encoders in ``test_molzoo``, heads in
``test_molpot``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cuequivariance as cue
import numpy as np
import pytest
import torch
from molpy import Box, Frame
from tensordict import TensorDict

from molix import config
from molix.data import TotalCharge
from molix.data.collate import collate_molecules
from molix.neighbor import NeighborList

if TYPE_CHECKING:
    from molpot.forcefield import ForceField


@pytest.fixture
def fp64():
    """Build models at fp64 for the duration of one test.

    cuEquivariance fixes its working precision at construction, so a model
    built under the default fp32 and then ``.double()``-d still computes in
    float32. Tests that need fp64 build under this fixture instead.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


def frame(
    *,
    Z,
    pos,
    forces=None,
    vel=None,
    box=None,
    bonds=None,
    bond_types=None,
    terms=None,
    energy=None,
    **meta,
):
    """Build one molpy ``Frame`` — the only data structure a fixture hands the collate.

    Shared by ``test_molix`` and ``test_molpot`` (and the builders below), so it
    lives here, at the nearest conftest common to both.

    Args:
        Z: Atomic numbers ``(N,)``.
        pos: Positions ``(N, 3)``.
        forces: Optional per-atom forces ``(N, 3)``, stored as ``fx``/``fy``/``fz``.
        vel: Optional per-atom velocities ``(N, 3)`` Å/fs, stored as ``vx``/``vy``/``vz``.
        box: Optional lattice ``(3, 3)`` in Å with the lattice vectors as
            **rows** (the ``graphs.cell`` convention); stored as the frame's
            molpy ``Box``, whose matrix holds them as columns.
        bonds: Optional ``(atomi, atomj)`` pair of index sequences.
        bond_types: Optional integer bond types, stored as ``type_id`` because
            molpy's frame schema reserves ``type`` for a string column.
        terms: Optional ``{block: {column: values}}`` for angles / propers /
            impropers, with molecule-local atom indices.
        energy: Optional potential energy, stored as the molrec per-step key
            ``pe`` — the collate turns it into ``graphs.energy``.
        **meta: Other per-frame scalars — each becomes ``graphs.<key>``.

    Returns:
        The frame.
    """
    z = np.asarray(Z)
    xyz = np.asarray(pos, dtype=np.float64).reshape(len(z), 3)
    f = Frame(meta=dict(meta) if energy is None else {"pe": energy, **meta})
    columns = {"atomic_number": z, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if forces is not None:
        fxyz = np.asarray(forces, dtype=np.float64).reshape(len(z), 3)
        columns |= {"fx": fxyz[:, 0], "fy": fxyz[:, 1], "fz": fxyz[:, 2]}
    if vel is not None:
        vxyz = np.asarray(vel, dtype=np.float64).reshape(len(z), 3)
        columns |= {"vx": vxyz[:, 0], "vy": vxyz[:, 1], "vz": vxyz[:, 2]}
    f["atoms"] = columns
    if box is not None:
        f.box = Box(np.asarray(box, dtype=np.float64).reshape(3, 3).T)
    if bonds is not None:
        block = {"atomi": np.asarray(bonds[0]), "atomj": np.asarray(bonds[1])}
        if bond_types is not None:
            block["type_id"] = np.asarray(bond_types, dtype=np.int64)
        f["bonds"] = block
    for name, block in (terms or {}).items():
        f[name] = {col: np.asarray(values) for col, values in block.items()}
    return f


def make_graph_batch(
    pos: torch.Tensor,
    Z: torch.Tensor,
    pair_index: torch.Tensor,
    batch: torch.Tensor,
    *,
    graphs: dict[str, torch.Tensor] | None = None,
    shifts: torch.Tensor | None = None,
) -> TensorDict:
    """Build a ``TensorDict`` from raw tensors.

    Args:
        pos: ``(N, 3)`` atomic positions.
        Z: ``(N,)`` atomic numbers.
        pair_index: ``(E, 2)`` source/target index pairs.
        batch: ``(N,)`` graph membership index per atom.
        graphs: Optional per-graph fields (e.g. ``{"total_charge": tensor}``)
            written to the ``"graphs"`` sub-tensordict. ``num_atoms`` is
            auto-derived from ``batch`` and always present.
        shifts: Optional PBC shift vectors ``(E, 3)`` (``unit_shifts @ cell``),
            written to ``edges.shifts`` and folded additively into
            ``pair_diff``. ``S_ij`` is constant w.r.t. ``pos``, so the
            position gradient of the edge geometry is unchanged.

    Returns:
        A fully-formed ``TensorDict`` with
        ``pair_diff = pos[dst] - pos[src] (+ shifts)`` and
        ``pair_dist = ‖pair_diff‖`` recomputed from ``pos``.
    """
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    if shifts is not None:
        pair_diff = pair_diff + shifts
    pair_dist = pair_diff.norm(dim=-1).clamp(min=1e-6)
    n_atoms = pos.shape[0]
    n_pairs = pair_index.shape[0]
    n_graphs = int(batch.max().item()) + 1 if n_atoms > 0 else 0

    num_atoms_per_graph = torch.zeros(n_graphs, dtype=torch.long)
    num_atoms_per_graph.scatter_add_(0, batch, torch.ones_like(batch))
    graph_data = TensorDict(num_atoms=num_atoms_per_graph, batch_size=[n_graphs])
    if graphs is not None:
        for k, v in graphs.items():
            graph_data[k] = v

    edge_data = TensorDict(
        pair_index=pair_index,
        pair_diff=pair_diff,
        pair_dist=pair_dist,
        batch_size=[n_pairs],
    )
    if shifts is not None:
        edge_data["shifts"] = shifts

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch, batch_size=[n_atoms]),
        pairs=edge_data,
        graphs=graph_data,
        batch_size=[],
    )


def translate_graph(batch: TensorDict, t: torch.Tensor) -> TensorDict:
    """Shift all atomic positions by ``t``. Edge geometry recomputes from pos.

    ``edges.shifts`` (if present) is carried unchanged: a rigid translation
    does not change the box, so ``S = n · h`` is untouched.
    """
    pos = batch["atoms", "pos"] + t
    extras = _extract_graph_extras(batch)
    return make_graph_batch(
        pos=pos,
        Z=batch["atoms", "Z"],
        pair_index=batch["pairs", "pair_index"],
        batch=batch["atoms", "batch"],
        graphs=extras,
        shifts=_extract_shifts(batch),
    )


def rotate_graph(batch: TensorDict, R: torch.Tensor) -> TensorDict:
    """Rotate all atomic positions by ``R`` (3×3 rotation matrix).

    ``edges.shifts`` (if present) rotates with the positions: ``S = n · h``
    is a lattice vector, so rotating the box rotates ``S`` by the same ``R``
    while the integer image ``n`` stays put.
    """
    pos = batch["atoms", "pos"] @ R.T
    extras = _extract_graph_extras(batch)
    shifts = _extract_shifts(batch)
    return make_graph_batch(
        pos=pos,
        Z=batch["atoms", "Z"],
        pair_index=batch["pairs", "pair_index"],
        batch=batch["atoms", "batch"],
        graphs=extras,
        shifts=None if shifts is None else shifts @ R.T,
    )


def permute_graph(batch: TensorDict, perm: torch.Tensor) -> TensorDict:
    """Relabel atoms by ``perm``; pair_index is remapped, per-graph fields kept.

    ``edges.shifts`` is per-edge and the edge *rows* keep their order (only the
    node labels inside ``pair_index`` are rewritten), so shifts stay aligned
    and are carried unchanged.
    """
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    pos = batch["atoms", "pos"][perm]
    Z = batch["atoms", "Z"][perm]
    batch_idx = batch["atoms", "batch"][perm]
    pair_index = inv_perm[batch["pairs", "pair_index"]]
    extras = _extract_graph_extras(batch)
    return make_graph_batch(
        pos=pos,
        Z=Z,
        pair_index=pair_index,
        batch=batch_idx,
        graphs=extras,
        shifts=_extract_shifts(batch),
    )


def recompute_edge_geometry(batch: TensorDict) -> TensorDict:
    """Re-derive ``pair_diff`` / ``pair_dist`` from ``pos`` in-place.

    Call this at the start of a pipeline forward when ``pos`` carries
    ``requires_grad`` so autograd can trace ``∂E/∂pos`` for force tests.
    ``edges.shifts`` (if present) is folded back in, exactly as
    :func:`make_graph_batch` does.
    """
    pos = batch["atoms", "pos"]
    pair_index = batch["pairs", "pair_index"]
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    shifts = _extract_shifts(batch)
    if shifts is not None:
        pair_diff = pair_diff + shifts
    pair_dist = pair_diff.norm(dim=-1).clamp(min=1e-6)
    batch["pairs", "pair_diff"] = pair_diff
    batch["pairs", "pair_dist"] = pair_dist
    return batch


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


_RESERVED_GRAPH_KEYS = {"num_atoms"}


def _extract_graph_extras(batch: TensorDict) -> dict[str, torch.Tensor] | None:
    """Lift user-set per-graph fields off ``batch`` so make_graph_batch can
    re-attach them after a transform. ``num_atoms`` is auto-derived and
    therefore skipped."""
    if "graphs" not in batch.keys():
        return None
    graphs = batch["graphs"]
    extras = {k: graphs[k] for k in graphs.keys() if k not in _RESERVED_GRAPH_KEYS}
    return extras or None


def _extract_shifts(batch: TensorDict) -> torch.Tensor | None:
    """Return ``edges.shifts`` if the batch carries PBC shift vectors."""
    if ("pairs", "shifts") not in batch.keys(include_nested=True):
        return None
    return batch["pairs", "shifts"]


# --------------------------------------------------------------------------
# Placed neighbour lists
# --------------------------------------------------------------------------


def place_edges(batch: TensorDict, *, cutoff: float) -> TensorDict:
    """Write ``batch["pairs"]`` with molix's neighbour list (``composable-a0-03b``).

    :class:`molix.data.NeighborList` is the repo's Edge Convention owner and
    the training-side of the two owners named in
    ``.claude/notes/architecture.md`` §Layer roles (the other being
    :class:`molix.md.NeighborList`). ``molpot`` **consumes** a placed ``pairs``
    namespace; it does not enumerate one, so every fixture that reaches a pair
    term has to arrive with a list, exactly as a collated training batch does.

    ``symmetry`` is left at its ``True`` default and that is load-bearing, not
    incidental: the half-pair mode emits ``pair_index[:, 0] > pair_index[:, 1]``,
    and the placed-list liveness rule keeps ``source < target``, so a
    ``symmetry=False`` list would read as **entirely dead** and every pair term
    would silently cost zero. The full bidirectional list carries each pair
    twice and the rule keeps exactly one orientation of each.

    The list is built on a throw-away probe batch and only the resulting
    namespace is moved across, so the caller's ``atoms`` / ``graphs`` /
    topology tensors are never replaced by copies of themselves.

    Args:
        batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` in Å and
            ``atoms.batch`` ``(N,)`` graph membership, optionally carrying
            ``graphs.cell`` — ``(3, 3)`` or ``(B, 3, 3)`` — which switches the
            build to the minimum-image convention.
        cutoff: Build radius in Å. Pass the cutoff the *asking style* declares
            so the live pair set is exactly the set that style would have
            enumerated; where a style declares none (open ``pair:coulomb``
            defaults to ``cutoff=None``, i.e. every pair), pass a radius past
            the fixture's largest separation.

    Returns:
        The same ``batch`` object, with ``pairs`` holding ``pair_index``
        ``(E, 2)``, ``pair_diff`` ``(E, 3)`` (``pos[target] - pos[source]``)
        and ``pair_dist`` ``(E,)``. ``E`` is ``0`` when no pair is inside
        ``cutoff`` — an empty list, not an absent one.
    """
    atoms = batch["atoms"]
    pos = atoms["pos"]
    n_atoms = int(pos.shape[0])
    membership = (
        atoms["batch"] if "batch" in atoms.keys() else torch.zeros(n_atoms, dtype=torch.long)
    )
    cell = None
    if "graphs" in batch.keys() and "cell" in batch["graphs"].keys():
        cell = batch["graphs", "cell"]
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)

    probe = TensorDict(
        {"atoms": TensorDict({"pos": pos, "batch": membership}, batch_size=[n_atoms])},
        batch_size=[],
    )
    if cell is not None:
        probe["graphs"] = TensorDict({"cell": cell}, batch_size=[int(cell.shape[0])])
    NeighborList(cutoff=cutoff, pbc=cell is not None).execute(probe)
    batch["pairs"] = probe["pairs"]
    return batch


# --------------------------------------------------------------------------
# Butane (n-C4H10) — the shared molecular fixture
# --------------------------------------------------------------------------


#: Atom order of the butane fixture: ``C0 C1 C2 C3`` then the hydrogens
#: ``H4 H5 H6`` on ``C0``, ``H7 H8`` on ``C1``, ``H9 H10`` on ``C2`` and
#: ``H11 H12 H13`` on ``C3``.
BUTANE_N_ATOMS = 14

#: The 13 covalent bonds of n-butane, ``(atomi, atomj)`` with ``i < j``.
#: Angles (24) and proper torsions (27) are *derived* from this list by
#: :func:`butane_angles` / :func:`butane_propers`, so the topology has a single
#: source of truth.
BUTANE_BONDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (1, 7),
    (1, 8),
    (2, 9),
    (2, 10),
    (3, 11),
    (3, 12),
    (3, 13),
)

#: A fixed all-atom butane geometry (Å). Every one of the 91 pairs lies between
#: 1.09 Å and 6.04 Å, so a 10 Å neighbour list keeps all of them — which is the
#: list the exclusion-scale golden in
#: ``tests/test_molpot/test_forcefield/conftest.py`` was captured against.
BUTANE_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.000000, 0.000000, 0.000000),
    (1.268427, 0.855565, 0.000000),
    (2.536855, 0.000000, 0.000000),
    (3.805282, 0.855565, 0.000000),
    (-0.974926, -0.487463, 0.000000),
    (-0.291315, -0.582630, 0.873944),
    (-0.291315, -0.582630, -0.873944),
    (1.268427, 1.460188, 0.906935),
    (1.268427, 1.460188, -0.906935),
    (2.536855, -0.604623, 0.906935),
    (2.536855, -0.604623, -0.906935),
    (4.780208, 1.343028, 0.000000),
    (4.096597, 1.438195, 0.873944),
    (4.096597, 1.438195, -0.873944),
)


def butane_neighbors() -> dict[int, list[int]]:
    """Adjacency of :data:`BUTANE_BONDS`, each neighbour list sorted."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(BUTANE_N_ATOMS)}
    for i, j in BUTANE_BONDS:
        adjacency[i].append(j)
        adjacency[j].append(i)
    return {atom: sorted(neighbors) for atom, neighbors in adjacency.items()}


def butane_angles() -> list[tuple[int, int, int]]:
    """The 24 ``(i, j, k)`` angles of butane — every neighbour pair of a centre."""
    adjacency = butane_neighbors()
    triples: list[tuple[int, int, int]] = []
    for centre in range(BUTANE_N_ATOMS):
        neighbors = adjacency[centre]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                triples.append((neighbors[a], centre, neighbors[b]))
    return triples


def butane_propers() -> list[tuple[int, int, int, int]]:
    """The 27 ``(i, j, k, l)`` proper torsions of butane, ``j-k`` the central bond."""
    adjacency = butane_neighbors()
    quads: list[tuple[int, int, int, int]] = []
    for j, k in BUTANE_BONDS:
        for i in adjacency[j]:
            if i == k:
                continue
            for end in adjacency[k]:
                if end == j:
                    continue
                quads.append((i, j, k, end))
    return quads


#: All-atom alkane charges (e), well depths (kcal/mol) and sizes (Å) of one
#: butane: CH3 carbon −0.18, CH2 carbon −0.12, every hydrogen +0.06 — a
#: neutral molecule, so a box of whole molecules satisfies ``Σ_g q = 0`` by
#: construction and no background charge is doing hidden work in the Ewald
#: sum. Alkane-shaped values, not a published parameter set (see
#: :func:`butane_pair_class_i_forcefield`).
BUTANE_Q: tuple[float, ...] = (-0.18, -0.12, -0.12, -0.18) + (0.06,) * 10
BUTANE_EPSILON: tuple[float, ...] = (0.066,) * 4 + (0.030,) * 10
BUTANE_SIGMA: tuple[float, ...] = (3.50,) * 4 + (2.50,) * 10

#: Cubic edge (Å) of the two-molecule periodic box. Every cutoff below has to
#: stay at or under the minimum-image limit ``L/2 = 10``.
BUTANE_PAIR_BOX = 20.0
#: Where the first copy of :data:`BUTANE_POSITIONS` is placed (Å). The raw
#: fixture has negative coordinates; shifting it here keeps both molecules
#: strictly inside ``[0, L)`` so the reference configuration is unwrapped and a
#: test that *does* wrap is visibly doing something.
BUTANE_PAIR_ORIGIN = (4.0, 4.0, 4.0)
#: Displacement of the second copy (Å). The two molecules interlock across the
#: 8 Å cutoff (closest intermolecular contact ≈ 2.4 Å, largest live pair
#: ≈ 7.7 Å) without overlapping, so the intermolecular LJ and real-space
#: Coulomb sums are both non-trivial.
BUTANE_PAIR_OFFSET = (6.0, 6.0, 6.0)
#: ``lj/cut`` cutoff and quintic switching window (Å). The window is inside the
#: box (``r_c = 8 < L/2 = 10``), so the taper is a property of the pair term
#: rather than of the minimum-image convention running out of room.
BUTANE_PAIR_LJ_CUTOFF = 8.0
BUTANE_PAIR_LJ_SWITCH_START = 7.0
#: Ewald splitting width ``σ`` (Å) and real-space cutoff ``r_c`` (Å) of the
#: periodic Coulomb sum. ``erfc(r_c / σ√2) = erfc(2.83) ≈ 6e-5``, so the
#: real-space half is converged and what the reciprocal half has to supply is
#: exactly the smooth remainder.
BUTANE_PAIR_SMEARING = 2.0
BUTANE_PAIR_REAL_SPACE_CUTOFF = 8.0
#: Shortest reciprocal wavelength resolved by the ``"ewald"`` sum (Å) — the
#: value the rocksalt Madelung fixture in ``test_coulomb.py`` runs at. With
#: ``σ = 2 Å`` the Gaussian factor at the shortest kept wavelength is
#: ``exp(−σ²k²/2) = e^−79``, so the k-sum is converged far past anything
#: asserted on this box.
BUTANE_PAIR_LR_WAVELENGTH = 0.5
#: The same, for the NVE rollout (Å). The explicit k-sum costs ``(L / λ)³``
#: vectors per force evaluation, so 0.5 Å is 8x the work of 1.0 Å and a
#: 200-step rollout feels it; 1.0 Å still keeps ``exp(−σ²k²/2) = e^−79`` at the
#: shortest wavelength dropped, i.e. converged past every digit either gate
#: reads. Two numbers rather than one because a single-point gate and a
#: trajectory buy accuracy at very different prices, and the trajectory's own
#: assertion (``rel_drift < 1e-3``) is nowhere near this resolution.
BUTANE_PAIR_LR_WAVELENGTH_MD = 1.0
#: Target mesh spacing (Å) handed to the ``"pme"`` spelling of the same style,
#: so ``calculator`` is the *only* difference between the two assemblies.
#: ``get_ns_mesh`` turns it into ``2^ceil(log2(2L/h + 1)) = 64`` points per
#: axis (an effective 0.31 Å grid), whose total electrostatic energy sits
#: 7e-5 kcal/mol from the Ewald sum of the same box — a converged mesh by
#: PME's own standards, and the repo default
#: (``PairStyleDefaults.mesh_spacing``).
BUTANE_PAIR_MESH_SPACING = 1.0
#: Build radius (Å) the pair fixture's list is expected at: the larger of the
#: two pair-style radii (they are equal here), so both nonbonded terms read the
#: rows they are allowed to and neither is starved by the list. It is not
#: passed anywhere — :meth:`~molpot.forcefield.ForceField.neighbor_list`
#: resolves the radius from the field's own styles (finding **F21**) — so this
#: is the number that resolution has to land on, and
#: :func:`butane_pair_batch` checks it.
BUTANE_PAIR_LIST_CUTOFF = 8.0
#: 1-2 / 1-3 / 1-4 nonbonded scales, hand-set: directly bonded pairs and the
#: two ends of an angle are excluded outright, 1-4 pairs are halved for both
#: the Lennard-Jones and the Coulomb kind.
BUTANE_PAIR_SCALES = (0.0, 0.0, 0.5)

#: Harmonic bond and angle force constants (kcal/mol/Å², kcal/mol/rad²) and the
#: offsets that place their equilibria off the reference geometry (Å, rad).
#:
#: The equilibria are the reference geometry's **own** lengths and angles less
#: a fixed offset, rather than tabulated values (``r0 = 1.529 / 1.090``,
#: ``θ0 = 1.911`` for sp3 carbon). :data:`BUTANE_POSITIONS` is a constructed geometry whose
#: terminal methyls are not tetrahedral (``1-0-4`` opens to 172.6° and
#: ``4-0-5`` closes to 61.4°), so tabulated angle equilibria would load 342
#: kcal/mol of strain into the start configuration and a short NVE rollout
#: would be measuring that relaxation instead of the integrator. The offset
#: keeps every bonded term live — non-zero energy *and* non-zero gradient, so a
#: symmetry gate can actually see it — while leaving the fixture near its
#: bonded minimum.
BUTANE_PAIR_BOND_K = 300.0
BUTANE_PAIR_BOND_OFFSET = 0.03
BUTANE_PAIR_ANGLE_K = 60.0
BUTANE_PAIR_ANGLE_OFFSET = 0.05
#: Proper-torsion barriers (kcal/mol), periodicities and phases (rad) of the
#: Fourier form ``Σ_m k_m [1 + cos(n_m φ − γ_m)]``. The barriers are invented
#: — alkane-scaled, not fitted. The **phases** are not: every one is ``0`` or
#: ``π``, which is what makes the torsion term achiral, since ``cos(nφ − γ)``
#: is even in ``φ`` for exactly those two values and so ``E(−φ) = E(φ)``.
BUTANE_PAIR_TORSION_K = (0.7, 0.1, 0.3)
BUTANE_PAIR_TORSION_PERIODICITY = (1, 2, 3)
BUTANE_PAIR_TORSION_PHASE = (0.0, math.pi, 0.0)

#: Atomic masses (amu) keyed by the fixture's two elements — carbon and
#: hydrogen — for the MD side of the fixture.
BUTANE_PAIR_MASS = {6: 12.011, 1: 1.008}

#: Column names of the pair fixture's **valence-term** namespaces, in index
#: order. ``impropers`` is absent: n-butane has no improper centre. ``bonds``
#: is absent for a different reason — ``collate_molecules`` turns a frame's
#: bonds block into the ``(2, N)`` COO ``bonds.bond_index`` rather than into
#: 1-D columns (the anti-alias shape of CLAUDE.md § pair_index vs bond_index),
#: so the batch's bonds are addressed as one tensor and not through this map.
BUTANE_PAIR_TOPOLOGY_COLUMNS: dict[str, tuple[str, ...]] = {
    "angles": ("atomi", "atomj", "atomk"),
    "propers": ("atomi", "atomj", "atomk", "atoml"),
}


def butane_pair_positions() -> torch.Tensor:
    """Two copies of :data:`BUTANE_POSITIONS`, ``(28, 3)`` fp64 in Å.

    Copy 0 sits at :data:`BUTANE_PAIR_ORIGIN`, copy 1 a further
    :data:`BUTANE_PAIR_OFFSET` away. Atom ``i`` of copy 1 is atom ``i + 14``,
    which is the offset every topology column of :func:`butane_pair_topology`
    is built with.
    """
    base = torch.tensor(BUTANE_POSITIONS, dtype=torch.float64) + torch.tensor(
        BUTANE_PAIR_ORIGIN, dtype=torch.float64
    )
    return torch.cat([base, base + torch.tensor(BUTANE_PAIR_OFFSET, dtype=torch.float64)])


def butane_pair_topology() -> dict[str, list[tuple[int, ...]]]:
    """The pair's ``bonds`` / ``angles`` / ``propers`` rows as index tuples.

    Each namespace is the single molecule's rows followed by the same rows with
    every index shifted by :data:`BUTANE_N_ATOMS` — 26 bonds, 48 angles and 54
    proper torsions.
    """
    shifted: dict[str, list[tuple[int, ...]]] = {}
    for name, rows in (
        ("bonds", [tuple(b) for b in BUTANE_BONDS]),
        ("angles", [tuple(a) for a in butane_angles()]),
        ("propers", [tuple(p) for p in butane_propers()]),
    ):
        shifted[name] = list(rows) + [
            tuple(index + BUTANE_N_ATOMS for index in row) for row in rows
        ]
    return shifted


def _butane_pair_equilibria() -> tuple[list[float], list[float]]:
    """Reference bond lengths (Å) and angles (rad) of the pair's topology rows."""
    pos = torch.tensor(BUTANE_POSITIONS, dtype=torch.float64)
    lengths = [float((pos[j] - pos[i]).norm()) for i, j in BUTANE_BONDS]
    angles: list[float] = []
    for i, j, k in butane_angles():
        left, right = pos[i] - pos[j], pos[k] - pos[j]
        cosine = float((left @ right) / (left.norm() * right.norm()))
        angles.append(math.acos(cosine))
    return lengths + lengths, angles + angles


def butane_pair_class_i_forcefield(
    *,
    calculator: str = "ewald",
    lr_wavelength: float = BUTANE_PAIR_LR_WAVELENGTH,
) -> "ForceField":
    """A hand-built Class-I model of the butane pair in its box.

    Harmonic bonds and angles, a periodic proper-torsion Fourier series,
    switched ``lj/cut`` and a periodic ``pair:coulomb`` — every Class-I
    contribution live at once, which is what makes the assembled energy worth
    putting a symmetry gate on. 1-2 / 1-3 pairs are excluded and 1-4 pairs are
    halved (:data:`BUTANE_PAIR_SCALES`) for both the Lennard-Jones and the
    Coulomb kind.

    **Not OPLS-AA, and deliberately not claiming to be.** The charges and
    Lennard-Jones parameters are alkane-shaped, but the bond and angle
    equilibria are read off this fixture's own geometry
    (:func:`_butane_pair_equilibria`), the torsion barriers are invented, and
    the exclusion scales are set by hand rather than by
    :class:`~molpot.forcefield.container.OPLSAAForceField`. Nothing here is a
    published parameter set and no test may read it as one: it is a *shape* —
    the full Class-I functional form with every term non-zero — assembled so
    the gates have something to be invariant about. Parameter fidelity to a
    published field is a different question with a different fixture.

    **Why Ewald and not the PME default.** A mesh method is anchored to the
    *box*, not to the atoms: the B-spline charge assignment interpolates onto a
    grid whose points are fixed in the cell, so the interpolation error is a
    function of where the atoms sit relative to that grid. Translate the whole
    contents of the box and the exact reciprocal sum does not move — every
    ``|S(k)|²`` is translation invariant — but the mesh estimate does, here by
    **7.4e-5 kcal/mol** for a random shift, five orders above the 1e-10 the
    symmetry gates hold the model to. No mesh spacing fixes that; it is a
    property of the estimator, not of the assembled model. The explicit
    k-vector sum has no grid and reproduces the same transform to 7e-14, so it
    is what the exactness gates run on. That this is only an *estimator*
    difference — that the mesh and the k-sum agree on the physics to their
    declared accuracy — is pinned separately, by the rocksalt Madelung test in
    ``test_coulomb.py``, which asserts the literature constant through both
    spellings.

    Args:
        calculator: Reciprocal-space algorithm of the periodic Coulomb sum —
            ``"ewald"`` (every k-vector summed explicitly; the default here)
            or ``"pme"`` (the FFT mesh, and the repo default under a cell).
            The two assemblies differ in nothing else, so an energy that moves
            between them moved by the mesh's own interpolation error.
        lr_wavelength: Shortest reciprocal wavelength the ``"ewald"`` sum
            resolves, in Å. The cost of the explicit k-sum is
            ``(L / λ)³`` vectors *per force evaluation*, so a rollout may
            legitimately ask for a coarser one than a single-point gate does;
            :data:`BUTANE_PAIR_LR_WAVELENGTH_MD` is the value the NVE gate
            passes. Both are converged far past what they assert — see the
            constant's note.

    Returns:
        The unbound :class:`molpot.forcefield.ForceField`, in LAMMPS ``real``
        units (kcal/mol, Å, e).
    """
    from molpot.forcefield import ForceField

    topology = butane_pair_topology()
    lengths, angles = _butane_pair_equilibria()
    n_propers = len(topology["propers"])
    forcefield = (
        ForceField(name="butane-pair", units="real")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_dihedralstyle("periodic")
        .def_pairstyle(
            "lj/cut",
            cutoff=BUTANE_PAIR_LJ_CUTOFF,
            switch_start=BUTANE_PAIR_LJ_SWITCH_START,
        )
        .def_pairstyle(
            "coulomb",
            calculator=calculator,
            smearing=BUTANE_PAIR_SMEARING,
            real_space_cutoff=BUTANE_PAIR_REAL_SPACE_CUTOFF,
            mesh_spacing=BUTANE_PAIR_MESH_SPACING,
            lr_wavelength=lr_wavelength,
        )
    )
    forcefield.set_special_bonds(BUTANE_PAIR_SCALES, BUTANE_PAIR_SCALES)
    return forcefield.with_params(
        bonds={
            "k": torch.full((len(topology["bonds"]),), BUTANE_PAIR_BOND_K, dtype=torch.float64),
            "r0": torch.tensor(lengths, dtype=torch.float64) - BUTANE_PAIR_BOND_OFFSET,
        },
        angles={
            "k": torch.full((len(topology["angles"]),), BUTANE_PAIR_ANGLE_K, dtype=torch.float64),
            "theta0": torch.tensor(angles, dtype=torch.float64) - BUTANE_PAIR_ANGLE_OFFSET,
        },
        propers={
            "k": torch.tensor([BUTANE_PAIR_TORSION_K] * n_propers, dtype=torch.float64),
            "periodicity": torch.tensor(BUTANE_PAIR_TORSION_PERIODICITY, dtype=torch.long),
            "phase": torch.tensor([BUTANE_PAIR_TORSION_PHASE] * n_propers, dtype=torch.float64),
        },
        atoms={
            "charge": torch.tensor(BUTANE_Q * 2, dtype=torch.float64),
            "epsilon": torch.tensor(BUTANE_EPSILON * 2, dtype=torch.float64),
            "sigma": torch.tensor(BUTANE_SIGMA * 2, dtype=torch.float64),
        },
    )


def butane_pair_frame(positions: torch.Tensor | None = None) -> "Frame":
    """The pair as one molpy ``Frame`` — this fixture's data at rest.

    **One frame, not two.** A frame per molecule collates to a batch of two
    graphs, and a periodic Coulomb sum is defined per *lattice*: with
    ``graphs.cell`` set, :class:`~molpot.forcefield.Potentials` refuses
    ``num_graphs > 1`` outright (ledger finding **F20**) rather than price the
    whole box in graph 0's cell and hand that number to both graphs. The
    periodic box is therefore one system, and its second molecule is 14 more
    atoms with the topology of the first shifted by
    :data:`BUTANE_N_ATOMS` — which is exactly what
    :func:`butane_pair_topology` builds.

    ``mass`` rides as a per-atom column, the lattice as the frame's box, a
    zero velocity as ``vx``/``vy``/``vz`` and the declared ``total_charge`` as
    a ``meta`` scalar, so all of them are frame data rather than something a
    fixture bolts onto the batch afterwards.

    Args:
        positions: Atom positions ``(28, 3)`` in Å, defaulting to
            :func:`butane_pair_positions`.

    Returns:
        The frame.
    """
    pos = butane_pair_positions() if positions is None else positions
    numbers = ([6] * 4 + [1] * 10) * 2
    topology = butane_pair_topology()
    built = frame(
        Z=numbers,
        pos=pos.detach().cpu().double().numpy(),
        vel=np.zeros((len(numbers), 3)),
        box=np.eye(3) * BUTANE_PAIR_BOX,
        bonds=(
            [row[0] for row in topology["bonds"]],
            [row[1] for row in topology["bonds"]],
        ),
        terms={
            name: {
                column: [row[axis] for row in topology[name]] for axis, column in enumerate(columns)
            }
            for name, columns in BUTANE_PAIR_TOPOLOGY_COLUMNS.items()
        },
        total_charge=0.0,
    )
    # ``frame()`` takes no per-atom extras (its keywords past ``forces`` are
    # meta scalars), so the mass column is written onto the block it built.
    built["atoms"]["mass"] = np.asarray([BUTANE_PAIR_MASS[z] for z in numbers], dtype=np.float64)
    return built


def butane_pair_batch(positions: torch.Tensor | None = None) -> TensorDict:
    """The pair's post-collate batch: atoms, topology, lattice and placed list.

    Built the way data reaches a model: :func:`butane_pair_frame` is the system
    at rest and ``collate_molecules`` is the only conversion — the topology
    columns, the atomic numbers, the masses, the lattice, the velocities and
    the declared total charge are all the frame's, none of them assembled here
    as tensors, and the batch keeps the frame's fp64.

    One graph, not two: the box is one periodic system (see
    :func:`butane_pair_frame`), so ``atoms.batch`` is all zeros and
    ``graphs.cell`` is the single lattice under ``batch_size=[1]``.
    ``graphs.total_charge`` is the frame's declared zero, normalised to the
    batch float dtype by :class:`~molix.data.TotalCharge` — the task that owns
    that column — rather than written by hand. ``atoms.mass`` (amu) and a zero
    ``atoms.vel`` (Å/fs) are carried too, so the same fixture is the one an MD
    rollout runs — the energy gates and the NVE gate must not be looking at two
    different boxes.

    The list is placed by :meth:`~molpot.forcefield.ForceField.neighbor_list`,
    so its radius is the field's own (finding **F21**) rather than a number
    this fixture picks; that radius is :data:`BUTANE_PAIR_LIST_CUTOFF`, which
    is pinned in ``test_container.py`` beside the other ``neighbor_list`` gates.

    Args:
        positions: Atom positions ``(28, 3)`` in Å, defaulting to
            :func:`butane_pair_positions`. A transformed geometry is passed
            here so the neighbour list is rebuilt for it: the list is a
            function of the positions, and reusing one across a move would be
            asking the energy about connectivity that is no longer there.

    Returns:
        The batch, with ``pairs`` placed at :data:`BUTANE_PAIR_LIST_CUTOFF`
        under the minimum-image convention.
    """
    batch = collate_molecules([butane_pair_frame(positions)])
    neighbors = butane_pair_class_i_forcefield().neighbor_list(pbc=True)
    neighbors.execute(batch)
    TotalCharge().execute(batch)
    return batch


# --------------------------------------------------------------------------
# Rotations for equivariance tests
# --------------------------------------------------------------------------


def random_rotation_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix using QR decomposition.

    Args:
        dtype: Data type for the rotation matrix.

    Returns:
        Random 3x3 rotation matrix with determinant +1.
    """
    # Generate random matrix
    M = torch.randn(3, 3, dtype=dtype)

    # QR decomposition
    Q, R = torch.linalg.qr(M)

    # Ensure determinant is +1 (proper rotation)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def rotation_matrix_z(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around z-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around z-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=dtype)


def rotation_matrix_x(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around x-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around x-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=dtype)


def rotation_matrix_y(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around y-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around y-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=dtype)


def rotate_vectors(vectors: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate vectors using rotation matrix.

    Args:
        vectors: Input vectors with shape (..., 3).
        rot_matrix: 3x3 rotation matrix.

    Returns:
        Rotated vectors with same shape as input.
    """
    return vectors @ rot_matrix.T


def rotate_irreps_features_simple(
    features: torch.Tensor,
    rot_matrix: torch.Tensor,
    irreps: str,
) -> torch.Tensor:
    """Rotate features for simple irreps (scalars and vectors only).

    This is a simplified rotation function that handles l=0 (scalars) and l=1 (vectors).
    For higher-order irreps, use Wigner D matrices.

    Args:
        features: Input features with shape (n_nodes, irreps_dim) in ir_mul layout.
        rot_matrix: 3x3 rotation matrix.
        irreps: Irreps string (e.g., "64x0e + 32x1o").

    Returns:
        Rotated features with same shape as input.
    """
    # Parse irreps
    irreps_obj = cue.Irreps("O3", irreps)

    n_nodes = features.shape[0]
    rotated = features.clone()

    offset = 0
    for mul_irrep in irreps_obj:
        mul = mul_irrep.mul
        l = mul_irrep.ir.l
        dim_l = 2 * l + 1

        if l == 0:
            # Scalars are invariant under rotation
            offset += mul * dim_l
        elif l == 1:
            # Vectors: rotate each vector independently
            # In ir_mul layout: features are ordered as (ir_dim, mul)
            # So for l=1 with mul=M, we have shape (3, M) flattened to (3*M,)
            # which means: [x1, x2, ..., xM, y1, y2, ..., yM, z1, z2, ..., zM]
            vec_block = features[:, offset : offset + mul * 3]  # (n_nodes, 3*mul)
            vec_block_reshaped = vec_block.reshape(n_nodes, 3, mul)  # (n_nodes, 3, mul)

            # Apply rotation to xyz components
            vec_block_rotated = torch.einsum("ij,njm->nim", rot_matrix, vec_block_reshaped)

            # Reshape back
            rotated[:, offset : offset + mul * 3] = vec_block_rotated.reshape(n_nodes, 3 * mul)
            offset += mul * dim_l
        else:
            # Higher-order irreps: for proper equivariance, need Wigner D matrices
            # For now, we just copy them (not equivariant!)
            # This function should only be used for testing scalars and vectors
            offset += mul * dim_l

    return rotated
