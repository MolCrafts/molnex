"""Test utilities wrap common functions in the tests"""

import math
from pathlib import Path
from typing import Optional

import torch

SQRT3 = math.sqrt(3)


def periodic_neighbor_list(
    positions: torch.Tensor,
    box: torch.Tensor,
    cutoff: float,
    full_list: bool,
    periodic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch periodic neighbour list (no external dependency).

    Convention (consumed by :func:`compute_distances` and the calculators):

    * the distance vector for pair ``(i, j)`` with integer shift ``S`` is
      ``pos[j] + S @ box - pos[i]`` and ``d = ‖·‖``;
    * the trivial self pair ``(i, i, S=0)`` is excluded, but self-images
      ``(i, i, S≠0)`` and same-cell cross pairs ``(i, j, S=0)`` are kept;
    * ``d < cutoff`` is strict;
    * the half list (``full_list=False``) keeps one representative per
      ``{(i, j, S), (j, i, -S)}`` class: ``i < j``, or ``i == j`` with ``S``
      lexicographically positive.

    Args:
        positions: ``(N, 3)`` Cartesian coordinates.
        box: ``(3, 3)`` cell with lattice vectors as rows (zeros ⇒ no PBC).
        cutoff: Neighbour cutoff radius.
        full_list: Return both ``(i, j, S)`` and ``(j, i, -S)`` if ``True``.
        periodic: Enumerate periodic images if ``True``.

    Returns:
        ``(pairs, shifts, distances)`` where ``pairs`` is ``(E, 2)`` long with
        source in column 0 / target in column 1, ``shifts`` is ``(E, 3)`` long,
        and ``distances`` is ``(E,)``.
    """
    n = positions.shape[0]
    dev, dt = positions.device, positions.dtype
    use_pbc = periodic and box is not None and torch.count_nonzero(box) > 0

    if use_pbc:
        # Cells needed along each lattice direction to cover the cutoff:
        # interplanar spacing along axis k is 1 / ‖recip[:, k]‖.
        recip = torch.linalg.inv(box)
        heights = 1.0 / torch.linalg.norm(recip, dim=0)
        reps = torch.ceil(cutoff / heights).to(torch.long) + 1
        axes = [torch.arange(-int(reps[k]), int(reps[k]) + 1, device=dev) for k in range(3)]
        shifts = torch.cartesian_prod(*axes).to(dt)
        offsets = shifts @ box
    else:
        shifts = torch.zeros(1, 3, dtype=dt, device=dev)
        offsets = shifts

    n_shift = shifts.shape[0]
    # dvec[s, i, j] = pos[j] + offset[s] - pos[i]
    dvec = (
        positions.view(1, 1, n, 3)
        + offsets.view(n_shift, 1, 1, 3)
        - positions.view(1, n, 1, 3)
    )
    dist = dvec.norm(dim=-1)

    si = torch.arange(n_shift, device=dev).view(n_shift, 1, 1).expand(n_shift, n, n)
    ii = torch.arange(n, device=dev).view(1, n, 1).expand(n_shift, n, n)
    jj = torch.arange(n, device=dev).view(1, 1, n).expand(n_shift, n, n)
    zero_shift = (shifts.abs().sum(-1) == 0).view(n_shift, 1, 1)
    self_pair = (ii == jj) & zero_shift
    mask = (dist > 1e-12) & (dist < cutoff) & (~self_pair)

    i_sel, j_sel = ii[mask], jj[mask]
    s_sel = shifts.to(torch.long)[si[mask]]
    d_sel = dist[mask]

    if not full_list:
        s0, s1, s2 = s_sel[:, 0], s_sel[:, 1], s_sel[:, 2]
        s_pos = (s0 > 0) | ((s0 == 0) & (s1 > 0)) | ((s0 == 0) & (s1 == 0) & (s2 > 0))
        keep = (i_sel < j_sel) | ((i_sel == j_sel) & s_pos)
        i_sel, j_sel, s_sel, d_sel = i_sel[keep], j_sel[keep], s_sel[keep], d_sel[keep]

    return torch.stack([i_sel, j_sel], dim=1), s_sel, d_sel

DIR_PATH = Path(__file__).parent
EXAMPLES = DIR_PATH / ".." / "examples"
COULOMB_TEST_FRAMES = EXAMPLES / "coulomb_test_frames.xyz"
DIPOLES_TEST_FRAMES = EXAMPLES / "dipoles_test_frames.xyz"
DEVICES = ["cpu", torch.device("cpu")] + torch.cuda.is_available() * ["cuda"]
DTYPES = [torch.float32, torch.float64]


def define_crystal(crystal_name="CsCl", dtype=None, device=None):
    # Define all relevant parameters (atom positions, charges, cell) of the reference
    # crystal structures for which the Madelung constants obtained from the Ewald sums
    # are compared with reference values.
    # see https://www.sciencedirect.com/science/article/pii/B9780128143698000078#s0015
    # More detailed values can be found in https://pubs.acs.org/doi/10.1021/ic2023852

    # Caesium-Chloride (CsCl) structure:
    # - Cubic unit cell
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    if crystal_name == "CsCl":
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        charges = torch.tensor([-1.0, 1.0])
        cell = torch.eye(3)
        madelung_ref = 2.0353610945260
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        charges = torch.tensor([1.0, -1.0])
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])  # fcc
        madelung_ref = 1.7475645946
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a cubic unit cell
    # - cubic unit cell
    # - 4 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_cubic":
        positions = torch.tensor(
            [
                [0.0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
        )
        charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1])
        cell = 2 * torch.eye(3)
        madelung_ref = 1.7475645946
        num_formula_units = 4

    # ZnS (zincblende) structure
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    # Remarks: we use a primitive unit cell which makes the lattice parameter of the
    # cubic cell equal to 2.
    elif crystal_name == "zincblende":
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        charges = torch.tensor([1.0, -1])
        cell = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        madelung_ref = 2 * 1.6380550533 / SQRT3
        num_formula_units = 1

    # Wurtzite structure
    # - non-cubic unit cell (triclinic)
    # - 2 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "wurtzite":
        u = 3 / 8
        c = math.sqrt(1 / u)
        positions = torch.tensor(
            [
                [0.5, 0.5 / SQRT3, 0.0],
                [0.5, 0.5 / SQRT3, u * c],
                [0.5, -0.5 / SQRT3, 0.5 * c],
                [0.5, -0.5 / SQRT3, (0.5 + u) * c],
            ],
        )
        charges = torch.tensor([1.0, -1, 1, -1])
        cell = torch.tensor(
            [[0.5, -0.5 * SQRT3, 0], [0.5, 0.5 * SQRT3, 0], [0, 0, c]],
        )
        madelung_ref = 1.64132 / (u * c)
        num_formula_units = 2

    # Fluorite structure (e.g. CaF2 with Ca2+ and F-)
    # - non-cubic (fcc) unit cell
    # - 1 neutral molecule per unit cell
    # - Cation-Anion ratio of 1:2
    elif crystal_name == "fluorite":
        a = 5.463
        a = 1.0
        positions = a * torch.tensor([[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]])
        charges = torch.tensor([-1, -1, 2])
        cell = torch.tensor([[a, a, 0], [a, 0, a], [0, a, a]]) / 2.0
        madelung_ref = 11.6365752270768
        num_formula_units = 1

    # Copper(I)-Oxide structure (e.g. Cu2O with Cu+ and O2-)
    # - cubic unit cell
    # - 2 neutral molecules per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "cu2o":
        a = 1.0
        positions = a * torch.tensor(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
            ],
        )
        charges = torch.tensor([-2, -2, 1, 1, 1, 1])
        cell = a * torch.eye(3)
        madelung_ref = 10.2594570330750
        num_formula_units = 2

    # Wigner crystal in simple cubic structure.
    # Wigner crystals are equivalent to the Jellium or uniform electron gas models.
    # For the purpose of this test, we define them to be structures in which the ion
    # cores form a perfect lattice, while the electrons are uniformly distributed over
    # the cell. In some sources, the role of the positive and negative charges are
    # flipped. These structures are used to test the code for cases in which the total
    # charge of the particles is not zero.
    # Wigner crystal energies are taken from "Zero-Point Energy of an Electron Lattice"
    # by Rosemary A., Coldwell‐Horsfall and Alexei A. Maradudin (1960), eq. (A21).
    elif crystal_name == "wigner_sc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.7601188
        wigner_seiz_radius = (3 / (4 * torch.pi)) ** (1 / 3)
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 2.83730
        num_formula_units = 1

    # Wigner crystal in bcc structure (note: this is the most stable structure).
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_bcc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * torch.pi * 2)) ** (1 / 3)  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive bcc cell
    elif crystal_name == "wigner_bcc_cubiccell":
        positions = torch.tensor([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]])
        charges = torch.tensor([1.0, 1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * torch.pi * 2)) ** (1 / 3)  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 2

    # Wigner crystal in fcc structure
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_fcc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]]) / 2

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * torch.pi * 4)) ** (1 / 3)  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive fcc cell
    elif crystal_name == "wigner_fcc_cubiccell":
        positions = 0.5 * torch.tensor([[0.0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
        charges = torch.tensor([1.0, 1, 1, 1])
        cell = torch.eye(3)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * torch.pi * 4)) ** (1 / 3)  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 4

    else:
        raise ValueError(f"crystal_name = {crystal_name} is not supported!")

    charges = charges.reshape((-1, 1))

    return (
        positions.to(device=device, dtype=dtype),
        charges.to(device=device, dtype=dtype),
        cell.to(device=device, dtype=dtype),
        torch.tensor(madelung_ref, device=device, dtype=dtype),
        num_formula_units,
    )


def neighbor_list(
    positions: torch.tensor,
    periodic: bool = True,
    box: Optional[torch.tensor] = None,
    cutoff: Optional[float] = None,
    full_neighbor_list: bool = False,
    neighbor_shifts: bool = False,
) -> tuple[torch.tensor, torch.tensor]:
    if box is None:
        box = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)

    if cutoff is None:
        cell_dimensions = torch.linalg.norm(box, dim=1)
        cutoff_torch = torch.min(cell_dimensions) / 2 - 1e-6
        cutoff = cutoff_torch.item()

    # Compute in float64 on CPU for geometric precision (the cell math and the
    # cutoff comparison are precision-sensitive), then cast back to the caller's
    # dtype/device.
    pairs, S, d = periodic_neighbor_list(
        positions.to(dtype=torch.float64, device="cpu"),
        box.to(dtype=torch.float64, device="cpu"),
        float(cutoff),
        full_list=full_neighbor_list,
        periodic=periodic,
    )

    neighbor_indices = pairs.to(dtype=torch.long, device=positions.device)
    d = d.to(dtype=positions.dtype, device=positions.device)
    S = S.to(dtype=positions.dtype, device=positions.device)

    if not neighbor_shifts:
        return neighbor_indices, d
    return neighbor_indices, S


def compute_distances(
    positions: torch.tensor,
    neighbor_indices: torch.tensor,
    cell: Optional[torch.tensor] = None,
    neighbor_shifts: Optional[torch.tensor] = None,
    norm: bool = True,
) -> torch.tensor:
    """Compute pairwise distance vectors or scalar distances."""
    atom_is = neighbor_indices[:, 0]
    atom_js = neighbor_indices[:, 1]

    pos_is = positions[atom_is]
    pos_js = positions[atom_js]

    distance_vectors = pos_js - pos_is

    if cell is not None and neighbor_shifts is not None:
        shifts = neighbor_shifts.type(cell.dtype)
        distance_vectors += shifts @ cell
    elif cell is not None and neighbor_shifts is None:
        raise ValueError("Provided `cell` but no `neighbor_shifts`.")
    elif cell is None and neighbor_shifts is not None:
        raise ValueError("Provided `neighbor_shifts` but no `cell`.")

    if norm:
        return torch.linalg.norm(distance_vectors, dim=1)
    return distance_vectors
