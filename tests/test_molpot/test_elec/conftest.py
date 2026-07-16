"""Test utilities wrap common functions in the tests"""

import torch


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
    dvec = positions.view(1, 1, n, 3) + offsets.view(n_shift, 1, 1, 3) - positions.view(1, n, 1, 3)
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


DEVICES = ["cpu", torch.device("cpu")] + torch.cuda.is_available() * ["cuda"]
DTYPES = [torch.float32, torch.float64]
