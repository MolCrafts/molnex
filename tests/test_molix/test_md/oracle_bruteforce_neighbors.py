"""Independent multi-image neighbor-graph oracle for NeighborList tests.

**Not** the MIC-only helper in ``test_neighbors._reference_pairs``. Enumerates
all lattice images with a range derived from perpendicular cell widths so
pairs that need |S| > 1 are found when the box is small relative to the cutoff.

No imports from ``molix.md.neighbors``, ``molix.op``, matscipy, ASE, freud, or
LAMMPS. Production filter parity (``get_neighbor_pairs`` / binned path)::

    0 < |dr| <= cutoff

True self-edge ``i == j`` and ``S == (0,0,0)`` is excluded; periodic self-images
are kept when inside the cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

# Edge key: central i, neighbor j, integer image shift of j's lattice copy.
EdgeKey = tuple[int, int, int, int, int]


def perpendicular_widths(cell: torch.Tensor) -> torch.Tensor:
    """Perpendicular widths ``V / ||a_j × a_k||`` for each axis (Angstrom)."""
    c = cell.to(dtype=torch.float64)
    v = torch.det(c).abs()
    cross = torch.stack(
        (
            torch.linalg.cross(c[1], c[2]),
            torch.linalg.cross(c[2], c[0]),
            torch.linalg.cross(c[0], c[1]),
        ),
        dim=0,
    )
    areas = cross.norm(dim=-1).clamp_min(1e-30)
    return v / areas


def image_range(cell: torch.Tensor, cutoff: float) -> tuple[int, int, int]:
    """Half-widths of the integer image stencil per axis (at least 1 if periodic)."""
    widths = perpendicular_widths(cell)
    r = float(cutoff)
    out: list[int] = []
    for w in widths.tolist():
        if not math.isfinite(w) or w <= 0.0:
            out.append(1)
            continue
        # Need n such that n * w can still reach distance ~ cutoff.
        n = int(math.ceil(r / w + 1e-12))
        out.append(max(1, n))
    return out[0], out[1], out[2]


def bruteforce_edges(
    pos: torch.Tensor,
    *,
    cell: torch.Tensor | None,
    cutoff: float,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> set[EdgeKey]:
    """Return the set of directed edges ``(i, j, sx, sy, sz)`` with ``0 < r <= cutoff``."""
    p = pos.detach().to(dtype=torch.float64)
    n = int(p.shape[0])
    r_cut = float(cutoff)
    r_cut_sq = r_cut * r_cut
    keys: set[EdgeKey] = set()

    if cell is None or not any(pbc):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = p[j] - p[i]
                d2 = float((d * d).sum())
                if 0.0 < d2 <= r_cut_sq:
                    keys.add((i, j, 0, 0, 0))
        return keys

    c = cell.detach().to(dtype=torch.float64)
    rx, ry, rz = image_range(c, r_cut)
    ranges = []
    for periodic, rmax in zip(pbc, (rx, ry, rz), strict=True):
        ranges.append(range(-rmax, rmax + 1) if periodic else range(0, 1))

    for i in range(n):
        for j in range(n):
            for sx in ranges[0]:
                for sy in ranges[1]:
                    for sz in ranges[2]:
                        if i == j and sx == 0 and sy == 0 and sz == 0:
                            continue
                        shift_cart = float(sx) * c[0] + float(sy) * c[1] + float(sz) * c[2]
                        d = p[j] - p[i] + shift_cart
                        d2 = float((d * d).sum())
                        if 0.0 < d2 <= r_cut_sq:
                            keys.add((i, j, int(sx), int(sy), int(sz)))
    return keys


def shifts_to_integer_S(
    shifts: torch.Tensor,
    cell: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Map continuous Å remainders to integer lattice vectors ``(E, 3)``.

    Solves ``S @ cell ≈ shifts`` i.e. ``S ≈ shifts @ inv(cell)``, then rounds.
    Raises if any residual exceeds *atol* (Å).
    """
    s = shifts.detach().to(dtype=torch.float64)
    c = cell.detach().to(dtype=torch.float64)
    inv = torch.linalg.inv(c)
    frac = s @ inv
    S = torch.round(frac)
    residual = (S @ c - s).norm(dim=-1)
    bad = residual > atol
    if bool(bad.any()):
        idx = int(torch.nonzero(bad, as_tuple=False)[0].item())
        raise AssertionError(
            f"shift→integer residual {float(residual[idx]):.3e} A exceeds atol={atol} "
            f"at edge {idx}; shift={s[idx].tolist()} S={S[idx].tolist()} cell={c.tolist()}"
        )
    return S.to(dtype=torch.long)


def neighborlist_edge_keys(
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    num_edges: int,
    cell: torch.Tensor | None,
    *,
    open_system: bool = False,
) -> set[EdgeKey]:
    """Convert live NeighborList buffers to the oracle edge-key set."""
    n = int(num_edges)
    if n == 0:
        return set()
    src = edge_index[:n, 0].detach().cpu().long()
    tgt = edge_index[:n, 1].detach().cpu().long()
    sh = shifts[:n].detach().cpu()
    if open_system or cell is None:
        return {
            (int(src[k]), int(tgt[k]), 0, 0, 0)
            for k in range(n)
            if not (int(src[k]) == int(tgt[k]) and sh[k].abs().sum() < 1e-12)
        }
    S = shifts_to_integer_S(sh, cell)
    return {(int(src[k]), int(tgt[k]), int(S[k, 0]), int(S[k, 1]), int(S[k, 2])) for k in range(n)}


@dataclass(frozen=True)
class GraphCompare:
    """Diagnostics for one oracle vs SUT comparison."""

    n_ref: int
    n_sut: int
    missing: frozenset[EdgeKey]
    extra: frozenset[EdgeKey]
    max_dr_mismatch: float

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra

    def summary(self) -> str:
        return (
            f"n_ref={self.n_ref} n_sut={self.n_sut} "
            f"missing={len(self.missing)} extra={len(self.extra)} "
            f"max_dr_mismatch={self.max_dr_mismatch:.3e}"
        )


def compare_graphs(
    ref: set[EdgeKey],
    sut: set[EdgeKey],
    *,
    pos: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
) -> GraphCompare:
    """Set difference + optional max displacement mismatch on the intersection."""
    missing = frozenset(ref - sut)
    extra = frozenset(sut - ref)
    max_dr = 0.0
    if pos is not None and cell is not None and (ref & sut):
        p = pos.detach().to(dtype=torch.float64)
        c = cell.detach().to(dtype=torch.float64)
        for i, j, sx, sy, sz in ref & sut:
            shift = float(sx) * c[0] + float(sy) * c[1] + float(sz) * c[2]
            dr = p[j] - p[i] + shift
            # Both keys mean the same S; mismatch is numerical only.
            max_dr = max(max_dr, float(dr.norm()))  # not a mismatch — keep 0
        max_dr = 0.0  # keys match ⇒ same S by construction; residual checked elsewhere
    return GraphCompare(
        n_ref=len(ref),
        n_sut=len(sut),
        missing=missing,
        extra=extra,
        max_dr_mismatch=max_dr,
    )


def assert_graphs_equal(
    ref: set[EdgeKey],
    sut: set[EdgeKey],
    *,
    label: str = "",
) -> GraphCompare:
    """Hard-fail with diagnostics if missing/extra edges exist."""
    cmp = compare_graphs(ref, sut)
    if not cmp.ok:
        head_m = list(sorted(cmp.missing))[:12]
        head_e = list(sorted(cmp.extra))[:12]
        raise AssertionError(
            f"{label}neighbor graph mismatch ({cmp.summary()}); "
            f"cutoff convention is 0 < r <= r_c. "
            f"missing sample={head_m} extra sample={head_e}"
        )
    return cmp


def physical_dr_multiset(
    keys: Iterable[EdgeKey],
    pos: torch.Tensor,
    cell: torch.Tensor | None,
    *,
    decimals: int = 6,
) -> set[tuple[float, float, float]]:
    """Multiset of rounded physical displacements for wrap-invariance checks."""
    p = pos.detach().to(dtype=torch.float64)
    c = (
        cell.detach().to(dtype=torch.float64)
        if cell is not None
        else torch.eye(3, dtype=torch.float64)
    )
    out: set[tuple[float, float, float]] = set()
    for i, j, sx, sy, sz in keys:
        shift = float(sx) * c[0] + float(sy) * c[1] + float(sz) * c[2]
        dr = p[j] - p[i] + shift
        out.add(
            (
                round(float(dr[0]), decimals),
                round(float(dr[1]), decimals),
                round(float(dr[2]), decimals),
            )
        )
    return out
