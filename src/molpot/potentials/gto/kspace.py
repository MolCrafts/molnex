"""Official Polar GTO reciprocal-space electrostatics.

A *Gaussian-type orbital* (GTO) charge model smears each atom's charge over a
Gaussian of width ``σ`` instead of leaving it at a point, which keeps the pair
energy finite as two atoms approach. Under periodic boundary conditions that
energy is summed over the **reciprocal lattice** — the wavevectors ``k`` (Å⁻¹)
whose plane waves ``exp(i k · r)`` carry the periodicity of the simulation cell
— because that sum converges where the real-space one does not.

Not PME (particle-mesh Ewald, the FFT-based charge splitting most molecular
dynamics codes use). Polar SI V.6 is the analytic GTO Fourier transform plus a
**direct reciprocal-lattice sum** (half-box, conjugate symmetry). The
contraction is LAMMPS ``ewald`` / ``eik_dot_r``: one-dimensional
``(cos, sin)`` tables of length ``2 N_c,d + 1`` (the integer box frozen on
:class:`KSpaceGrid`), then a separable sum onto that Miller grid. Tables stay
real so inductor can codegen them — ``torch.complex`` is an ATen fallback.
Workspace is ``O(N N_c²)``, never the plane-wave matrix ``(N, M)`` with
``M ~ N_c³``, and there is no arbitrary k-tile. ``E`` depends on ``cell``
through the reciprocal basis, so ``∂E/∂pos`` and the virial ``∂E/∂ε`` are
ordinary autograd.

The released MACE-POLAR-1 checkpoint — "the dump" throughout this package —
uses the Gaussian convention ``erf(r/(2σ))/r`` (not paper Eq. 37). Two
Gaussians of width ``σ`` have form factor ``exp(-σ² k² / 2)`` each, so the pair
kernel in reciprocal space is ``exp(-σ² k²) / k²``. Energy in eV::

    E = eV_A · (2π / Ω) Σ_{k ≠ 0} exp(-σ² k²) / k² |S(k)|²
    S(k) = Σ_i (q_i + i k · μ_i) exp(i k · r_i)

``Ω`` is the cell volume in Å³, ``q_i`` the charge of atom ``i`` in e, ``μ_i``
its dipole in e·Å, ``r_i`` its position in Å, and ``eV_A = 1/(4πε₀)`` the
Coulomb constant in eV·Å/e². ``S(k)`` is the *structure factor*, the Fourier
transform of that multipole density — one complex number per k-vector.

Cutoff (training ``--kspace_cutoff_factor=κ``, SI typical ``κ = 1.5``), with
``σ_min`` the smallest GTO width in play and ``l_max`` the highest multipole
order (0 = charges only, 1 = charges and dipoles)::

    k_cutoff = κ · 2.25 / σ_min · (l_max + 1)^{0.3}

That line is a **reference-implementation convention, not a paper equation**:
``graph_longrange`` spells it ``gto_basis_kspace_cutoff = 0.75 · 3 / σ_min ·
(l + 1)^{0.3}`` and ``κ`` scales its ``0.75``. The paper fixes no reciprocal
cutoff, so do not attribute an equation number to it.

Units throughout: charge e, position Å, energy eV, reciprocal vectors Å⁻¹.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math

import torch

_TWOPI = 2.0 * math.pi

#: Raised by :meth:`KSpaceGrid.vectors` before the first :meth:`KSpaceGrid.bind`.
#: Named on purpose: an unbound grid must fail with the call the caller is
#: missing, not with a matmul shape error from a ``None`` integer box.
_UNBOUND_MESSAGE = "KSpaceGrid is not bound — call bind_cell(cell) before compiling or exporting"


def kspace_cutoff(sigma: float, l_max: int, kappa: float = 1.0) -> float:
    """Polar reciprocal cutoff in Å⁻¹ (reference-implementation convention).

    The cutoff is the largest wavevector magnitude ``|k|`` the reciprocal-space
    sum keeps; beyond it the Gaussian form factor ``exp(-σ² k²)`` has already
    crushed the terms. The formula is ``κ · 2.25 / σ · (l_max + 1)^{0.3}`` —
    see the module docstring: this is ``graph_longrange``'s
    ``gto_basis_kspace_cutoff``, not a paper equation.

    Args:
        sigma: Smallest GTO (Gaussian-type orbital) width in Å.
        l_max: Multipole order entering the cutoff exponent, dimensionless
            (0 = charges only, 1 = charges and dipoles).
        kappa: ``kspace_cutoff_factor`` ``κ``, dimensionless. Training dump
            default is ``1.0``.

    Returns:
        ``k_cutoff`` in Å⁻¹.

    Raises:
        ValueError: If ``sigma`` or ``kappa`` is not strictly positive, or if
            ``l_max`` is negative.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0 Å, got {sigma}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    return float(kappa) * 2.25 / float(sigma) * (float(l_max) + 1.0) ** 0.3


def cartesian_dipole(feats: torch.Tensor) -> torch.Tensor:
    """Pull the Cartesian dipole ``μ = (px, py, pz)`` out of an ``ir_mul`` row.

    ``ir_mul`` (irreducible-representation-major) is the cuEquivariance feature
    layout used across this package: the four columns of an ``l_max = 1``
    multipole are ``(q, py, pz, px)`` — the monopole first, then the three
    ``l = 1`` components in the ``(y, z, x)`` order real spherical harmonics
    use. This is the reordering to the physics convention ``(x, y, z)``;
    nothing is rescaled.

    Args:
        feats: Multipoles ``(N, >= 4)``: column 0 the monopole in e, columns
            1-3 the dipole components in e·Å.

    Returns:
        Dipoles ``(N, 3)`` in e·Å, ordered ``(x, y, z)``.
    """
    return torch.stack((feats[:, 3], feats[:, 1], feats[:, 2]), dim=-1)


def _as_cell(cell: torch.Tensor) -> torch.Tensor:
    """Squeeze a ``(1, 3, 3)`` batch cell to ``(3, 3)``."""
    if cell.dim() == 3:
        if cell.shape[0] != 1:
            raise ValueError(
                f"GTO k-space is single-system: cell must be (3, 3) or (1, 3, 3), "
                f"got {tuple(cell.shape)}"
            )
        cell = cell[0]
    if tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must be (3, 3), got {tuple(cell.shape)}")
    return cell


def reciprocal_basis(cell: torch.Tensor) -> torch.Tensor:
    """Rows ``b_i`` of ``2π cell^{-T}`` in Å⁻¹.

    Same ``g`` :meth:`KSpaceGrid.vectors` uses: ``k(n) = n · g``.
    """
    return _TWOPI * torch.linalg.inv(_as_cell(cell)).T


class KSpaceGrid:
    """Fixed-capacity integer half-box of reciprocal indices, sized once.

    A runtime geometry cache, not a model parameter: it is a plain class, not
    an ``nn.Module``, so it never enters a ``state_dict`` (same reasoning as
    :class:`molix.md.NeighborList`). Holders keep it as a plain attribute.

    Each reciprocal-lattice vector is an integer triple ``n = (n₁, n₂, n₃)``
    read against the reciprocal basis, ``k(n) = Σ_i n_i b_i``, where ``b_i`` is
    row ``i`` of ``2π cell⁻ᵀ`` in Å⁻¹ and its partner ``a_i``, row ``i`` of
    ``cell`` in Å, is a lattice vector. The grid stores those **integers**:
    they do not move when the cell deforms — only the weights computed in
    :meth:`vectors` do.

    **Fixed-box equivalence.** With ``n ∈ [-N_c, N_c]³`` and ``k(n)`` as above,
    the weight

        ``w(n) = 2 · 1[first non-zero n_i > 0] · 1[0 < |k|² < k_cutoff²]``

    reproduces the masked half-sphere term by term: real ``charge``, ``μ``, ``r``
    give ``S(-k) = conj(S(k))``, so one member of each ``±n`` pair carries
    weight ``2`` and the rest carry exactly ``0.0`` (IEEE 754 arithmetic gives
    ``x + 0.0 == x``, so padding rows are free). ``k = 0`` is excluded by
    construction *and* the denominator is guarded **before** the division —
    ``1/0 = inf`` followed by ``0 · inf = NaN`` would poison value and gradient
    alike. The cutoff comparison is strict ``<``, as in the pre-change mask.

    **Capacity bound and its overflow predicate.** ``a_i · b_j = 2π δ_ij``
    gives ``2π|n_i| = |a_i · k| ≤ ‖a_i‖ |k|``, so ``|k| < k_cutoff`` implies
    ``|n_i| < x_i := ‖a_i‖ k_cutoff / 2π`` — exact for a triclinic cell (the
    general case, where the three lattice vectors are neither orthogonal nor
    equal in length). The capacity is ``N_c,i = ceil(capacity_factor · x_i)``,
    computed from the **real-valued** ``x_i``; ``ceil(f · n_max)`` under-sizes
    (``x = 2.9``, ``f = 1.5``: ``floor(4.35) = 4 > ceil(3) = 3``). The overflow
    predicate is ``floor(x_i) > N_c,i``, used **identically** by the eager half
    (:meth:`check_capacity` raises) and the in-graph half (:meth:`vectors`
    returns NaN weights) — so both fail on the same cell, and an over-capacity
    grid **never truncates the k-sum silently**.

    *The strict inequality leaves a deliberate conservative edge.* Since the
    bound is ``|n_i| < x_i``, a sufficient box is ``N_c,i ≥ ceil(x_i) - 1``.
    When ``x_i`` is exactly an integer (a measure-zero set) the predicate fires
    one notch early, at ``x_i = N_c,i + 1``, although no k-point would in fact
    be dropped. That is intentional fail-closed behaviour: do **not** "fix" it
    to ``>=`` (which fires earlier still), and do not rewrite it as
    ``ceil(x) - 1 > N_c`` (a second home for one bound). A cell that contracts
    — as it may under a barostat, the constant-pressure (NPT: fixed particle
    number, pressure and temperature) control that rescales the box — never
    overflows; one that expands by a factor ``λ ≤ capacity_factor`` never
    overflows either.

    **Lifecycle.** ``construct → to(device) → bind_cell(cell)`` (eager, sizes
    the box once) ``→ compile / export → forward`` every step. Only the first
    :meth:`bind` sizes anything; every later call is a pure
    :meth:`check_capacity`. **Resizing means building a new** ``KSpaceGrid`` —
    there is deliberately no ``rebind`` / ``set_cell``, exactly as
    :class:`molix.md.NeighborList` computes its capacity once and afterwards
    raises on overflow rather than quietly growing a buffer.

    The half box costs k-points: ``(2 N_c + 1)³`` against a half-sphere's
    ``≈ (2/3)π n_max³`` is ``(2f)³ / (4π/3)`` asymptotically (3.30 at
    ``f = 1.2``), and worse on small cells. A half-sphere list is *not* a safe
    static superset for a triclinic cell — the k-ball is a strain-dependent
    ellipsoid in ``n`` space — so the box is the only shape with a guarantee.
    A caller running at fixed volume (the NVT ensemble: fixed particle number,
    volume and temperature) never expands the cell and can pass
    ``capacity_factor=1.0``.

    Args:
        k_cutoff: Reciprocal cutoff in Å⁻¹, frozen at construction (see
            :func:`kspace_cutoff`).
        capacity_factor: Dimensionless head-room on the per-axis index,
            ``N_c = ceil(f x)``. Defaults to ``1.2``, matching the periodic
            Polar regression's ``NeighborList``.
        max_axis_index: Per-axis (not total) index ceiling, default ``32``. A
            blown-up MD cell makes the mesh ``O(L³)``; fail closed instead of
            allocating tens of GiB.

    Raises:
        ValueError: If ``k_cutoff`` is not strictly positive, if
            ``capacity_factor`` is below ``1`` (a box tighter than the bound
            would drop k-points that lie inside the cutoff), or if
            ``max_axis_index`` is below ``1``.

    Reference:
        Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, SI V.6.
    """

    def __init__(
        self,
        k_cutoff: float,
        *,
        capacity_factor: float = 1.2,
        max_axis_index: int = 32,
    ) -> None:
        if k_cutoff <= 0.0:
            raise ValueError(f"k_cutoff must be > 0 Å⁻¹, got {k_cutoff}")
        if capacity_factor < 1.0:
            raise ValueError(f"capacity_factor must be >= 1, got {capacity_factor}")
        if max_axis_index < 1:
            raise ValueError(f"max_axis_index must be >= 1, got {max_axis_index}")
        self.k_cutoff = float(k_cutoff)
        self.capacity_factor = float(capacity_factor)
        self.max_axis_index = int(max_axis_index)
        self._n: torch.Tensor | None = None
        self._n_cap: torch.Tensor | None = None
        self._capacity: tuple[int, int, int] | None = None

    @property
    def n(self) -> torch.Tensor:
        """Integer half-box ``(M, 3)`` int64, one row per kept ``n`` triple.

        The same tensor object across repeated :meth:`bind` calls — the box is
        sized once and never resized. :meth:`to` is the one exception: a device
        move replaces it.
        """
        if self._n is None:
            raise RuntimeError(_UNBOUND_MESSAGE)
        return self._n

    @property
    def capacity(self) -> tuple[int, int, int]:
        """Per-axis index capacity ``N_c`` fixed by the first :meth:`bind`.

        Three dimensionless integer indices ``(N_c,1, N_c,2, N_c,3)``, one per
        lattice vector.
        """
        if self._capacity is None:
            raise RuntimeError(_UNBOUND_MESSAGE)
        return self._capacity

    def bind(self, cell: torch.Tensor) -> None:
        """Size the integer box on the first call, check capacity afterwards.

        **Eager only** — the one entry point allowed to sync with the host.
        The first call reads three lattice norms back to Python to fix
        ``N_c,i = ceil(capacity_factor · ‖a_i‖ k_cutoff / 2π)`` and builds the
        half box ``n ∈ [-N_c, N_c]³`` keeping rows whose first non-zero
        component is positive (the origin is dropped, so
        ``M = (Π_i (2 N_c,i + 1) − 1) / 2``). The integer triples do not depend
        on ``cell``; only the weights in :meth:`vectors` do. Every later call
        is exactly :meth:`check_capacity`: the box is never resized, neither
        down (a contracted cell always fits) nor up (build a new grid).

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å, one per
                row.

        Raises:
            ValueError: If ``cell`` is neither ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: On the first call when ``max(N_c) > max_axis_index``;
                on a later call when the cell has outgrown the capacity.
        """
        cell = _as_cell(cell)
        if self._n is not None:
            self.check_capacity(cell)
            return
        # ``detach``: a virial step binds with a strain-tracking cell, and
        # reading a grad-tracking scalar back to Python is a warning, not a
        # gradient — the capacity is metadata, never part of the graph.
        norms = tuple(float(torch.linalg.norm(cell[i].detach())) for i in range(3))
        scale = self.capacity_factor * self.k_cutoff / _TWOPI
        capacity = tuple(math.ceil(scale * norm) for norm in norms)
        if max(capacity) > self.max_axis_index:
            raise RuntimeError(
                f"GTO k-grid exploded: N_c={capacity} for |cell|={norms} Å, "
                f"k_cutoff={self.k_cutoff} Å⁻¹, capacity_factor={self.capacity_factor} "
                f"(max_axis_index={self.max_axis_index}). The lattice is unphysically large "
                f"— fix the MD (forces / barostat) rather than summing this mesh."
            )
        device = cell.device
        axes = [torch.arange(-c, c + 1, dtype=torch.int64, device=device) for c in capacity]
        box = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        first = torch.argmax((box != 0).to(torch.int64), dim=1)
        sign = torch.gather(box, 1, first.unsqueeze(1)).squeeze(-1)
        self._n = box[sign > 0].contiguous()
        self._n_cap = torch.tensor(capacity, dtype=torch.int64, device=device)
        self._capacity = (capacity[0], capacity[1], capacity[2])

    def check_capacity(self, cell: torch.Tensor) -> None:
        """Raise if ``cell`` has outgrown the box. **Eager only** (host sync).

        The eager half of the ``floor(x_i) > N_c,i`` guard described in the
        class docstring. Its in-graph twin lives in :meth:`vectors` and fires
        on exactly the same cell.

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å.

        Raises:
            ValueError: If ``cell`` is neither ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: When the grid was never bound, or when any axis needs
                an index above the capacity. The overflow message names both
                ``KSpaceGrid`` and ``capacity_factor`` — the neighbour-list
                overflow uses the same word, and the two must be told apart at
                a glance.
        """
        cell = _as_cell(cell)
        capacity = self.capacity
        for i in range(3):
            x = float(torch.linalg.norm(cell[i].detach())) * self.k_cutoff / _TWOPI
            needed = math.floor(x)
            if needed > capacity[i]:
                raise RuntimeError(
                    f"KSpaceGrid overflow: cell needs per-axis index {needed} on axis {i} > "
                    f"capacity {capacity[i]} (x={x:.6g}, k_cutoff={self.k_cutoff} Å⁻¹). Raise "
                    f"capacity_factor (currently {self.capacity_factor}) or build a new "
                    f"KSpaceGrid."
                )

    def vectors(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reciprocal vectors, guarded denominators and 0/2 weights.

        Fully traceable and fixed-shape: every returned tensor has ``M`` rows
        whatever the cell, booleans are only ever multiplicative numbers and
        never indices, and no scalar is read back to the host.

        The in-graph overflow sentinel is the same ``floor(x) > N_c`` predicate
        :meth:`check_capacity` raises on, evaluated as a tensor compare: when
        it trips, **all** weights become NaN. An over-capacity grid poisons the
        energy, forces and virial together — loudly and fail-closed — and never
        silently truncates the k-sum.

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å.

        Returns:
            ``kvec (M, 3)`` in Å⁻¹, ``k_sq_guarded (M,)`` in Å⁻² and
            ``weights (M,)`` in ``{0.0, 2.0}`` (or all-NaN on overflow).
            ``k_sq_guarded`` is **not** ``|k|²``: it reads ``1`` where
            ``|k|² == 0``. It is safe anywhere ``weights`` masks the row —
            both in-tree callers feed it to the Gaussian exponent
            ``exp(-σ² k²)`` as well as to the divisor, and every row it
            substitutes carries weight ``0.0``, so the substituted value never
            reaches the sum. Reading it as ``|k|²`` *outside* a
            ``weights``-masked expression is wrong; ``weights`` remains the
            single source of masking.

        Raises:
            ValueError: If ``cell`` is neither ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: When the grid was never bound, or when the integer
                box and the cell live on different devices.
        """
        cell = _as_cell(cell)
        box = self._n
        n_cap = self._n_cap
        if box is None or n_cap is None:
            raise RuntimeError(_UNBOUND_MESSAGE)
        if box.device != cell.device:
            raise RuntimeError(
                f"KSpaceGrid device mismatch: the integer box is on {box.device} but the cell "
                f"is on {cell.device}. Call grid.to({cell.device}) before binding, or build a "
                f"new KSpaceGrid on that device."
            )
        dtype = cell.dtype
        g = reciprocal_basis(cell)
        kvec = box.to(dtype) @ g
        k_sq = (kvec * kvec).sum(dim=-1)
        live = (k_sq > 0.0) & (k_sq < self.k_cutoff * self.k_cutoff)
        weights = 2.0 * live.to(dtype)
        k_sq_guarded = torch.where(k_sq > 0.0, k_sq, torch.ones_like(k_sq))
        x = torch.linalg.norm(cell, dim=1) * (self.k_cutoff / _TWOPI)
        overflow = (torch.floor(x) > n_cap).any()
        weights = torch.where(overflow, weights.new_full((), float("nan")), weights)
        return kvec, k_sq_guarded, weights

    def to(self, device: torch.device | str) -> KSpaceGrid:
        """Move the integer box in place and return ``self``.

        Warning:
            **Tensors handed out earlier do not travel.** The move rebinds
            :attr:`n` (and the capacity sentinel) to *new* tensors on the new
            device, so any ``kvec`` / ``weights`` triple taken from
            :meth:`vectors` beforehand still points at the old device and would
            keep feeding the pre-move grid into a live graph. Re-reading from
            inside ``to`` is deliberately not done — the grid does not own its
            consumers — so the **owner** re-reads: call :meth:`vectors` again
            after the move (this is what ``GTOHartreeEnergy._apply`` and
            ``GTOFieldFeatures._apply`` arrange by construction, since they
            move the grid alongside their own buffers).

        Args:
            device: Target device. An unbound grid records nothing and is a
                no-op — :meth:`bind` builds the box on the cell's device.

        Returns:
            ``self``, so a move can be chained.
        """
        target = torch.device(device)
        if self._n is not None:
            self._n = self._n.to(target)
        if self._n_cap is not None:
            self._n_cap = self._n_cap.to(target)
        return self


def _miller_tables(
    pos: torch.Tensor,
    recip: torch.Tensor,
    capacity: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """LAMMPS ``eik_dot_r`` 1-D tables as ``(cos, sin)`` pairs.

    ``n_d`` runs over ``[-N_c,d, N_c,d]`` — the integer box already frozen on
    :class:`KSpaceGrid`, not an arbitrary tile. Each table is ``(N, 2 N_c + 1)``
    real; three axes, never ``(N, M)``. Cos/sin stay real so inductor can
    codegen (``torch.complex`` is an ATen fallback).
    """
    theta = pos @ recip.transpose(0, 1)
    tables: list[torch.Tensor] = []
    for dim, n_cap in enumerate(capacity):
        n1d = torch.arange(-n_cap, n_cap + 1, device=pos.device, dtype=pos.dtype)
        phase = theta[:, dim].unsqueeze(-1) * n1d
        tables.append(torch.cos(phase))
        tables.append(torch.sin(phase))
    return tables[0], tables[1], tables[2], tables[3], tables[4], tables[5]


def _cmul(
    ar: torch.Tensor,
    ai: torch.Tensor,
    br: torch.Tensor,
    bi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ar + i ai)(br + i bi)`` as a real pair."""
    return ar * br - ai * bi, ar * bi + ai * br


def _separable_sum(
    cx: torch.Tensor,
    sx: torch.Tensor,
    cy: torch.Tensor,
    sy: torch.Tensor,
    cz: torch.Tensor,
    sz: torch.Tensor,
    amp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``S[nx, ny, nz] = Σ_i amp_i exp(i n · θ_i)`` as ``(real, imag)``.

    Intermediate ``(N, N_c,x, N_c,y)`` from the two in-plane tables, then one
    GEMM over atoms onto the third axis. ``N_c`` is the Miller box, not ``M``.
    """
    plane_r, plane_i = _cmul(cx.unsqueeze(2), sx.unsqueeze(2), cy.unsqueeze(1), sy.unsqueeze(1))
    az_r = amp.unsqueeze(-1) * cz
    az_i = amp.unsqueeze(-1) * sz
    sr = torch.einsum("nab,nc->abc", plane_r, az_r) - torch.einsum("nab,nc->abc", plane_i, az_i)
    si = torch.einsum("nab,nc->abc", plane_r, az_i) + torch.einsum("nab,nc->abc", plane_i, az_r)
    return sr, si


def _structure_factor_grid(
    cx: torch.Tensor,
    sx: torch.Tensor,
    cy: torch.Tensor,
    sy: torch.Tensor,
    cz: torch.Tensor,
    sz: torch.Tensor,
    charges: torch.Tensor,
    recip: torch.Tensor,
    capacity: tuple[int, int, int],
    dipoles: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full Miller-box ``S(n)`` as ``(real, imag)`` cubes."""
    dtype = charges.dtype
    sr, si = _separable_sum(cx, sx, cy, sy, cz, sz, charges)
    if dipoles is None:
        return sr, si
    alpha = (dipoles @ recip.transpose(0, 1)).to(dtype=dtype)
    nx = torch.arange(-capacity[0], capacity[0] + 1, device=charges.device, dtype=dtype)
    ny = torch.arange(-capacity[1], capacity[1] + 1, device=charges.device, dtype=dtype)
    nz = torch.arange(-capacity[2], capacity[2] + 1, device=charges.device, dtype=dtype)
    d0r, d0i = _separable_sum(cx, sx, cy, sy, cz, sz, alpha[:, 0])
    d1r, d1i = _separable_sum(cx, sx, cy, sy, cz, sz, alpha[:, 1])
    d2r, d2i = _separable_sum(cx, sx, cy, sy, cz, sz, alpha[:, 2])
    dip_r = nx.reshape(-1, 1, 1) * d0r + ny.reshape(1, -1, 1) * d1r + nz.reshape(1, 1, -1) * d2r
    dip_i = nx.reshape(-1, 1, 1) * d0i + ny.reshape(1, -1, 1) * d1i + nz.reshape(1, 1, -1) * d2i
    # ``S += i · dip``: ``(sr - dip_i, si + dip_r)``.
    return sr - dip_i, si + dip_r


def _gather_half_box(
    grid: torch.Tensor,
    n: torch.Tensor,
    capacity: tuple[int, int, int],
) -> torch.Tensor:
    """Pull half-box rows ``n`` out of a full Miller cube."""
    return grid[n[:, 0] + capacity[0], n[:, 1] + capacity[1], n[:, 2] + capacity[2]]


def _scatter_half_box(
    n: torch.Tensor,
    capacity: tuple[int, int, int],
    values: torch.Tensor,
) -> torch.Tensor:
    """Scatter half-box rows into a zero Miller cube. Other hemisphere stays 0."""
    kx = 2 * capacity[0] + 1
    ky = 2 * capacity[1] + 1
    kz = 2 * capacity[2] + 1
    out = values.new_zeros((kx, ky, kz, *values.shape[1:]))
    ix = n[:, 0] + capacity[0]
    iy = n[:, 1] + capacity[1]
    iz = n[:, 2] + capacity[2]
    return out.index_put((ix, iy, iz), values)


def structure_factor(
    pos: torch.Tensor,
    charges: torch.Tensor,
    n: torch.Tensor,
    recip: torch.Tensor,
    capacity: tuple[int, int, int],
    dipoles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structure factor ``S(k) = Σ_i (q_i + i k·μ_i) exp(i k · r_i)``.

    The structure factor is the Fourier transform of the multipole density: one
    complex number per wavevector, encoding where the charges and dipoles sit.
    It is returned as separate real and imaginary parts because the callers
    only ever need ``|S|²`` and products of it with real weights.

    Evaluated with LAMMPS ``eik_dot_r`` on the integer box ``capacity``, then
    gathered onto the half-box rows ``n``. Same SI V.6 sum as
    ``Σ_i (q_i + i k·μ_i) exp(i k·r_i)`` with ``k = n · recip``; the plane-wave
    matrix ``(N, M)`` is never built.

    Args:
        pos: Positions ``(N, 3)`` in Å.
        charges: Monopoles ``(N,)`` in e.
        n: Integer Miller indices ``(M, 3)`` from :attr:`KSpaceGrid.n`.
        recip: Reciprocal basis ``(3, 3)`` Å⁻¹ from :func:`reciprocal_basis`.
        capacity: Per-axis ``N_c`` from :attr:`KSpaceGrid.capacity`.
        dipoles: Optional Cartesian dipoles ``(N, 3)`` in e·Å. ``k·μ`` is then
            in e, so it adds to ``charge`` with no unit conversion.

    Returns:
        ``S_real (M,)``, ``S_imag (M,)``, both in e.
    """
    cx, sx, cy, sy, cz, sz = _miller_tables(pos, recip, capacity)
    sr, si = _structure_factor_grid(cx, sx, cy, sy, cz, sz, charges, recip, capacity, dipoles)
    return _gather_half_box(sr, n, capacity), _gather_half_box(si, n, capacity)


def kspace_phi_field(
    charges: torch.Tensor,
    positions: torch.Tensor,
    n: torch.Tensor,
    recip: torch.Tensor,
    capacity: tuple[int, int, int],
    k_sq_guarded: torch.Tensor,
    weights: torch.Tensor,
    sigma: torch.Tensor,
    pre: torch.Tensor,
    dipoles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-space potential ``φ`` and Cartesian ``+∇φ`` at each atom.

    Same GTO reciprocal sum as :func:`structure_factor` followed by the
    per-width form factor. Inverse transform is the separable conjugate of
    ``eik_dot_r`` — ``φ_i = Σ_n W(n) Re[S(n) exp(-i k·r_i)]`` — so ``(N, M)``
    is never materialised.

    Args:
        charges: Monopoles ``(N,)`` e.
        positions: ``(N, 3)`` Å.
        n: Integer Miller indices ``(M, 3)`` from :attr:`KSpaceGrid.n`.
        recip: Reciprocal basis ``(3, 3)`` Å⁻¹ from :func:`reciprocal_basis`.
        capacity: Per-axis ``N_c`` from :attr:`KSpaceGrid.capacity`.
        k_sq_guarded: Guarded ``|k|²`` ``(M,)`` Å⁻² from :meth:`KSpaceGrid.vectors`.
        weights: Half-box 0/2 weights ``(M,)``.
        sigma: Combined GTO widths ``(R,)`` or ``(1, R)`` Å.
        pre: Scalar ``eV_A · 4π / Ω``.
        dipoles: Optional Cartesian dipoles ``(N, 3)`` e·Å.

    Returns:
        ``phi (N, R)`` eV/e, ``field (N, 3, R)`` eV/(e·Å) as ``+∇φ``.
    """
    cx, sx, cy, sy, cz, sz = _miller_tables(positions, recip, capacity)
    sr, si = _structure_factor_grid(cx, sx, cy, sy, cz, sz, charges, recip, capacity, dipoles)
    guarded = k_sq_guarded.unsqueeze(-1)
    width = sigma.reshape(1, -1)
    kfac = torch.exp(-(width * width) * guarded) / guarded
    kernel = _scatter_half_box(n, capacity, weights.unsqueeze(-1) * kfac)
    pot_r = kernel * sr.unsqueeze(-1)
    pot_i = kernel * si.unsqueeze(-1)
    plane_r, plane_i = _cmul(cx.unsqueeze(2), -sx.unsqueeze(2), cy.unsqueeze(1), -sy.unsqueeze(1))
    inner_r = torch.einsum("nab,abcr->ncr", plane_r, pot_r) - torch.einsum(
        "nab,abcr->ncr", plane_i, pot_i
    )
    inner_i = torch.einsum("nab,abcr->ncr", plane_r, pot_i) + torch.einsum(
        "nab,abcr->ncr", plane_i, pot_r
    )
    cz_e = cz.unsqueeze(-1)
    sz_e = sz.unsqueeze(-1)
    phi = pre * (inner_r * cz_e + inner_i * sz_e).sum(dim=1)
    nx = torch.arange(-capacity[0], capacity[0] + 1, device=positions.device, dtype=positions.dtype)
    ny = torch.arange(-capacity[1], capacity[1] + 1, device=positions.device, dtype=positions.dtype)
    nz = torch.arange(-capacity[2], capacity[2] + 1, device=positions.device, dtype=positions.dtype)
    kvec = (
        nx.reshape(-1, 1, 1, 1) * recip[0]
        + ny.reshape(1, -1, 1, 1) * recip[1]
        + nz.reshape(1, 1, -1, 1) * recip[2]
    )
    wk_r = pot_r.unsqueeze(-2) * kvec.unsqueeze(-1)
    wk_i = pot_i.unsqueeze(-2) * kvec.unsqueeze(-1)
    inner_fr = torch.einsum("nab,abcdr->ncdr", plane_r, wk_r) - torch.einsum(
        "nab,abcdr->ncdr", plane_i, wk_i
    )
    inner_fi = torch.einsum("nab,abcdr->ncdr", plane_r, wk_i) + torch.einsum(
        "nab,abcdr->ncdr", plane_i, wk_r
    )
    cz_f = cz[:, :, None, None]
    sz_f = sz[:, :, None, None]
    field = pre * (inner_fi * cz_f - inner_fr * sz_f).sum(dim=1)
    return phi, field
