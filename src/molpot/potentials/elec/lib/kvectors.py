"""Reciprocal-space vector generation for Ewald and mesh-based methods."""

import torch
from torch.nn.utils.rnn import pad_sequence


class MeshShape:
    """The per-axis mesh point counts, in the two forms the mesh code reads them.

    A mesh library needs the counts twice over: as three Python ints (to size
    ``torch.zeros`` / ``fftfreq`` and to index), and as a ``(3,)`` tensor (to
    scale positions and mesh spacings). Which of the two is cheap depends on
    how the shape arrived, so this holds both and remembers which mode it is
    in:

    * **frozen** — the caller passed ``tuple[int, int, int]``, a shape fixed
      ahead of time by a ``bind``. :attr:`frozen` is that tuple, so every int
      read is a host-side constant and the region traces as a single graph.
    * **live** — the caller passed a ``(3,)`` tensor it sized off the cell.
      :attr:`frozen` is ``None`` and :attr:`counts` reads the tensor back to
      the host, which is a synchronisation and therefore breaks a compiled
      graph. That mode stays supported for callers that size their mesh per
      cell, and is only ever run eagerly.

    :attr:`tensor` is the arithmetic form in both modes. In frozen mode it is
    rebuilt only when the counts actually change, so a bound step allocates
    nothing and never copies host ints to the device.

    Args:
        ns: Mesh points per axis — a ``(3,)`` integer tensor, or three Python
            ints.
        device: Device the tensor form must live on (the cell's).
    """

    def __init__(self, ns: torch.Tensor | tuple[int, int, int], device: torch.device) -> None:
        self.frozen: tuple[int, int, int] | None = None
        self.tensor: torch.Tensor
        self.update(ns, device)

    def update(self, ns: torch.Tensor | tuple[int, int, int], device: torch.device) -> None:
        """Adopt ``ns``, reusing the tensor form when the frozen counts are unchanged.

        Args:
            ns: Mesh points per axis, tensor or three Python ints.
            device: Device a freshly built tensor form must live on.

        Raises:
            ValueError: A tensor ``ns`` is not ``(3,)``, or an int ``ns`` does
                not have three entries.
        """
        if isinstance(ns, torch.Tensor):
            if ns.shape != (3,):
                raise ValueError(f"shape {list(ns.shape)} of `ns_mesh` has to be (3,)")
            self.frozen = None
            self.tensor = ns
            return

        if len(ns) != 3:
            raise ValueError(f"`ns_mesh` has {len(ns)} entries but has to have 3")
        counts = (int(ns[0]), int(ns[1]), int(ns[2]))
        if counts != self.frozen:
            self.frozen = counts
            self.tensor = torch.tensor(counts, dtype=torch.long, device=device)

    @property
    def counts(self) -> tuple[int, int, int]:
        """The three counts as Python ints.

        Frozen mode returns the ints the caller froze — free. Live mode reads
        them back off :attr:`tensor`, a device-to-host synchronisation; call it
        from eager code only.
        """
        if self.frozen is not None:
            return self.frozen
        return (int(self.tensor[0]), int(self.tensor[1]), int(self.tensor[2]))


def get_ns_mesh(cell: torch.Tensor, mesh_spacing: float) -> torch.Tensor:
    """Compute mesh size from target spacing, rounded to powers of 2 for FFT.

    Axis ``i`` of length ``‖a_i‖`` would need ``‖a_i‖ / h`` points to be sampled
    at spacing ``h``. The count is first taken to ``2 ‖a_i‖/h + 1`` so a feature
    of size ``h`` is sampled at least twice per period, then rounded **up** to a
    power of two — the length a fast Fourier transform (FFT, the ``O(n log n)``
    algorithm the mesh calculators convolve with) factorises fastest.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors.
        mesh_spacing: Target real-space spacing ``h`` between mesh points, in Å.
            Smaller means a finer grid and a more accurate reciprocal sum.

    Returns:
        Mesh dimensions ``(3,)`` as a long tensor, each entry a power of two.
    """
    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_approx = basis_norms / mesh_spacing
    ns_actual_approx = 2 * ns_approx + 1
    return torch.tensor(2).pow(torch.ceil(torch.log2(ns_actual_approx)).long())


def get_ns_ewald(cell: torch.Tensor, lr_wavelength: float) -> torch.Tensor:
    """Compute the explicit Ewald k-vector count per axis from a target wavelength.

    Every k-vector down to ``lr_wavelength`` is kept: with the cutoff
    ``k_cutoff = 2π/λ`` and ``a_i · b_j = 2π δ_ij``, axis ``i`` needs
    ``ceil(‖a_i‖ k_cutoff / 2π)`` vectors.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors.
        lr_wavelength: Shortest reciprocal-space wavelength resolved, in Å.

    Returns:
        K-vector counts ``(3,)`` as a long tensor on ``cell``'s device.
    """
    k_cutoff = 2 * torch.pi / lr_wavelength
    basis_norms = torch.linalg.norm(cell, dim=1)
    return torch.ceil(k_cutoff * basis_norms / 2 / torch.pi).long()


def _generate_kvectors(
    cell: torch.Tensor,
    ns: torch.Tensor | tuple[int, int, int],
    for_ewald: bool,
) -> torch.Tensor:
    """Generate k-vectors on a mesh grid for FFT or Ewald summation.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å.
        ns: Grid counts per axis. A ``tuple[int, int, int]`` is a shape frozen
            by the caller ahead of time — three Python ints cost nothing to
            read, so the grid below is built by plain factory calls and the sum
            traces as a single graph. A ``(3,)`` integer tensor is a shape the
            caller sized off the live cell; reading it back to the host is a
            synchronisation, and that read is exactly what breaks a compiled
            graph. It stays supported because the mesh calculators hold
            ``ns_mesh`` as a tensor.
        for_ewald: If True, return full explicit grid; otherwise FFT-optimized.

    Returns:
        K-vectors with shape ``(nx, ny, nz, 3)`` for mesh,
        or ``(n_kvecs, 3)`` for Ewald, in Å⁻¹.

    Raises:
        ValueError: ``cell`` is not ``(3, 3)``, or a tensor ``ns`` is not
            ``(3,)`` or lives on another device than ``cell``.
    """
    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if isinstance(ns, torch.Tensor):
        if ns.shape != (3,):
            raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

        if ns.device != cell.device:
            raise ValueError(
                f"`ns` and `cell` are not on the same device, got {ns.device} and {cell.device}."
            )

        # Host read of a device tensor — the graph break the int form avoids.
        nx, ny, nz = (int(n) for n in ns)
    else:
        nx, ny, nz = ns

    if cell.is_cuda:
        inverse_cell = torch.linalg.inv_ex(cell)[0]
    else:
        inverse_cell = torch.linalg.inv(cell)

    reciprocal_cell = 2 * torch.pi * inverse_cell.T
    bx = reciprocal_cell[0]
    by = reciprocal_cell[1]
    bz = reciprocal_cell[2]

    kxs = (bx * nx) * torch.fft.fftfreq(nx, device=cell.device, dtype=cell.dtype).unsqueeze(-1)
    kys = (by * ny) * torch.fft.fftfreq(ny, device=cell.device, dtype=cell.dtype).unsqueeze(-1)

    if for_ewald:
        kzs = (bz * nz) * torch.fft.fftfreq(nz, device=cell.device, dtype=cell.dtype).unsqueeze(-1)
    else:
        kzs = (bz * nz) * torch.fft.rfftfreq(nz, device=cell.device, dtype=cell.dtype).unsqueeze(-1)

    return kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]


def generate_kvectors_for_mesh(
    cell: torch.Tensor,
    ns: torch.Tensor | tuple[int, int, int],
) -> torch.Tensor:
    """Compute reciprocal-space vectors for FFT-based mesh calculators.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å, rows being the lattice vectors.
        ns: Mesh dimensions ``(3,)`` (preferably powers of 2), as a tensor or
            as three Python ints — see :func:`_generate_kvectors`.

    Returns:
        K-vectors ``(nx, ny, nz//2+1, 3)`` in Å⁻¹, with
        ``k_vectors[0,0,0] = [0,0,0]``. The last axis is halved because a real
        input has a conjugate-symmetric transform, so ``rfftn`` stores only
        half of it.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=False)


def generate_kvectors_for_ewald(
    cell: torch.Tensor,
    ns: torch.Tensor | tuple[int, int, int],
) -> torch.Tensor:
    """Compute all reciprocal-space vectors for explicit Ewald summation.

    Args:
        cell: Unit cell matrix ``(3, 3)`` in Å — read live, so the vectors
            follow a lattice that moves under a barostat.
        ns: Number of k-vectors along each axis, as a ``(3,)`` tensor or as
            three Python ints. The int form is the frozen shape a
            ``bind_cell`` fixed once and the only one that traces without a
            graph break — see :func:`_generate_kvectors`.

    Returns:
        K-vectors ``(n_kvecs, 3)`` in Å⁻¹, with ``k_vectors[0] = [0,0,0]``.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=True).reshape(-1, 3)


def compute_batched_kvectors(
    lr_wavelength: float,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Generate k-vectors for multiple systems in a batch.

    Each cell is sized independently by :func:`get_ns_ewald`, so the ``B``
    systems generally yield different k-vector counts; the shorter ones are
    zero-padded up to the longest.

    Args:
        lr_wavelength: Shortest reciprocal-space wavelength resolved, in Å.
        cells: Unit cell matrices ``(B, 3, 3)`` in Å, rows being the lattice
            vectors of each system.

    Returns:
        Padded k-vectors ``(B, max_n_kvecs, 3)`` in Å⁻¹, zero-padded past each
        system's own count.
    """
    all_kvectors = []
    for cell in cells:
        kvectors = generate_kvectors_for_ewald(ns=get_ns_ewald(cell, lr_wavelength), cell=cell)
        all_kvectors.append(kvectors)
    return pad_sequence(all_kvectors, batch_first=True)
