"""Per-atom energy heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from molpy.core import UnitsError

from molix import config
from molix.units import units
from molpot.derivation.energy import EnergyAggregation

_SCHEMES = ("isolated_atom", "combined_hof_linear", "empirical_fit")


def _require_scheme(scheme: str) -> str:
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
    return scheme


def _require_energy_units(name: str) -> str:
    """Resolve ``name`` through :data:`molix.units.units`; do not convert ``c[Z]``."""
    try:
        (1.0 * units.parse(name)).to(units.parse("eV"))
    except UnitsError as exc:
        raise ValueError(f"unknown energy unit {name!r}") from exc
    return name


class AtomicReferenceEnergy(nn.Module):
    """Frozen Z-indexed map ``c[Z]`` (eV, float64), additive reference energy.

    Stores a dense lookup ``atomic_energies[Z]`` always in float64, independent
    of the model dtype, plus a boolean ``supported`` mask. A stored ``0.0``
    is a legal coefficient only when ``supported[Z]`` is true. The module does
    not fit ``c``; it applies a published or already-fitted table.

    ``scheme`` is provenance only and does not change arithmetic:

    * ``isolated_atom`` — MACE E0 (default; the ``MACEEncoder`` call).
    * ``combined_hof_linear`` — published HOF+linear tables such as OMol25
      ``omol_elem_refs``.
    * ``empirical_fit`` — a table fitted elsewhere.

    ``units`` defaults to ``eV`` and is likewise provenance: names resolve
    through :data:`molix.units.units` and unknown energy units raise; values
    are not converted.

    Support vs structural mask: ``supported`` answers "does this *element*
    have a defined ``c[Z]``?". ``atoms.mask`` answers "is this *row* a real
    atom?". Ghost rows from ``MoleculePadder(pad_atom_type=1)`` carry
    hydrogen, which is a supported element on any H-containing table, so they
    contribute unless the caller passes a structural mask. ``forward`` is the
    compile-clean index (optional mask is a multiply, never a slice) and does
    **not** call :meth:`validate`. :meth:`pooled` validates only when eager
    (``not torch.compiler.is_compiling()``), then sums with
    :class:`~molpot.derivation.energy.EnergyAggregation`. ``∂e0/∂r = 0``:
    neither method reads positions.

    Args:
        atomic_energies: Element-table-ordered coefficients (with
            ``atomic_numbers``) or a dense Z-indexed vector. Cast to float64
            once at construction.
        atomic_numbers: Z-table mapping table position → atomic number.
            Scatters ``atomic_energies`` into the dense lookup; ``supported``
            is true exactly at those Z. Duplicate Z raises. Mutually exclusive
            with ``supported``.
        max_z: Optional explicit table size ``max_z + 1`` on the scatter path;
            defaults to the largest given ``Z``.
        supported: Required on the dense path (no ``atomic_numbers``). Same
            length as the dense vector. Dense zeros do not imply support.
        scheme: One of ``isolated_atom``, ``combined_hof_linear``,
            ``empirical_fit``.
        units: Energy unit name, default ``eV``.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
        Levine et al. "OMol25: the Open Molecules 2025 dataset"
        https://arxiv.org/html/2505.08762v1
        Tran et al. "The Open Catalyst 2022 (OC22) dataset"
        https://arxiv.org/html/2206.08917 eqs (158)–(159)
        FAIR ``omol_direct.yaml`` (``fairchem@41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9``)
        stores the table as ``torch.DoubleTensor``.
    """

    def __init__(
        self,
        *,
        atomic_energies: torch.Tensor | Sequence[float],
        atomic_numbers: torch.Tensor | Sequence[int] | None = None,
        max_z: int | None = None,
        supported: torch.Tensor | Sequence[bool] | None = None,
        scheme: str = "isolated_atom",
        units: str = "eV",
    ) -> None:
        super().__init__()
        self.scheme = _require_scheme(scheme)
        self.units = _require_energy_units(units)
        e0 = torch.as_tensor(atomic_energies, dtype=torch.float64).flatten()

        if atomic_numbers is not None and supported is not None:
            raise ValueError("atomic_numbers and supported cannot both be given")

        if atomic_numbers is not None:
            z = torch.as_tensor(atomic_numbers).flatten().to(device=e0.device, dtype=torch.long)
            if z.shape != e0.shape:
                raise ValueError(
                    f"atomic_energies ({tuple(e0.shape)}) and atomic_numbers "
                    f"({tuple(z.shape)}) must have the same length."
                )
            uniq, counts = torch.unique(z, return_counts=True)
            dups = uniq[counts > 1]
            if dups.numel():
                raise ValueError(f"duplicate atomic numbers {dups.tolist()}")
            size = (int(max_z) if max_z is not None else int(z.max().item())) + 1
            lookup = torch.zeros(size, dtype=torch.float64, device=e0.device)
            lookup[z] = e0
            support = torch.zeros(size, dtype=torch.bool, device=e0.device)
            support[z] = True
        else:
            if supported is None:
                raise ValueError(
                    "supported is required when atomic_numbers is omitted; "
                    "dense zeros do not imply support"
                )
            lookup = e0
            support = (
                torch.as_tensor(supported, dtype=torch.bool).flatten().to(device=lookup.device)
            )
            if support.shape != lookup.shape:
                raise ValueError(
                    f"supported ({tuple(support.shape)}) and atomic_energies "
                    f"({tuple(lookup.shape)}) must have the same length."
                )

        self.register_buffer("atomic_energies", lookup)
        self.atomic_energies: torch.Tensor
        self.register_buffer("supported", support)
        self.supported: torch.Tensor
        self._pool = EnergyAggregation(pooling="sum")

    def validate(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Raise if any unmasked ``Z`` is out of range or unsupported.

        Eager host-sync seam, not a hot-path method. Masked-out rows are not
        checked for support. Every ``Z``, masked or not, must lie in
        ``[0, table_size)`` so the lookup is a legal index — out-of-range with
        a mask still raises. Offending numbers are not clipped onto ``max_z``.

        Args:
            Z: Atomic numbers ``(N,)``.
            mask: Optional structural mask ``(N,)``; ``True`` / nonzero rows
                are real atoms.

        Raises:
            ValueError: Listing the unique offending atomic numbers.
        """
        z = Z.to(torch.long)
        n_table = self.atomic_energies.shape[0]
        out_of_range = (z < 0) | (z >= n_table)
        if mask is None:
            live = torch.ones(z.shape, dtype=torch.bool, device=z.device)
        else:
            live = mask.to(dtype=torch.bool)
        # Clamp is only a legal index for the support gather; reported Z is raw.
        safe_z = z.clamp(min=0, max=max(n_table - 1, 0))
        unsupported = live & ~out_of_range & ~self.supported[safe_z]
        unknown = torch.unique(z[out_of_range | unsupported])
        if unknown.numel():
            raise ValueError(
                f"atomic numbers {unknown.tolist()} are unsupported or out of range "
                f"for this {n_table}-slot reference table"
            )

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-atom reference energy ``c[Z]`` as float64 ``(N,)``.

        Compile-clean index into :attr:`atomic_energies`. Optional ``mask`` is
        a multiply, never a slice. Does **not** call :meth:`validate`.

        Args:
            Z: Atomic numbers ``(N,)`` (integer).
            mask: Optional structural mask ``(N,)``.

        Returns:
            Per-atom reference energies ``(N,)``, float64.
        """
        values = self.atomic_energies[Z.to(torch.long)]
        if mask is not None:
            values = values * mask.to(dtype=values.dtype)
        return values

    def pooled(
        self,
        Z: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``Σ_i c[Z_i]`` per graph (atomic dressing), float64 ``(B,)``.

        Eager (``not torch.compiler.is_compiling()``) calls :meth:`validate`
        then :class:`~molpot.derivation.energy.EnergyAggregation` on
        :meth:`forward`. Without ``mask``, every given row is summed, including
        ghost hydrogen.

        Args:
            Z: Atomic numbers ``(N,)``.
            batch: Graph membership ``(N,)``.
            num_graphs: Graph count ``B``.
            mask: Optional structural mask ``(N,)``.

        Returns:
            Dressed baseline ``(B,)``, float64.
        """
        if not torch.compiler.is_compiling():
            self.validate(Z, mask=mask)
        return self._pool(self.forward(Z, mask=mask), batch.long(), num_graphs)

    def get_extra_state(self) -> dict[str, str]:
        """Return ``scheme`` and ``units`` for ``state_dict`` round-trip."""
        return {"scheme": self.scheme, "units": self.units}

    def set_extra_state(self, state: Mapping[str, str]) -> None:
        """Restore ``scheme`` and ``units`` from :meth:`get_extra_state`."""
        self.scheme = _require_scheme(state["scheme"])
        self.units = _require_energy_units(state["units"])

    def _apply(self, *args, **kwargs):
        """Recast ``atomic_energies`` to float64 after ``.to`` / ``.float``."""
        module = super()._apply(*args, **kwargs)
        module.atomic_energies = module.atomic_energies.to(dtype=torch.float64)
        return module

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Recast ``atomic_energies`` to float64; ``load_state_dict`` skips ``_apply``.

        Day-1 Class-I checkpoints predate ``supported`` and ``_extra_state``.
        Fill them from the constructed table / current scheme so resume is
        not a missing-key error.
        """
        supported_key = prefix + "supported"
        if supported_key not in state_dict:
            state_dict[supported_key] = self.supported
        extra_key = prefix + "_extra_state"
        if extra_key not in state_dict:
            state_dict[extra_key] = self.get_extra_state()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.atomic_energies = self.atomic_energies.to(dtype=torch.float64)

    def extra_repr(self) -> str:
        """Render table size, scheme, and units for ``repr(module)``."""
        return f"num_z={self.atomic_energies.shape[0]}, scheme={self.scheme}, units={self.units}"


class AtomicEnergyMLP(nn.Module):
    """MLP predicting per-atom energy from atomic features.

    Single responsibility: map atomic feature vectors to scalar
    per-atom energy values. Does NOT perform graph-level pooling.
    Use ``molpot.derivation.EnergyAggregation`` for pooling.

    Args:
        hidden_dim: Dimension of input hidden representation.
    """

    def __init__(self, hidden_dim: int = 64, *, dtype: torch.dtype | None = None):
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, atoms_h: torch.Tensor) -> torch.Tensor:
        """Predict per-atom energies.

        Args:
            atoms_h: Atomic hidden states ``(N, D)``.

        Returns:
            Per-atom energies ``(N,)``.
        """
        return self.mlp(atoms_h).squeeze(-1)


class EnergyHead(nn.Module):
    """Predict molecular energy from atomic representations.

    Bundles an atomic MLP and sum pooling. For new code, prefer
    composing ``AtomicEnergyMLP`` + ``molpot.derivation.EnergyAggregation``
    explicitly.
    """

    def __init__(self, hidden_dim: int = 64, *, dtype: torch.dtype | None = None):
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.atomic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, atoms_h: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
        """Predict molecular energy.

        Args:
            atoms_h: Atomic hidden states ``(N, D)``.
            graph_batch: Molecule indices ``(N,)``.

        Returns:
            Molecular energies ``(B,)``.
        """
        atomic_energies = self.atomic_mlp(atoms_h).squeeze(-1)

        num_molecules = int(graph_batch.max()) + 1
        molecular_energies = torch.zeros(
            num_molecules, dtype=atomic_energies.dtype, device=atomic_energies.device
        )
        molecular_energies.index_add_(0, graph_batch, atomic_energies)

        return molecular_energies
