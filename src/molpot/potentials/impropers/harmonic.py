"""Harmonic improper torsion: ``E = ½ k (χ − χ₀)²``.

``improper_index [4, N]`` uses molrs layout ``[center, i, j, k]`` (center at
**row 0**). χ is the dihedral angle of the ordered quartet
``(i, center, j, k)``.

This is the CHARMM / some-SMIRNOFF harmonic improper form — **not** a Class-I
proper torsion. For AMBER/GAFF/SMIRNOFF propers use
:class:`~molpot.potentials.dihedrals.periodic.ProperTorsionPeriodic`.

References:
    OpenMM User Guide §19.5 (harmonic impropers)
    CHARMM force field improper form
    molrs ``Topology`` impropers ``[center, i, j, k]``
"""

from __future__ import annotations

from typing import Any

import torch

from molpot.potentials.base import BasePotential
from molpot.potentials.dihedrals.periodic import _dihedral_phi, _require_index_4_by_n
from molpot.potentials.impropers.periodic import _improper_to_dihedral_index


class ImproperHarmonic(BasePotential):
    """Harmonic improper torsion potential.

    Energy formula::

        E = 0.5 * k * (χ − χ₀)²

    Parameters are type-indexed vectors:

    - ``k``: force constants ``[n_types]``
    - ``chi0``: equilibrium improper angles in radians ``[n_types]``

    Central atom is at **row index 0** of ``improper_index`` (molrs layout).

    Attributes:
        k: Force constants ``[n_types]``.
        chi0: Equilibrium improper angles in radians ``[n_types]``.
    """

    name = "improper_harmonic_torch"
    type = "improper"

    k: torch.Tensor
    chi0: torch.Tensor

    def __init__(self, k: torch.Tensor, chi0: torch.Tensor) -> None:
        """Initialize ImproperHarmonic.

        Args:
            k: Force constant vector ``[n_types]``.
            chi0: Equilibrium improper angles in radians ``[n_types]``.
        """
        super().__init__()

        if k.shape != chi0.shape:
            raise ValueError(
                f"k and chi0 must have same shape, got k: {k.shape}, chi0: {chi0.shape}"
            )
        if k.ndim != 1:
            raise ValueError(f"k must be 1D vector [n_types], got shape {k.shape}")

        self.register_buffer("k", k)
        self.register_buffer("chi0", chi0)

    def forward(self, data: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute harmonic improper energy.

        Args:
            data: Optional dictionary with molecular fields.
            **kwargs: Explicit tensors:
                - pos: Positions ``[n_atoms, 3]``.
                - improper_index: COO improper indices ``[4, N]`` molrs layout
                  ``[center, i, j, k]`` (center at **row 0**).
                - improper_types: Improper types ``[N]``.

        Returns:
            Total improper energy (scalar).

        Raises:
            ValueError: Missing inputs or ``improper_index`` not ``[4, N]``.
        """
        pos = kwargs.get("pos")
        improper_index = kwargs.get("improper_index")
        improper_types = kwargs.get("improper_types")

        if data is not None and isinstance(data, dict):
            if pos is None:
                pos = data.get("pos")
                if pos is None and isinstance(data.get("atoms"), dict):
                    pos = data["atoms"].get("pos")
            if improper_index is None:
                improper_index = data.get("improper_index")
            if improper_types is None:
                improper_types = data.get("improper_types")

        if pos is None or improper_index is None or improper_types is None:
            raise ValueError("ImproperHarmonic requires pos, improper_index, and improper_types.")

        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            improper_index = torch.from_numpy(improper_index).long()
            improper_types = torch.from_numpy(improper_types).long()

        _require_index_4_by_n(improper_index, "improper_index")

        if improper_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        chi = _dihedral_phi(pos, _improper_to_dihedral_index(improper_index))
        k_imp = self.k[improper_types]
        chi0_imp = self.chi0[improper_types]
        energy_per = 0.5 * k_imp * (chi - chi0_imp) ** 2
        return energy_per.sum()

    def __repr__(self) -> str:
        return f"ImproperHarmonic(n_types={len(self.k)})"
