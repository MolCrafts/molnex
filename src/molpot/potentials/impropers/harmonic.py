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

    def forward(
        self,
        *,
        pos: torch.Tensor,
        improper_index: torch.Tensor,
        improper_types: torch.Tensor | None = None,
        term_batch: torch.Tensor | None = None,
        num_graphs: int,
        k: torch.Tensor | None = None,
        chi0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute harmonic improper energy.

        Args:
            pos: Positions ``[n_atoms, 3]``.
            improper_index: COO improper indices ``[4, N]`` molrs layout
                ``[center, i, j, k]`` (center at **row 0**).
            improper_types: Improper types ``[N]``, or ``None`` when the tables
                already carry one row per improper.
            term_batch: Graph index per improper ``[N]``; ``None`` sums to a
                scalar.
            num_graphs: Graph count ``B`` — the length of the per-graph energy.
            k: Per-call force-constant table; ``None`` prices the registered one.
            chi0: Per-call equilibrium-angle table; ``None`` prices the
                registered one.

        Returns:
            Per-graph energy ``(B,)`` when ``term_batch`` is given, else a scalar.

        Raises:
            ValueError: ``improper_index`` is not ``[4, N]``.
        """
        _require_index_4_by_n(improper_index, "improper_index")
        tables = {"k": k, "chi0": chi0}

        chi = _dihedral_phi(pos, _improper_to_dihedral_index(improper_index))
        k_imp = self.rows("k", tables, improper_types)
        chi0_imp = self.rows("chi0", tables, improper_types)
        energy_per = 0.5 * k_imp * (chi - chi0_imp) ** 2
        return self.per_graph(energy_per, term_batch, num_graphs)

    def __repr__(self) -> str:
        return f"ImproperHarmonic(n_types={len(self.k)})"
