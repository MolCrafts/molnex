"""Paper-units domain contract for ``WaterLESSource``.

Acceptance trace
----------------
* ac-008 → ``TestPaperUnits``

The LES paper (Cheng 2025, npj Comput. Mater. 11:80, doi:10.1038/s41524-025-01577-7)
ships RPBE-D3 bulk-water training data in **eV / eV·Å⁻¹ / Å** with energies near
``-10 eV/atom`` (so a 64-H₂O / 192-atom configuration sits near ``-2000 eV``).
``WaterLESSource`` must be a pass-through over the extxyz file's numeric values
— no implicit unit conversion, no cell scaling — so that downstream consumers
see the same numbers the paper's training pipeline saw.

The conftest fixture now uses paper-realistic per-atom energies (``-10 eV/atom``
± per-frame jitter, forces in ``[-1, 1] eV·Å⁻¹``, ``12 Å`` cubic cell). The
tests below pin the round-trip to those paper-scale ranges so any future
regression that drops a unit conversion in (e.g. by reading energies in
Hartree or forces in Hartree/Bohr) fails loudly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molix.datasets import WaterLESSource


class TestPaperUnits:
    def test_energy_in_eV_paper_scale(self, water_les_root: Path) -> None:
        """ac-008: energy is eV-scale per the LES paper (≈ -10 eV/atom for water)."""
        src = WaterLESSource(water_les_root, split="test")
        for i in range(len(src)):
            sample = src[i]
            n_atoms = int(sample["Z"].shape[0])
            e_per_atom = sample["energy"] / n_atoms
            # Healthy RPBE-D3 water sits near -10 eV/atom; the fixture's
            # `-60 eV` (2 H₂O = 6 atoms) lands at exactly that scale.
            # A loader that silently dropped a Hartree→eV factor would
            # land near -370 eV/atom; one that dropped a kJ/mol→eV
            # factor would land near -0.1 eV/atom. Either fails this.
            assert -20.0 < e_per_atom < -5.0, (
                f"sample {i} energy {sample['energy']} eV → "
                f"{e_per_atom:.3f} eV/atom; outside paper-scale "
                f"window [-20, -5] eV/atom"
            )
            assert isinstance(sample["energy"], float)

    def test_forces_in_eV_per_Angstrom(self, water_les_root: Path) -> None:
        """ac-008: forces are eV·Å⁻¹-scale (≤ ~50 eV/Å for healthy water frames)."""
        src = WaterLESSource(water_les_root, split="test")
        for i in range(len(src)):
            sample = src[i]
            f_max = sample["forces"].abs().max().item()
            assert f_max <= 50.0, (
                f"sample {i} max|F|={f_max:.4f} eV·Å⁻¹; exceeds the "
                f"50 eV·Å⁻¹ sustained-force ceiling for healthy water frames"
            )
            assert sample["forces"].dtype == torch.float32

    def test_cell_in_Angstrom_paper_scale(self, water_les_root: Path) -> None:
        """ac-008: cell is Å-scale (paper uses ~12 Å cubic for 64 H₂O)."""
        src = WaterLESSource(water_les_root, split="test")
        for i in range(len(src)):
            sample = src[i]
            cell = sample["cell"]
            assert cell.shape == (3, 3)
            edge_lengths = cell.norm(dim=1)
            # Paper cell ≈ 12.4 Å for 64 H₂O at ambient density; the fixture
            # uses 12.0 Å edges. A loader that silently dropped a Bohr→Å
            # factor (×0.529) would land at ~6.3 Å; one that scaled by Å→nm
            # would land at 0.12. Pin the eV-scale-equivalent window.
            for edge in edge_lengths:
                assert 5.0 < edge.item() < 100.0, (
                    f"sample {i} cell edge {edge.item():.4f} Å outside "
                    f"physically plausible water-box window [5, 100] Å"
                )

    def test_loader_does_not_scale_cell(self, water_les_root: Path) -> None:
        """ac-008: cell passes through verbatim — no implicit unit scaling.

        The fixture writes ``Lattice="12.0 0 0 0 12.0 0 0 0 12.0"`` (Å,
        cubic). Loaded ``cell`` must equal ``12·I₃`` to within float32 ulp.
        """
        src = WaterLESSource(water_les_root, split="test")
        expected = torch.eye(3, dtype=torch.float32) * 12.0
        for i in range(len(src)):
            sample = src[i]
            assert torch.allclose(sample["cell"], expected, atol=1e-4)

    def test_loader_does_not_scale_energy(self, water_les_root: Path) -> None:
        """ac-008: energy passes through verbatim — no implicit unit scaling.

        The fixture writes ``energy = -60.0 - 0.1·i`` (eV, paper-scaled
        per-atom). Loaded ``energy`` must equal the written value to
        Python-float precision.
        """
        src = WaterLESSource(water_les_root, split="test")
        for i in range(len(src)):
            assert src[i]["energy"] == pytest.approx(-60.0 - 0.1 * i, abs=1e-4)

    def test_loader_does_not_scale_forces(self, water_les_root: Path) -> None:
        """ac-008: forces pass through verbatim — no implicit unit scaling.

        The fixture writes forces drawn from ``uniform(-1, 1) eV·Å⁻¹``.
        Loaded ``|forces|.max()`` must therefore stay strictly within that
        bound (allowing one float32 ulp).
        """
        src = WaterLESSource(water_les_root, split="test")
        for i in range(len(src)):
            f_max = src[i]["forces"].abs().max().item()
            assert f_max <= 1.0 + 1e-6, (
                f"sample {i} |F|.max={f_max} > 1 + ulp; loader applied an unexpected scaling"
            )
