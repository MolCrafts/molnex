"""Optional ASE-Calculator shell over the PiNet force seam.

ASE is an optional dependency: this module degrades gracefully when it is absent
(``HAS_ASE`` is ``False`` and :func:`make_pinet_calculator` raises). The in-process
integrator remains the primary path; the calculator is a thin wrapper that reuses
the same force seam so external ASE drivers can call PiNet.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from molix.md.forcefield import PotentialForceField

try:
    from ase.calculators.calculator import Calculator

    HAS_ASE = True
except ImportError:  # pragma: no cover - exercised only where ASE is absent
    HAS_ASE = False


def make_pinet_calculator(model: nn.Module, template: TensorDict):
    """Build an ASE ``Calculator`` delegating energy/forces to the PiNet force seam.

    Args:
        model: A ``PiNetPotential``.
        template: Molecule TensorDict carrying topology (positions are overwritten
            per ``calculate`` from the ASE ``Atoms``).

    Returns:
        An ASE ``Calculator`` instance.

    Raises:
        RuntimeError: If ASE is not importable.
    """
    if not HAS_ASE:
        raise RuntimeError("ASE is not installed; use the in-process integrator instead.")

    force = PotentialForceField(model, template)
    ref_pos = template["atoms", "pos"]

    class PiNetCalculator(Calculator):  # type: ignore[misc, valid-type]
        implemented_properties = ("energy", "forces")  # noqa: RUF012

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):  # noqa: ANN001, ANN204
            super().calculate(atoms, properties, system_changes or [])
            pos = torch.as_tensor(
                self.atoms.get_positions(), dtype=ref_pos.dtype, device=ref_pos.device
            )
            out = force(pos)
            self.results["energy"] = float(out.energy)
            self.results["forces"] = out.forces.cpu().numpy()

    return PiNetCalculator()
