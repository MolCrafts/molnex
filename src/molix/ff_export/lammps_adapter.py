"""LAMMPS force-spec adapter (pure Python; no live ``lmp`` import).

Translates Class-I :class:`~molpot.ir.PotentialIR` into LAMMPS ``units real``
coeff records. Energy/length are already kcal/mol and Å in Class-I canonical
units; the load-bearing reparameterizations are:

* harmonic bond/angle: molnex ``E = ½ k x²`` → LAMMPS ``E = K x²`` with
  ``K = k/2``; angle ``theta0`` converted to degrees.
* proper torsions: multi-term cosine bags; when ``n∈{1..4}`` with OPLS phases
  they are also emitted as a compact ``dihedral_style opls`` ``K1..K4`` group.

References:
    LAMMPS docs: bond_style harmonic, angle_style harmonic, dihedral_style opls
    Spec: learnable-classical-ff-08-ff-export (peer of OpenMMAdapter)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.force_spec import ForceSpec
from molpot.ir import PotentialIR, ProperTorsionBag

__all__ = ["LammpsAdapter"]


class LammpsAdapter(BackendAdapter):
    """Translate Class-I IR into LAMMPS ``units real`` force-spec records.

    Does **not** invoke LAMMPS. Unit tests pin hard-coded numeric goldens
    (bond ``k=100`` → LAMMPS ``K=50``; OPLS phases → ``dihedral_coeff``).
    """

    name = "lammps"

    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Build a LAMMPS-oriented :class:`ForceSpec` from ``ir``.

        Args:
            ir: Class-I bags in kcal/mol, Å, e, rad.
            meta: Optional metadata merged into the force-spec.

        Returns:
            Force-spec with LAMMPS ``units real`` coeff conventions.
        """
        forces: list[dict[str, Any]] = []

        if ir.bonds is not None:
            forces.append(self._bonds(ir))
        if ir.angles is not None:
            forces.append(self._angles(ir))
        if ir.propers is not None:
            forces.extend(self._propers(ir))
        if ir.impropers_harmonic is not None:
            forces.append(self._impropers_harmonic(ir))
        if ir.impropers_periodic is not None:
            forces.append(self._impropers_periodic(ir))
        if ir.lj is not None:
            forces.append(self._lj(ir))
        if ir.charges is not None:
            forces.append(self._charges(ir))

        metadata: dict[str, Any] = {
            "unit_system_source": ir.unit_system,
            "unit_system_target": "lammps_real",
            "units": "real",
        }
        if meta:
            metadata.update(meta)

        return ForceSpec(
            backend=self.name,
            forces=forces,
            scaling=self.scaling_dict(ir.scaling),
            metadata=metadata,
        )

    def _bonds(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.bonds is not None
        k = self.as_floats(ir.bonds.k)
        r0 = self.as_floats(ir.bonds.r0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "r0": ri,
            }
            for i, (ki, ri) in enumerate(zip(k, r0, strict=True))
        ]
        return {
            "type": "bond_style harmonic",
            "style": "harmonic",
            "category": "bond",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _angles(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.angles is not None
        k = self.as_floats(ir.angles.k)
        theta0 = self.as_floats(ir.angles.theta0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "theta0_deg": math.degrees(ti),
            }
            for i, (ki, ti) in enumerate(zip(k, theta0, strict=True))
        ]
        return {
            "type": "angle_style harmonic",
            "style": "harmonic",
            "category": "angle",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _propers(self, ir: PotentialIR) -> list[dict[str, Any]]:
        assert ir.propers is not None
        bag = ir.propers
        if self._is_opls_manifold(bag):
            return [self._propers_opls(bag)]
        return [self._propers_fourier(bag)]

    def _propers_opls(self, bag: ProperTorsionBag) -> dict[str, Any]:
        n_types = bag.k.shape[0]
        periods = self.as_ints(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"proper idivf[{t}] must be non-zero")
            K = [0.0, 0.0, 0.0, 0.0]
            for m, n_per in enumerate(periods):
                k_tm = float(bag.k[t, m].item()) / idivf
                if n_per in (1, 2, 3, 4):
                    K[n_per - 1] += 2.0 * k_tm
            params.append(
                {
                    "type_index": t + 1,
                    "K1": K[0],
                    "K2": K[1],
                    "K3": K[2],
                    "K4": K[3],
                }
            )
        return {
            "type": "dihedral_style opls",
            "style": "opls",
            "category": "dihedral",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _propers_fourier(self, bag: ProperTorsionBag) -> dict[str, Any]:
        n_types, n_terms = bag.k.shape
        periods = self.as_ints(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"proper idivf[{t}] must be non-zero")
            terms = [
                {
                    "periodicity": periods[m],
                    "k": float(bag.k[t, m].item()) / idivf,
                    "phase": float(bag.phase[t, m].item()),
                }
                for m in range(n_terms)
            ]
            params.append({"type_index": t + 1, "terms": terms})
        return {
            "type": "dihedral_style fourier",
            "style": "fourier",
            "category": "dihedral",
            "case": TranslationCase.DECOMPOSE.value,
            "parameters": params,
        }

    @staticmethod
    def _is_opls_manifold(bag: ProperTorsionBag) -> bool:
        """True when propers are exactly n=1..4 with OPLS phases (0, π, 0, π)."""
        periods = bag.periodicity.detach().long().tolist()
        if periods != [1, 2, 3, 4]:
            return False
        expected = torch.tensor(
            [0.0, math.pi, 0.0, math.pi],
            device=bag.phase.device,
            dtype=bag.phase.dtype,
        )
        delta = torch.remainder(bag.phase - expected.unsqueeze(0) + math.pi, 2 * math.pi) - math.pi
        return bool(torch.all(torch.abs(delta) < 1e-6))

    def _impropers_harmonic(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.impropers_harmonic is not None
        bag = ir.impropers_harmonic
        k = self.as_floats(bag.k)
        chi0 = self.as_floats(bag.chi0)
        params = [
            {
                "type_index": i + 1,
                "K": 0.5 * ki,
                "chi0_deg": math.degrees(ci),
            }
            for i, (ki, ci) in enumerate(zip(k, chi0, strict=True))
        ]
        return {
            "type": "improper_style harmonic",
            "style": "harmonic",
            "category": "improper",
            "case": TranslationCase.FORM_REPARAMETERIZE.value,
            "parameters": params,
        }

    def _impropers_periodic(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.impropers_periodic is not None
        bag = ir.impropers_periodic
        n_types, n_terms = bag.k.shape
        periods = self.as_ints(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            terms = [
                {
                    "periodicity": periods[m],
                    "k": float(bag.k[t, m].item()) / idivf,
                    "phase": float(bag.phase[t, m].item()),
                }
                for m in range(n_terms)
            ]
            params.append({"type_index": t + 1, "terms": terms})
        return {
            "type": "improper_style cvff",
            "style": "cvff",
            "category": "improper",
            "case": TranslationCase.DECOMPOSE.value,
            "parameters": params,
        }

    def _lj(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.lj is not None
        eps = self.as_floats(ir.lj.epsilon)
        sig = self.as_floats(ir.lj.sigma)
        params = [
            {"type_index": i + 1, "epsilon": e, "sigma": s}
            for i, (e, s) in enumerate(zip(eps, sig, strict=True))
        ]
        return {
            "type": "pair_style lj/cut",
            "style": "lj/cut",
            "category": "pair",
            "case": TranslationCase.DIRECT_UNIT_SCALE.value,
            "parameters": params,
        }

    def _charges(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.charges is not None
        q = self.as_floats(ir.charges.q)
        return {
            "type": "atom charges",
            "category": "charge",
            "case": TranslationCase.DIRECT_UNIT_SCALE.value,
            "parameters": [{"atom_index": i + 1, "q": qi} for i, qi in enumerate(q)],
        }
