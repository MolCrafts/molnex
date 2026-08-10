"""OpenMM force-spec adapter (pure Python; no live ``openmm`` import).

Emits JSON-friendly records matching OpenMM User Guide §19 force names:

* ``HarmonicBondForce`` — ``k`` kJ/mol/nm², ``r0`` nm
* ``HarmonicAngleForce`` — ``k`` kJ/mol/rad², ``theta0`` rad
* ``PeriodicTorsionForce`` — ``k`` kJ/mol, ``periodicity``, ``phase`` rad
* ``NonbondedForce`` — ``charge`` e, ``sigma`` nm, ``epsilon`` kJ/mol

Particle indices are **not** required for type-level IR bags; records carry
``type_index`` (and ``term_index`` for multi-term torsions). Topology binding
is a later consumer concern.

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-08-ff-export
"""

from __future__ import annotations

from typing import Any

import torch

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.conventions import (
    ConventionTable,
    scale_angle_k,
    scale_bond_k,
    scale_energy,
    scale_length,
    scale_torsion_k,
)
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.force_spec import ForceSpec
from molpot.ir import NonbondedScaling, PotentialIR

__all__ = ["OpenMMAdapter"]


def _as_float_list(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().reshape(-1).tolist()]


def _as_int_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.detach().cpu().reshape(-1).tolist()]


class OpenMMAdapter(BackendAdapter):
    """Translate Class-I :class:`~molpot.ir.PotentialIR` into OpenMM force-spec records.

    Does **not** import OpenMM. Unit tests and regressions pin hard-coded
    numeric goldens (bond ``k`` 100 → 41840; torsion Vn=2 → ``k`` 4.184).
    """

    name = "openmm"

    def __init__(self, conventions: ConventionTable | None = None) -> None:
        if conventions is None:
            conventions = ConventionTable.default_openmm()
        self.conventions = conventions

    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Build an OpenMM-oriented :class:`ForceSpec` from ``ir``.

        Args:
            ir: Class-I bags in kcal/mol, Å, e, rad.
            meta: Optional metadata merged into the force-spec.

        Returns:
            Force-spec with OpenMM units (kJ/mol, nm, e, rad).

        Raises:
            UnsupportedTermError: For bags marked
                :attr:`TranslationCase.UNSUPPORTED` (e.g. harmonic improper).
        """
        forces: list[dict[str, Any]] = []

        if ir.bonds is not None:
            forces.append(self._bonds(ir))
        if ir.angles is not None:
            forces.append(self._angles(ir))
        if ir.propers is not None:
            forces.append(self._propers(ir))
        if ir.impropers_periodic is not None:
            forces.append(self._impropers_periodic(ir))
        if ir.impropers_harmonic is not None:
            row = self.conventions.row("improper_harmonic")
            raise UnsupportedTermError(
                "improper_harmonic",
                row.notes or "no OpenMM built-in harmonic improper force",
                case=row.case,
            )
        if ir.lj is not None or ir.charges is not None:
            forces.append(self._nonbonded(ir))

        scaling = self._scaling_dict(ir.scaling)
        metadata: dict[str, Any] = {
            "unit_system_source": ir.unit_system,
            "unit_system_target": "openmm_standard",
        }
        if meta:
            metadata.update(meta)

        return ForceSpec(
            backend=self.name,
            forces=forces,
            scaling=scaling,
            metadata=metadata,
        )

    def _bonds(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.bonds is not None
        row = self.conventions.row("bond_harmonic")
        k = _as_float_list(ir.bonds.k)
        r0 = _as_float_list(ir.bonds.r0)
        params = [
            {
                "type_index": i,
                "k": scale_bond_k(ki),
                "r0": scale_length(ri),
            }
            for i, (ki, ri) in enumerate(zip(k, r0, strict=True))
        ]
        return {
            "type": "HarmonicBondForce",
            "case": row.case.value,
            "parameters": params,
        }

    def _angles(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.angles is not None
        row = self.conventions.row("angle_harmonic")
        k = _as_float_list(ir.angles.k)
        theta0 = _as_float_list(ir.angles.theta0)
        params = [
            {
                "type_index": i,
                "k": scale_angle_k(ki),
                "theta0": ti,  # rad unchanged
            }
            for i, (ki, ti) in enumerate(zip(k, theta0, strict=True))
        ]
        return {
            "type": "HarmonicAngleForce",
            "case": row.case.value,
            "parameters": params,
        }

    def _propers(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.propers is not None
        bag = ir.propers
        n_types, n_terms = bag.k.shape
        periodicity = _as_int_list(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"proper torsion idivf[{t}] must be non-zero")
            for m in range(n_terms):
                k_ir = float(bag.k[t, m].item()) / idivf
                params.append(
                    {
                        "type_index": t,
                        "term_index": m,
                        "periodicity": periodicity[m],
                        "phase": float(bag.phase[t, m].item()),
                        "k": scale_torsion_k(k_ir),
                    }
                )
        case = TranslationCase.DECOMPOSE if n_terms > 1 else TranslationCase.FORM_REPARAMETERIZE
        return {
            "type": "PeriodicTorsionForce",
            "case": case.value,
            "parameters": params,
        }

    def _impropers_periodic(self, ir: PotentialIR) -> dict[str, Any]:
        assert ir.impropers_periodic is not None
        bag = ir.impropers_periodic
        n_types, n_terms = bag.k.shape
        periodicity = _as_int_list(bag.periodicity)
        params: list[dict[str, Any]] = []
        for t in range(n_types):
            idivf = float(bag.idivf[t].item())
            if idivf == 0.0:
                raise ValueError(f"improper periodic idivf[{t}] must be non-zero")
            for m in range(n_terms):
                k_ir = float(bag.k[t, m].item()) / idivf
                params.append(
                    {
                        "type_index": t,
                        "term_index": m,
                        "periodicity": periodicity[m],
                        "phase": float(bag.phase[t, m].item()),
                        "k": scale_torsion_k(k_ir),
                    }
                )
        case = TranslationCase.DECOMPOSE if n_terms > 1 else TranslationCase.FORM_REPARAMETERIZE
        return {
            "type": "PeriodicTorsionForce",
            "role": "improper",
            "case": case.value,
            "parameters": params,
            "index_convention": "molrs_center_first",
        }

    def _nonbonded(self, ir: PotentialIR) -> dict[str, Any]:
        row_lj = self.conventions.row("lj")
        row_q = self.conventions.row("charge")
        params: list[dict[str, Any]] = []
        charges: list[float] = []

        if ir.lj is not None:
            eps = _as_float_list(ir.lj.epsilon)
            sig = _as_float_list(ir.lj.sigma)
            for i, (e, s) in enumerate(zip(eps, sig, strict=True)):
                params.append(
                    {
                        "type_index": i,
                        "epsilon": scale_energy(e),
                        "sigma": scale_length(s),
                    }
                )
        if ir.charges is not None:
            charges = _as_float_list(ir.charges.q)

        return {
            "type": "NonbondedForce",
            "case": row_lj.case.value,
            "parameters": params,
            "charges": charges,
            "charge_case": row_q.case.value,
        }

    @staticmethod
    def _scaling_dict(scaling: NonbondedScaling | None) -> dict[str, float] | None:
        if scaling is None:
            return None
        return {
            "scale_q_12": float(scaling.scale_q_12),
            "scale_q_13": float(scaling.scale_q_13),
            "scale_q_14": float(scaling.scale_q_14),
            "scale_lj_12": float(scaling.scale_lj_12),
            "scale_lj_13": float(scaling.scale_lj_13),
            "scale_lj_14": float(scaling.scale_lj_14),
        }
