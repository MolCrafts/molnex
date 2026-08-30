"""Public-API regression for the LAMMPS meta schema-v3 unit contract.

Spec: ``lammps-iface-01-meta`` (goldens), ``lammps-iface-05-pbc`` (schema bump).

Hard-coded goldens only — no third-party oracle, no subprocess. Pins:

1. ``preset_energy_ratio("real", "metal")`` reproduces the SI-defined value of
   1 kcal/mol in eV, and the reverse direction its reciprocal. The goldens are
   decimal literals hand-computed from 2019-SI exact constants — never derived
   from molpy at capture time (molpy-vs-molpy would pin nothing).
2. ``LAMMPS_META_SCHEMA`` is the literal ``3`` — the value the C++
   ``LammpsContract`` hard-validates; drift on either side must be loud.
   Bumped 2 → 3 by ``lammps-iface-05-pbc`` (``local_mask`` input, per-atom
   energy output, ``receptive_field_layers`` / ``requires_sorted_edges``
   fields). The C++ expected version is bumped inside the same spec but in a
   later batch, so this pin is **transiently red between the Python and C++
   batches by design** — that is the bidirectional tie doing its job, not a
   broken regression.

Provenance
----------
    formula          : 1 kcal/mol = 4184 J/mol / (N_A · e) eV with exact
                       cal_th = 4.184 J, N_A = 6.02214076e23 /mol,
                       e = 1.602176634e-19 C (2019 SI definitions)
    tool             : hand computation (SI exact constants); cross-checked
                       against molpy UnitSystem (pint) at capture time only
    capture command  : PYTHONPATH=src python regressions/lammps-iface-01-meta.py
    date             : 2026-08-30
    tolerance        : rtol 1e-7 on both conversion directions
"""

from __future__ import annotations

import sys

# Hand-computed 2019-SI goldens (see Provenance) — literals, not molpy output.
KCALMOL_IN_EV = 0.043364104
EV_IN_KCALMOL = 23.060548
RTOL = 1e-7


def _close(got: float, want: float, rtol: float) -> bool:
    return abs(got - want) <= rtol * abs(want)


def main() -> int:
    from molix.engine.export import LAMMPS_META_SCHEMA
    from molix.units import preset_energy_ratio

    real_to_metal = preset_energy_ratio("real", "metal")
    metal_to_real = preset_energy_ratio("metal", "real")

    assert _close(real_to_metal, KCALMOL_IN_EV, RTOL), (
        f"real->metal {real_to_metal!r} != golden {KCALMOL_IN_EV!r} (rtol {RTOL})"
    )
    assert _close(metal_to_real, EV_IN_KCALMOL, RTOL), (
        f"metal->real {metal_to_real!r} != golden {EV_IN_KCALMOL!r} (rtol {RTOL})"
    )
    # Auxiliary self-consistency (not a golden): the two directions invert.
    assert abs(real_to_metal * metal_to_real - 1.0) < 1e-12, "round trip broken"

    assert LAMMPS_META_SCHEMA == 3, (
        f"LAMMPS_META_SCHEMA is {LAMMPS_META_SCHEMA!r}, regression pins literal 3; "
        "a deliberate schema bump must update this regression and the C++ "
        "expected version together"
    )

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
