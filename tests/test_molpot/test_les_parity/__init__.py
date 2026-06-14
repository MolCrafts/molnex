"""Numerical-parity tests for `EwaldMultipoleEnergy` vs the brute-force oracle.

Every test in this directory loads `tests/_oracles/screened_coulomb.py`
as the reference implementation (pure NumPy, dependency-free) and asserts
≤1e-6 (float64) agreement against the production-side
`molpot.potentials.EwaldMultipoleEnergy`. No upstream `les` import.
"""
