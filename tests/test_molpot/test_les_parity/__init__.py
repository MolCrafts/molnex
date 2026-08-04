"""Numerical-parity tests for `EwaldMultipoleEnergy` vs the brute-force oracle.

Every test in this directory loads `tests/test_molpot/test_les_parity/conftest.py`
as the reference implementation (pure NumPy, dependency-free) and asserts
≤1e-6 (float64) agreement against the production-side
`molpot.potentials.EwaldMultipoleEnergy`. No upstream `les` import.
"""
