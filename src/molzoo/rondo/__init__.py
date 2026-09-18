"""Rondo — MolNex's learnable Class-I force field.

A graph neural network reads a molecule's covalent topology and emits every
parameter of a general Class-I functional form, which ``molpot`` evaluates as
an ordinary classical potential. Lineage: Espaloma (arXiv:2010.01196);
contrast: ByteFF (arXiv:2408.12817). The three deliberate departures — charges
from electronegativity equalisation, learned Lennard-Jones, and Cartesian
torsion amplitudes — are argued in ``src/molzoo/specs/rondo.md`` §1.3.

Public surface::

    from molzoo.rondo import Rondo, RondoForceField, RondoPotential, RondoSpec
"""

from .forcefield import RondoForceField
from .parameterizer import Rondo
from .potential import RondoPotential
from .spec import RondoSpec

__all__ = [
    "Rondo",
    "RondoForceField",
    "RondoPotential",
    "RondoSpec",
]
