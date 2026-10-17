"""Rondo — MolNex's learnable Class-I force field.

A graph neural network reads a molecule's covalent topology and emits every
open parameter of the Class-I force field it is built on, and the same module
prices that field through the molpot potentials of the force field's styles.
Lineage: Espaloma (arXiv:2010.01196); structure aligned with ByteFF
(arXiv:2408.12817): joint multi-objective training, GAFF2 non-bonded start,
ByteFF's input features. The departures that remain — charges from
electronegativity equalisation, learned Lennard-Jones around GAFF2 — are argued
in ``src/molzoo/specs/rondo.md`` §1.3.

Public surface::

    from molzoo.rondo import Rondo, RondoSpec, EncoderSpec, HeadsSpec
"""

from .model import Rondo
from .spec import EncoderSpec, HeadsSpec, RondoSpec

__all__ = ["EncoderSpec", "HeadsSpec", "Rondo", "RondoSpec"]
