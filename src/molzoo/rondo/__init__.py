"""Rondo — MolNex's learnable Class-I force field.

A covalent-graph encoder (:class:`~molrep.encoders.covalent_graph.encoder.CovalentGraphEncoder`)
reads a molecule's topology and returns one feature row per atom and per bond
(:class:`~molrep.encoders.features.EncoderFeatures`). One parameter head per style
the force field prices turns those rows into the style's parameters, and the
same module prices them through the molpot potentials. :class:`Rondo` prices
charges by electronegativity equalisation (D1) and Lennard-Jones around GAFF2
(D2); :class:`molzoo.byteff.ByteFF` uses bond charge corrections and directly
predicted σ, ε instead.

Specs: ``src/molzoo/specs/rondo.md``; ``.claude/specs/rondo-enc-01-graph-encoder.md``.

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817

Public surface::

    from molzoo.rondo import Rondo

    model = Rondo(ff, dtype=torch.float64)
"""

from .model import Rondo

__all__ = ["Rondo"]
