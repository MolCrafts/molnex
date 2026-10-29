"""MolZoo: named networks.

Each subpackage is one network: a composition of encoders
(:mod:`molrep.encoders`) and heads (:mod:`molpot`) into a model with its own
spec and, where official weights exist, their loader. Nothing is exported at
this level; import a network from its subpackage, eagerly::

    from molzoo.mace import MaceMatpes, MACEMatpesSpec
    from molzoo.allegro import AllegroPotential, AllegroSpec
    from molzoo.pinet import PiNetPotential, PiNetSpec
    from molzoo.rondo import Rondo
    from molzoo.byteff import ByteFF

* :mod:`molzoo.mace` — MACE (MatPES, OMol, OFF, LES, polar products)
* :mod:`molzoo.allegro` — Allegro (``.nequip.zip`` official packages)
* :mod:`molzoo.pinet` — PiNet2 (energy, dipole, polarizability products)
* :mod:`molzoo.rondo` — Rondo, the learnable Class-I force field
* :mod:`molzoo.byteff` — ByteFF (ByteFF-26 official weights)
"""
