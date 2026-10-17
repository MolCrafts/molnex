"""MolZoo: molecular model zoo — encoder architectures and foundation models.

An *encoder* here maps a batch of atoms (their chemical element numbers and
Cartesian positions) to a feature vector per atom, which a downstream head
turns into an energy or another property; a *foundation* model is one shipped
with weights already fitted on a large, chemically broad dataset, used
unchanged rather than trained by the caller.

Nothing is exported at this level. Each family is a subpackage, imported
eagerly and by its own name::

    from molzoo.mace import MaceMatpes, MACEMatpesSpec
    from molzoo.allegro import AllegroPotential, AllegroSpec
    from molzoo.pinet import PiNetPotential, PiNetSpec
    from molzoo.rondo import Rondo, RondoSpec
    from molzoo.chem import ChemPerception, ChemPerceptionSpec

* :mod:`molzoo.mace` — MACE (MatPES, OMol, OFF, LES, polar products)
* :mod:`molzoo.allegro` — Allegro (``.nequip.zip`` official packages)
* :mod:`molzoo.pinet` — PiNet2 (energy, dipole, polarizability products)
* :mod:`molzoo.rondo` — Rondo, the learnable Class-I force field
* :mod:`molzoo.chem` — continuous chemical perception
"""
