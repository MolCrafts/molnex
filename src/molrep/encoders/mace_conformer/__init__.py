"""MACE conformer encoder: a MACE body on one reference conformer per molecule.

Read out onto the features the force-field heads take.
"""

from molrep.encoders.mace_conformer.encoder import MaceConformerEncoder
from molrep.encoders.mace_conformer.spec import MaceConformerEncoderSpec

__all__ = ["MaceConformerEncoder", "MaceConformerEncoderSpec"]
