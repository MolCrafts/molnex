"""molrep.encoders — encoders: a batch in, per-atom (and per-bond) features out.

Every encoder lives here, one subpackage per architecture, with the blocks it
is built from. A network in :mod:`molzoo` composes one of them with heads
(:mod:`molpot`) into a named model. The force-field heads read
:class:`EncoderFeatures`; :class:`ElementTable` is the element-table
precondition every encoder shares.
"""

from molrep.encoders.element_table import ElementTable
from molrep.encoders.features import EncoderFeatures

__all__ = ["ElementTable", "EncoderFeatures"]
