"""ByteFF — the covalent-graph encoder with the published bonded and pair heads.

* :mod:`molzoo.byteff.potential` — :class:`ByteFF` (a :class:`~molzoo.rondo.model.Rondo`
  with bond-charge-correction charges and directly predicted σ, ε; ``.load``)
* :mod:`molzoo.byteff.checkpoint` — the ByteFF-26 key table

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
"""

from .potential import ByteFF

__all__ = ["ByteFF"]
