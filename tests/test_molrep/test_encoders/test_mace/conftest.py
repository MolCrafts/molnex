"""The MACE encoder tests build their bodies from the shipped specs, as the product tests do.

A MACE body's irreps schedule is stated by each shipped spec
(:mod:`molzoo.mace.spec`), so the encoder is exercised through the same tiny
specs and clusters the product tests use.
"""

from tests.test_molzoo.test_mace.conftest import (
    DTYPE,
    cluster,
    omol_cluster,
    periodic_cluster,
    tiny_matpes_spec,
    tiny_off_spec,
    tiny_omol_spec,
)

__all__ = [
    "DTYPE",
    "cluster",
    "omol_cluster",
    "periodic_cluster",
    "tiny_matpes_spec",
    "tiny_off_spec",
    "tiny_omol_spec",
]
