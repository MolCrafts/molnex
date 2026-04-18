"""Standard data sources for molecular machine learning.

Each source owns a ``TARGET_SCHEMA`` class attribute (the set of graph-level
and atom-level targets it exposes). Downloaders, when applicable, live as
classmethods on the source itself (e.g. :meth:`QM9Source.download`).
"""

from molix.datasets.qm9 import QM9Source
from molix.datasets.revmd17 import RevMD17Source
from molix.datasets.threebpa import ThreeBPASource

__all__ = [
    "QM9Source",
    "RevMD17Source",
    "ThreeBPASource",
]
