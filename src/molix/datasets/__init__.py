"""Standard data sources for molecular machine learning.

Most sources own a ``TARGET_SCHEMA`` class attribute (the set of graph-level
and atom-level targets they expose). Downloaders, when applicable, live as
classmethods on the source itself (e.g. :meth:`QM9Source.download`).
:class:`MolRecSource` is the exception — it builds a per-instance
``target_schema`` from the observables discovered in the record (see its
docstring for the rationale).
"""

from molix.datasets.molrec import MolRecSource
from molix.datasets.qm9 import QM9Source
from molix.datasets.revmd17 import RevMD17Source
from molix.datasets.threebpa import ThreeBPASource
from molix.datasets.water_les import WaterLESSource

__all__ = [
    "MolRecSource",
    "QM9Source",
    "RevMD17Source",
    "ThreeBPASource",
    "WaterLESSource",
]
