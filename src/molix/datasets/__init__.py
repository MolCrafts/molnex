"""Standard data sources for molecular machine learning.

Most sources own a ``TARGET_SCHEMA`` class attribute (the set of graph-level
and atom-level targets they expose). Downloaders, when applicable, live as
classmethods on the source itself (e.g. :meth:`QM9Source.download`).
:class:`MolRecSource` is the exception — it builds a per-instance
``target_schema`` from the observables discovered in the record (see its
docstring for the rationale).

``MolRecSource`` depends on the optional ``molpy.MolRec`` public surface
(labeled-configuration records). It is soft-imported so that importing
:mod:`molix.datasets` (and other sources that do not need ``MolRec``) still
succeeds when the installed molpy build does not expose it. Calling the
stub raises a clear :class:`ImportError` instead of ``None(...)``.

**Hard rule:** molnex Python code never imports ``molrs`` — only ``molpy``.
"""

from molix.datasets.qm9 import QM9Source
from molix.datasets.revmd17 import RevMD17Source
from molix.datasets.threebpa import ThreeBPASource
from molix.datasets.water_les import WaterLESSource

try:
    from molix.datasets.molrec import MolRecSource
except ImportError:  # molpy.MolRec (or molrec module deps) not available

    class MolRecSource:
        """Placeholder when ``molpy.MolRec`` is not on the public API."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MolRecSource requires molpy.MolRec (labeled-configuration "
                "records). Install a molpy build that exposes MolRec; "
                "do not import molrs from molnex."
            )


__all__ = [
    "MolRecSource",
    "QM9Source",
    "RevMD17Source",
    "ThreeBPASource",
    "WaterLESSource",
]
