"""Tests for the ``molix.nn`` public import surface (``src/molix/nn/__init__.py``).

That file is a pure re-export module, so the only contract it owns is the *set*
of names it publishes and the fact that each one resolves. Two rules are pinned
here:

* ``NeighborList`` is **gone**. ``molix.nn.locality.NeighborList`` was an
  ``nn.Module`` wrapper over :func:`molix.F.locality.get_neighbor_pairs` with
  zero in-tree call sites, and it collided by name with two live classes
  (:class:`molix.data.tasks.neighbor.NeighborList`, which builds the pipeline's
  edge tensors, and :class:`molix.md.PeriodicNeighborList`). Spec
  ``md-neighborlist-skin-02-prune`` deletes it; the name must stay free.
* ``__all__`` stays alphabetized (``.claude/notes/notes.md:243``). The deletion
  rewrites the literal anyway, so the sorted order is pinned in the same value.

Following the one-assertion-family-per-method pattern of
``tests/test_molzoo/test_imports.py``. No physics here — this module computes
nothing, so there is no domain tier.
"""

from __future__ import annotations

import importlib.util
import inspect

import molix.nn

#: The full public surface, hard-coded rather than derived. Written out as a
#: literal so it pins **two** things at once: that ``NeighborList`` is no longer
#: exported, and that the remaining four names are in alphabetical order. A
#: ``sorted(...)``-based assertion would only catch the second.
EXPECTED_ALL = ["BatchAggregation", "KeyedMLP", "KeyedMLPSpec", "ScatterSum"]


class TestNnExports:
    """The names ``molix.nn`` publishes, and the ones it must not."""

    def test___all___is_the_pinned_literal(self) -> None:
        """``__all__`` is exactly the four sorted names — value, not just set."""
        assert molix.nn.__all__ == EXPECTED_ALL

    def test_every_export_resolves(self) -> None:
        """Each ``__all__`` entry is actually bound on the module.

        Guards the failure mode a re-sorted list invites: an entry kept in
        ``__all__`` whose ``from .x import y`` line was dropped, which
        ``from molix.nn import *`` would only report at the caller's site.
        """
        unresolved: list[str] = [name for name in molix.nn.__all__ if not hasattr(molix.nn, name)]
        assert unresolved == []

    def test_no_stray_public_attribute(self) -> None:
        """The public non-module attributes are exactly ``set(__all__)``.

        Sub-module names (``mlp``, ``scatter``) are bound on the package as a
        side effect of ``from .x import y`` and are filtered with
        :func:`inspect.ismodule`; ``molix.nn`` defines no ``__dir__``, so a bare
        ``set(dir(molix.nn)) == set(molix.nn.__all__)`` would fail on those and
        be a false red. What is left after the filter is the surface a caller
        can bind, and it must not exceed what is declared.
        """
        public_attributes = {
            name
            for name in vars(molix.nn)
            if not name.startswith("_") and not inspect.ismodule(getattr(molix.nn, name))
        }
        assert public_attributes == set(molix.nn.__all__)

    def test_neighborlist_not_reintroduced(self) -> None:
        """Neither ``__all__`` nor the module namespace carries ``NeighborList``.

        Two distinct regressions: re-adding the export trips the first clause,
        while a bare ``from .locality import NeighborList`` with no ``__all__``
        edit trips only the second.
        """
        assert "NeighborList" not in molix.nn.__all__
        assert not hasattr(molix.nn, "NeighborList")

    def test_locality_module_gone(self) -> None:
        """``molix.nn.locality`` is not importable — the file itself is gone.

        The name check above passes for a module that still ships but is merely
        unexported; this asserts the deletion, not just the hidden re-export.
        """
        assert importlib.util.find_spec("molix.nn.locality") is None
