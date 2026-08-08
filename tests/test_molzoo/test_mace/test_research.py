"""Tests for the research MACE encoder (``molzoo.mace.research``).

Migration note (mace-subpackage-restructure-02-core): the deleted
``tests/test_molzoo/test_mace.py`` contained **no** research-encoder cases —
it only exercised ``EmbeddingBlock`` / ``InteractionBlock`` / ``ProductHead``,
which 01 promoted to ``molrep`` and whose cases now live in the
``tests/test_molrep`` mirrors. The ``molzoo.mace`` re-export surface that file
imported is already guarded by
``tests/test_molrep/test_reexport_compat.py::TestMolzooMaceShim``.

What is left un-tested is the hot-path fix this step owes the research encoder:
``forward`` reads ``self.config.num_interactions`` on every call
(``src/molzoo/mace.py:254``), i.e. a pydantic attribute lookup inside the loop
head — the same dynamo graph-break hazard ``mace_matpes.py:118`` already
avoids with a plain ``self.num_interactions``. Those two cases are below.

The import stays ``from molzoo.mace import MACE``: it resolves through the flat
module today and through the package shim once ``mace.py`` becomes
``mace/research.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo.mace import MACE

NUM_ELEMENTS = 5
NUM_FEATURES = 8
NUM_INTERACTIONS = 2


def _config_reads(cls: type, method: str) -> list[str]:
    """Attribute names read off ``self.config`` inside ``cls.method``."""
    source = Path(inspect.getsourcefile(cls) or "").read_text(encoding="utf-8")
    class_def = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )
    func = next(
        node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == method
    )
    return [
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    ]


@pytest.fixture
def encoder() -> MACE:
    """A tiny research MACE — two layers, eight scalar channels."""
    return MACE(
        node_attr_specs=[
            DiscreteEmbeddingSpec(input_key="Z", num_classes=NUM_ELEMENTS, emb_dim=NUM_FEATURES)
        ],
        num_elements=NUM_ELEMENTS,
        num_features=NUM_FEATURES,
        r_max=5.0,
        num_bessel=4,
        l_max=1,
        num_interactions=NUM_INTERACTIONS,
        correlation=2,
    )


class TestMACE:
    """Test the research MACE encoder."""

    def test_num_interactions_is_a_plain_attribute(self, encoder):
        """The layer count is frozen into a plain ``int`` at construction."""
        assert encoder.num_interactions == NUM_INTERACTIONS
        assert isinstance(encoder.num_interactions, int)

    def test_forward_does_not_read_the_pydantic_config(self):
        """``forward`` must not touch ``self.config`` — it is a graph break."""
        assert _config_reads(MACE, "forward") == []
