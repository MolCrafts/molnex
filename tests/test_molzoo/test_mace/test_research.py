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

Migration note (mace-subpackage-restructure-06-wire): the one case of the
deleted ``tests/test_molzoo/test_mace_encoder.py`` — the encoder-output
contract — lands here and not in ``test_encoder.py``. That file mirrors
``molzoo/mace/encoder.py`` (:class:`~molzoo.mace.encoder.MACEEncoder`, the
foundation backbone); the symbol the case pins is
:class:`molzoo.mace.research.MACE`, whose mirror is this module, and whose
``TestMACE`` class already exists here. Its ``torch.randn`` geometry and
``torch.randint`` species were replaced by the package's fixed cluster, so the
case no longer depends on the ambient RNG.

The import stays ``from molzoo.mace import MACE``, the stable package surface.

fp64 support is pinned by :meth:`TestMACE.test_builds_and_runs_at_fp64`.
Until that case is green the research encoder **cannot be built at fp64**:
under ``config.set_precision("fp64")`` it comes out mixed-precision (17 fp64 /
14 fp32 parameters) and the first forward dies with ``mat1 and mat2 must have
the same dtype``. Two ``molrep`` layer families ignore the ``config["ftype"]``
singleton that ``molrep.embedding.mlp`` (``:57``) honours:

* ``molrep.embedding.node`` — ``nn.Embedding`` / ``nn.Linear`` / ``cuet.Linear``
  built without ``dtype=`` (``src/molrep/embedding/node.py:143,147,156,264,
  269-276``), which is where the two ``embedding.node_embedding.*`` fp32
  parameters come from;
* ``molrep.interaction.radial.RadialWeightMLP`` — ``nn.Linear`` without
  ``dtype=`` (``src/molrep/interaction/radial.py:79,82``), which is where the
  twelve ``interactions.*.radial_mlp.*`` fp32 parameters come from.

The remaining fp32-only cases below (:func:`fp32`) are **not** a weakening —
they are the coverage the deleted flat file had, kept green across the fix so
the default precision cannot regress while fp64 is being enabled. Once fp64 is
green, whoever lands the fix should re-read this note: the site lists above are
the only part of it that goes stale.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix import config
from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo.mace import MACE
from tests.conftest import make_graph_batch
from tests.test_molzoo.test_mace.conftest import CLUSTER_POS, SINGLE_GRAPH, full_edge_index

NUM_ELEMENTS = 5
NUM_FEATURES = 8
NUM_INTERACTIONS = 2

#: Species labels of :data:`~tests.test_molzoo.test_mace.conftest.CLUSTER_POS`
#: for the research encoder. Unlike the foundation variants it has no z-table:
#: ``Z`` is a plain 0-based class index into a ``num_classes`` embedding, so the
#: labels must stay below :data:`NUM_ELEMENTS`.
CLUSTER_SPECIES = [0, 1, 1, 2, 1]


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
def fp32(fp64) -> None:
    """Override the package's autouse fp64 for the research encoder.

    Depends on ``fp64`` so it is torn down first and hands the precision back
    in the state that fixture expects. See the module docstring for why the
    research encoder cannot be built at fp64.
    """
    config.set_precision("fp32")
    yield
    config.set_precision("fp64")


def tiny_mace() -> MACE:
    """Build a tiny research MACE at the ambient precision.

    Two layers, eight scalar channels. Called by the fp32 :func:`encoder`
    fixture and by the fp64 case, so both exercise the same configuration and
    only the ambient ``config["ftype"]`` differs.
    """
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


def tiny_cluster() -> TensorDict:
    """The package's five-atom cluster at the ambient precision."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=config.ftype),
        Z=torch.tensor(CLUSTER_SPECIES, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_SPECIES), dtype=torch.long),
    )


@pytest.fixture
def encoder(fp32) -> MACE:
    """A tiny research MACE — two layers, eight scalar channels, fp32."""
    return tiny_mace()


@pytest.fixture
def species_cluster(fp32) -> TensorDict:
    """The package's five-atom cluster relabelled with research species indices."""
    return tiny_cluster()


class TestMACE:
    """Test the research MACE encoder."""

    def test_num_interactions_is_a_plain_attribute(self, encoder):
        """The layer count is frozen into a plain ``int`` at construction."""
        assert encoder.num_interactions == NUM_INTERACTIONS
        assert isinstance(encoder.num_interactions, int)

    def test_forward_does_not_read_the_pydantic_config(self):
        """``forward`` must not touch ``self.config`` — it is a graph break."""
        assert _config_reads(MACE, "forward") == []

    # -- migrated from tests/test_molzoo/test_mace_encoder.py::TestMACE by
    #    mace-subpackage-restructure-06-wire (assertion verbatim) ------------

    def test_forward_writes_per_layer_node_features(self, encoder, species_cluster):
        """The encoder contract: ``atoms.node_features`` is ``(N, layers, features)``."""
        node_features = encoder(species_cluster)["atoms", "node_features"]
        assert isinstance(node_features, torch.Tensor)
        assert node_features.shape == (len(CLUSTER_SPECIES), NUM_INTERACTIONS, NUM_FEATURES)

    def test_builds_and_runs_at_fp64(self):
        """The encoder is buildable and runnable at the package's ambient fp64.

        Deliberately does **not** request the :func:`fp32` override: it runs
        under the package-wide autouse ``fp64`` fixture, which is the precision
        the foundation-weight path uses. Every parameter must come out fp64 —
        a single fp32 layer both loses the precision the caller asked for and
        breaks the forward on a dtype-mismatched matmul.
        """
        encoder = tiny_mace()

        assert {p.dtype for p in encoder.parameters()} == {torch.float64}

        node_features = encoder(tiny_cluster())["atoms", "node_features"]
        assert node_features.dtype == torch.float64
        assert node_features.shape == (len(CLUSTER_SPECIES), NUM_INTERACTIONS, NUM_FEATURES)
