"""Tests for molzoo.mace.encoder — the spec-driven MACE backbone.

The chain gate is ``state_dict`` parity: :class:`~molzoo.mace.encoder.MACEEncoder`
must register exactly the modules the keyword-constructed
:class:`molzoo.mace.variants.MACEMatpes` / :class:`~molzoo.mace.variants.MACEOMol`
register, under exactly the same names, so an official checkpoint keeps loading
without a key rewrite (05-checkpoint) and the variant classes can simply inherit
the backbone.

Everything else here pins one primitive at a time: the encoder exposes
``validate_elements`` / ``node_attrs`` / ``initial_node_features`` /
``angular_features`` / ``radial_features`` / ``conditioning`` /
``layer_features`` and the caller composes them.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from molzoo.mace.encoder import MACEEncoder
from tests.conftest import rotate_graph, translate_graph
from tests.test_molzoo.test_mace.conftest import (
    CLUSTER_Z,
    TINY_MATPES_KWARGS,
    TINY_OMOL_KWARGS,
)

#: Rigid translation applied by the invariance test (Å).
TRANSLATION = [0.31, -0.72, 1.13]

#: Rotation angles (rad) of the ``Rz(γ) @ Rx(β)`` test rotation.
ROTATION_Z = 0.7
ROTATION_X = 0.4

#: Scalar features are dimensionless activations; fp64 invariance holds to ~1e-13.
INVARIANCE_ATOL = 1e-10

#: ``Z = [8, 1, 6, 1]`` against the ascending table ``[1, 6, 8]``.
ONE_HOT_Z = [8, 1, 6, 1]
EXPECTED_ONE_HOT = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
]


def _rotation_matrix() -> torch.Tensor:
    """``Rz(0.7) @ Rx(0.4)`` in fp64 — a fixed, general SO(3) element."""
    cz, sz = math.cos(ROTATION_Z), math.sin(ROTATION_Z)
    cx, sx = math.cos(ROTATION_X), math.sin(ROTATION_X)
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=torch.float64)
    return rz @ rx


def _layer_inputs(encoder: MACEEncoder, batch: TensorDict) -> dict[str, Any]:
    """Compose the encoder primitives into ``layer_features`` keyword arguments."""
    Z = batch["atoms", "Z"]
    edge_index = batch["edges", "edge_index"]
    vectors = batch["edges", "edge_diff"]
    lengths = batch["edges", "edge_dist"]

    node_attrs = encoder.node_attrs(Z, vectors.dtype)
    edge_feats, cutoff = encoder.radial_features(lengths, Z, edge_index)
    return {
        "node_feats": encoder.initial_node_features(node_attrs),
        "node_attrs": node_attrs,
        "edge_attrs": encoder.angular_features(vectors),
        "edge_feats": edge_feats,
        "edge_index": edge_index,
        "cutoff": cutoff,
    }


def _spec_reading_methods(cls: type) -> list[str]:
    """Names of ``cls`` methods (other than ``__init__``) that read ``self._spec``."""
    source = Path(inspect.getsourcefile(cls) or "").read_text(encoding="utf-8")
    class_def = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )
    offenders: list[str] = []
    for node in class_def.body:
        if not isinstance(node, ast.FunctionDef) or node.name == "__init__":
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "_spec"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                offenders.append(node.name)
                break
    return offenders


@pytest.fixture
def matpes_encoder(tiny_matpes_spec) -> MACEEncoder:
    """``MACEEncoder`` on the tiny MatPES spec."""
    return MACEEncoder(tiny_matpes_spec).eval()


@pytest.fixture
def omol_encoder(tiny_omol_spec) -> MACEEncoder:
    """``MACEEncoder`` on the tiny OMOL spec."""
    return MACEEncoder(tiny_omol_spec).eval()


class TestMACEEncoder:
    """Test the configuration-driven MACE backbone."""

    # -- state_dict parity: the chain gate -----------------------------------

    def test_state_dict_keys_match_the_matpes_variant(self, matpes_encoder, matpes_variant):
        """Same module names as ``MACEMatpes`` — checkpoints load without a rewrite."""
        assert set(matpes_encoder.state_dict()) == set(matpes_variant.state_dict())

    def test_state_dict_shapes_and_dtypes_match_the_matpes_variant(
        self, matpes_encoder, matpes_variant
    ):
        """Every shared MatPES entry agrees on ``shape`` and ``dtype``."""
        oracle = matpes_variant.state_dict()
        own = matpes_encoder.state_dict()
        mine = {k: (tuple(v.shape), v.dtype) for k, v in own.items() if k in oracle}
        theirs = {k: (tuple(v.shape), v.dtype) for k, v in oracle.items() if k in own}
        assert mine == theirs

    def test_state_dict_keys_match_the_omol_variant(self, omol_encoder, omol_variant):
        """Same module names as ``MACEOMol`` — the second variant of one backbone."""
        assert set(omol_encoder.state_dict()) == set(omol_variant.state_dict())

    def test_state_dict_shapes_and_dtypes_match_the_omol_variant(self, omol_encoder, omol_variant):
        """Every shared OMOL entry agrees on ``shape`` and ``dtype``."""
        oracle = omol_variant.state_dict()
        own = omol_encoder.state_dict()
        mine = {k: (tuple(v.shape), v.dtype) for k, v in own.items() if k in oracle}
        theirs = {k: (tuple(v.shape), v.dtype) for k, v in oracle.items() if k in own}
        assert mine == theirs

    # -- element table primitives --------------------------------------------

    def test_node_attrs_is_the_hard_coded_one_hot(self, matpes_encoder):
        """``Z`` is looked up in the ascending table and one-hot encoded."""
        attrs = matpes_encoder.node_attrs(torch.tensor(ONE_HOT_Z, dtype=torch.long), torch.float64)
        expected = torch.tensor(EXPECTED_ONE_HOT, dtype=torch.float64)
        assert torch.equal(attrs, expected)

    def test_node_attrs_honours_the_requested_dtype(self, matpes_encoder):
        """The dtype argument is the caller's, not the module's."""
        attrs = matpes_encoder.node_attrs(torch.tensor(ONE_HOT_Z, dtype=torch.long), torch.float32)
        assert attrs.dtype == torch.float32

    def test_validate_elements_accepts_the_table(self, matpes_encoder):
        """Atomic numbers inside the table pass silently."""
        assert matpes_encoder.validate_elements(torch.tensor([1, 6, 8, 1])) is None

    def test_validate_elements_lists_the_out_of_table_numbers(self, matpes_encoder):
        """An unknown element must name itself — ``searchsorted`` would snap it."""
        with pytest.raises(ValueError, match=r"\[2, 7\]"):
            matpes_encoder.validate_elements(torch.tensor([1, 7, 2, 6]))

    # -- conditioning ---------------------------------------------------------

    def test_conditioning_rejects_an_unconditioned_variant(self, matpes_encoder, cluster):
        """MatPES has no charge/spin embedding — asking for one is an error."""
        with pytest.raises(ValueError):
            matpes_encoder.conditioning(
                cluster["atoms", "batch"],
                total_spin=torch.tensor([1], dtype=torch.long),
                total_charge=torch.tensor([0], dtype=torch.long),
            )

    def test_conditioning_returns_one_row_per_atom(self, omol_encoder, cluster):
        """The OMOL joint embedding broadcasts per-graph scalars onto atoms."""
        conditioned = omol_encoder.conditioning(
            cluster["atoms", "batch"],
            total_spin=torch.tensor([1], dtype=torch.long),
            total_charge=torch.tensor([0], dtype=torch.long),
        )
        assert conditioned.shape == (len(CLUSTER_Z), TINY_OMOL_KWARGS["num_features"])

    # -- layer_features -------------------------------------------------------

    def test_layer_features_returns_one_tensor_per_interaction(self, matpes_encoder, cluster):
        """A list (not a stacked tensor): MatPES layer widths differ."""
        feats = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, cluster))
        assert isinstance(feats, list)
        assert len(feats) == TINY_MATPES_KWARGS["num_interactions"]

    def test_layer_features_rows_match_the_atom_count(self, matpes_encoder, cluster):
        """Every layer carries one row per atom, at the configured precision."""
        feats = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, cluster))
        assert all(f.shape[0] == len(CLUSTER_Z) for f in feats)
        assert all(f.dtype == torch.float64 for f in feats)

    def test_last_layer_features_are_translation_invariant(self, matpes_encoder, cluster):
        """The last layer is pure-scalar irreps — a rigid shift cannot move it."""
        moved = translate_graph(cluster, torch.tensor(TRANSLATION, dtype=torch.float64))
        before = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, cluster))[-1]
        after = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, moved))[-1]
        assert torch.allclose(before, after, atol=INVARIANCE_ATOL, rtol=0.0)

    def test_last_layer_features_are_rotation_invariant_with_shifts(
        self, matpes_encoder, periodic_cluster
    ):
        """``S = n · h`` rotates with the box, so scalar features stay put."""
        turned = rotate_graph(periodic_cluster, _rotation_matrix())
        before = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, periodic_cluster))[
            -1
        ]
        after = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, turned))[-1]
        assert torch.allclose(before, after, atol=INVARIANCE_ATOL, rtol=0.0)

    # -- hot-path discipline ---------------------------------------------------

    def test_primitives_run_without_the_spec_attribute(self, matpes_encoder, cluster):
        """``_spec`` is provenance only; deleting it must not break the model."""
        del matpes_encoder._spec
        feats = matpes_encoder.layer_features(**_layer_inputs(matpes_encoder, cluster))
        assert len(feats) == TINY_MATPES_KWARGS["num_interactions"]

    def test_primitive_methods_never_read_the_spec(self):
        """Only ``__init__`` may touch ``self._spec`` — pydantic attribute reads
        on a hot path are a dynamo graph break (cf. mace_matpes.py:118)."""
        assert _spec_reading_methods(MACEEncoder) == []
