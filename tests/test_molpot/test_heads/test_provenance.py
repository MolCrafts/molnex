"""Tests for molpot.heads.provenance (coverage + ParameterProvenance).

Spec: learnable-classical-ff-09-provenance.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from molpot.heads.provenance import (
    CoverageRegime,
    ParameterProvenance,
    SupportClassifier,
    attach_provenance,
)
from molpot.heads.type import TypeHead
from molrep.condensation.classes import InteractionClass
from molrep.embedding.support import ChemicalSupportIndex


class TestCoverageRegime:
    def test_members(self):
        names = {m.name for m in CoverageRegime}
        assert names == {
            "IN_SUPPORT",
            "NEAR_SUPPORT",
            "EXTRAPOLATING",
            "UNKNOWN",
        }


class TestSupportClassifier:
    """Confidence bands + support membership → CoverageRegime."""

    def test_in_support_high_confidence(self):
        support = ChemicalSupportIndex.from_type_ids({0, 1, 2}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([0, 1], dtype=torch.long)
        conf = torch.tensor([0.95, 0.90])
        regimes = clf.classify(type_ids, conf)
        assert regimes == [
            CoverageRegime.IN_SUPPORT,
            CoverageRegime.IN_SUPPORT,
        ]

    def test_near_support_mid_confidence(self):
        support = ChemicalSupportIndex.from_type_ids({0, 1}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([0], dtype=torch.long)
        conf = torch.tensor([0.65])
        assert clf.classify(type_ids, conf) == [CoverageRegime.NEAR_SUPPORT]

    def test_low_confidence_is_unknown(self):
        """Documented policy: conf < conf_near → UNKNOWN (usable floor)."""
        support = ChemicalSupportIndex.from_type_ids({0}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([0], dtype=torch.long)
        conf = torch.tensor([0.2])
        assert clf.classify(type_ids, conf) == [CoverageRegime.UNKNOWN]

    def test_out_of_support_never_in_support(self):
        support = ChemicalSupportIndex.from_type_ids({0}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([99, 99], dtype=torch.long)
        conf = torch.tensor([0.99, 0.60])
        regimes = clf.classify(type_ids, conf)
        assert CoverageRegime.IN_SUPPORT not in regimes
        assert regimes[0] == CoverageRegime.EXTRAPOLATING
        assert regimes[1] == CoverageRegime.EXTRAPOLATING

    def test_out_of_support_low_conf_unknown(self):
        support = ChemicalSupportIndex.from_type_ids({0}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([99], dtype=torch.long)
        conf = torch.tensor([0.1])
        assert clf.classify(type_ids, conf) == [CoverageRegime.UNKNOWN]

    def test_missing_support_confidence_only(self):
        """No support index: confidence bands alone (membership assumed True)."""
        clf = SupportClassifier(None, conf_in=0.8, conf_near=0.5)
        type_ids = torch.tensor([3, 4, 5], dtype=torch.long)
        conf = torch.tensor([0.9, 0.6, 0.1])
        assert clf.classify(type_ids, conf) == [
            CoverageRegime.IN_SUPPORT,
            CoverageRegime.NEAR_SUPPORT,
            CoverageRegime.UNKNOWN,
        ]

    def test_embedding_query_path(self):
        """Continuous embedding bank: classify via L2 membership."""
        bank = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        support = ChemicalSupportIndex(bank, radius=0.25, k=1)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        # type_ids unused for membership when query= is given
        type_ids = torch.tensor([0, 0], dtype=torch.long)
        conf = torch.tensor([0.95, 0.95])
        query = torch.tensor([[0.0, 0.0], [5.0, 5.0]], dtype=torch.float64)
        regimes = clf.classify(type_ids, conf, query=query)
        assert regimes == [
            CoverageRegime.IN_SUPPORT,
            CoverageRegime.EXTRAPOLATING,
        ]

    def test_reuses_typehead_decode_with_confidence(self):
        """(indices, confidence) from TypeHead — no local softmax helper."""
        head = TypeHead(hidden_dim=4, num_types=3)
        # Hand-crafted logits so softmax max is deterministic.
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],  # high conf → class 0
                [0.0, 1.0, 0.5],  # mid conf → class 1
                [0.1, 0.0, 0.0],  # low conf → class 0
            ],
            dtype=torch.float64,
        )
        # Bypass nn weights: call decode on synthetic logits directly.
        indices, confidence = head.decode_with_confidence(logits)

        support = ChemicalSupportIndex.from_type_ids({0, 1}, radius=0.0)
        clf = SupportClassifier(support, conf_in=0.8, conf_near=0.5)
        regimes = clf.classify(indices, confidence)

        assert int(indices[0]) == 0
        assert float(confidence[0]) > 0.8
        assert regimes[0] == CoverageRegime.IN_SUPPORT
        # mid-band softmax max on [0,1,0.5] ≈ 0.51 → NEAR_SUPPORT
        assert regimes[1] == CoverageRegime.NEAR_SUPPORT
        # low-band on [0.1,0,0] ≈ 0.36 → UNKNOWN
        assert regimes[2] == CoverageRegime.UNKNOWN

        # Prove provenance package does not reimplement softmax-max.
        prov_root = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "molpot"
            / "heads"
            / "provenance"
        )
        for py in prov_root.glob("*.py"):
            src = py.read_text()
            # No parallel softmax implementation (docs may mention the word).
            assert "torch.softmax" not in src
            assert "F.softmax" not in src
            assert "nn.functional.softmax" not in src


class TestParameterProvenance:
    def test_construction_and_fields(self):
        rec = ParameterProvenance(
            interaction="bond",
            type_id=3,
            confidence=0.91,
            regime=CoverageRegime.IN_SUPPORT,
            source="condensed_type",
            pattern="[#6]-[#8]",
        )
        assert rec.interaction == "bond"
        assert rec.type_id == 3
        assert rec.confidence == 0.91
        assert rec.regime is CoverageRegime.IN_SUPPORT
        assert rec.source == "condensed_type"
        assert rec.pattern == "[#6]-[#8]"
        assert rec.ir_units == "class_i_canonical"
        assert rec.notes == ""

    def test_frozen(self):
        rec = ParameterProvenance(
            interaction="angle",
            type_id=None,
            confidence=None,
            regime=CoverageRegime.UNKNOWN,
            source="neural_continuous",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            rec.type_id = 1  # type: ignore[misc]

    def test_as_dict_json_friendly(self):
        rec = ParameterProvenance(
            interaction="lj",
            type_id=1,
            confidence=0.5,
            regime=CoverageRegime.NEAR_SUPPORT,
            source="symbolic",
            pattern=None,
            notes="unit",
        )
        d = rec.as_dict()
        assert d["interaction"] == "lj"
        assert d["regime"] == "NEAR_SUPPORT"
        assert d["source"] == "symbolic"
        assert d["type_id"] == 1
        assert d["confidence"] == 0.5
        assert d["ir_units"] == "class_i_canonical"

    def test_interaction_class_serializes(self):
        rec = ParameterProvenance(
            interaction=InteractionClass.BOND,
            type_id=0,
            confidence=0.9,
            regime=CoverageRegime.IN_SUPPORT,
            source="condensed_type",
        )
        assert rec.as_dict()["interaction"] == "bond"

    def test_attach_provenance_on_metadata_object(self):
        class _Spec:
            def __init__(self) -> None:
                self.metadata: dict = {}

        rec = ParameterProvenance(
            interaction="bond",
            type_id=1,
            confidence=0.9,
            regime=CoverageRegime.IN_SUPPORT,
            source="neural_continuous",
        )
        spec = _Spec()
        out = attach_provenance(spec, [rec])
        assert out is spec
        assert spec.metadata["provenance"][0]["type_id"] == 1
        assert spec.metadata["provenance"][0]["regime"] == "IN_SUPPORT"

    def test_attach_provenance_on_dict(self):
        rec = ParameterProvenance(
            interaction="angle",
            type_id=None,
            confidence=None,
            regime=CoverageRegime.UNKNOWN,
            source="symbolic",
        )
        meta: dict = {}
        attach_provenance(meta, [rec])
        assert meta["provenance"][0]["interaction"] == "angle"


class TestNoActiveLearningApis:
    def test_public_modules_have_no_acquisition(self):
        prov_root = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "molpot"
            / "heads"
            / "provenance"
        )
        forbidden = (
            "acquisition",
            "active_learning",
            "query_selector",
            "retrain_loop",
            "select_batch",
        )
        for py in prov_root.glob("*.py"):
            tree = ast.parse(py.read_text())
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name.lower())
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id.lower())
            for bad in forbidden:
                assert not any(bad in n for n in names), (py.name, names)
