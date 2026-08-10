"""Public-API regression for confidence / coverage / provenance surfaces.

Spec: `learnable-classical-ff-09-provenance`.

Hard-coded goldens only — no third-party oracle. Pins:

1. ChemicalSupportIndex kNN L2 membership + coverage_fraction
2. from_type_ids exact set membership
3. SupportClassifier regime policy (in / near / extrapolating / unknown)
4. TypeHead.decode_with_confidence → classify (no local softmax)
5. ParameterProvenance frozen + as_dict
6. No molpot import under molrep.embedding.support; no AL APIs

Provenance
----------
    capture command  : PYTHONPATH=src python \
                      regressions/learnable-classical-ff-09-provenance.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch


def main() -> int:
    from molpot.heads.provenance import (
        CoverageRegime,
        ParameterProvenance,
        SupportClassifier,
        attach_provenance,
    )
    from molpot.heads.type import TypeHead
    from molrep.embedding.support import ChemicalSupportIndex

    torch.manual_seed(0)

    # --- 1. Continuous kNN L2 bank ---
    bank = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    index = ChemicalSupportIndex(bank, radius=0.5, k=1)
    query = torch.tensor(
        [
            [0.0, 0.0],  # in
            [0.3, 0.0],  # in
            [2.0, 2.0],  # out
        ],
        dtype=torch.float64,
    )
    mask = index.contains(query)
    assert mask.tolist() == [True, True, False], mask
    cov = index.coverage_fraction(query)
    assert math.isclose(cov, 2 / 3, rel_tol=0, abs_tol=1e-6), cov
    dist, nn_idx = index.knn(query[:1])
    assert int(nn_idx[0, 0]) == 0
    assert math.isclose(float(dist[0, 0]), 0.0, abs_tol=1e-6)

    # --- 2. Discrete type-id bank ---
    tindex = ChemicalSupportIndex.from_type_ids({0, 2, 5}, radius=0.0)
    ids = torch.tensor([0, 1, 2, 5, 7], dtype=torch.long)
    assert tindex.contains_type_ids(ids).tolist() == [True, False, True, True, False]
    assert math.isclose(tindex.coverage_fraction_type_ids(ids), 0.6, abs_tol=1e-6)

    # --- 3. Classifier policy ---
    clf = SupportClassifier(tindex, conf_in=0.8, conf_near=0.5)
    regimes = clf.classify(
        torch.tensor([0, 0, 0, 99, 99], dtype=torch.long),
        torch.tensor([0.95, 0.65, 0.20, 0.99, 0.10]),
    )
    assert regimes == [
        CoverageRegime.IN_SUPPORT,
        CoverageRegime.NEAR_SUPPORT,
        CoverageRegime.UNKNOWN,
        CoverageRegime.EXTRAPOLATING,
        CoverageRegime.UNKNOWN,
    ], regimes

    # confidence-only (no support)
    clf_free = SupportClassifier(None, conf_in=0.8, conf_near=0.5)
    assert clf_free.classify(
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0.9]),
    ) == [CoverageRegime.IN_SUPPORT]

    # --- 4. TypeHead.decode_with_confidence reuse ---
    head = TypeHead(hidden_dim=4, num_types=3)
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )
    indices, confidence = head.decode_with_confidence(logits)
    assert int(indices[0]) == 0
    assert float(confidence[0]) > 0.8
    regimes2 = clf.classify(indices, confidence)
    assert regimes2[0] is CoverageRegime.IN_SUPPORT

    # provenance package must not reimplement softmax
    prov_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "molpot"
        / "heads"
        / "provenance"
    )
    for py in prov_root.glob("*.py"):
        text = py.read_text()
        assert "torch.softmax" not in text and "F.softmax" not in text, py

    # --- 5. ParameterProvenance ---
    rec = ParameterProvenance(
        interaction="bond",
        type_id=int(indices[0]),
        confidence=float(confidence[0]),
        regime=regimes2[0],
        source="condensed_type",
        pattern=None,
    )
    d = rec.as_dict()
    assert d["regime"] == "IN_SUPPORT"
    assert d["ir_units"] == "class_i_canonical"
    try:
        rec.type_id = 99  # type: ignore[misc]
        raise AssertionError("ParameterProvenance must be frozen")
    except (FrozenInstanceError, AttributeError):
        pass

    meta: dict = {}
    attach_provenance(meta, [rec])
    assert meta["provenance"][0]["source"] == "condensed_type"

    # --- 6. Import / AL boundaries ---
    support_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "molrep"
        / "embedding"
        / "support.py"
    )
    tree = ast.parse(support_src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("molpot"), alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("molpot"), node.module

    for py in list(prov_root.glob("*.py")) + [support_src]:
        text = py.read_text().lower()
        for bad in ("active_learning", "acquisition_function", "query_selector"):
            assert bad not in text, (py, bad)

    print("learnable-classical-ff-09-provenance: all hard-coded goldens OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
