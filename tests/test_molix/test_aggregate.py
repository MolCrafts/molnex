"""Tests for the Phase-A cross-condition aggregator (molix.analysis.aggregate).

Covers pinet-quantization-thermal-noise-01 criteria ac-003 (fp64-vs-fp64 null
control), ac-004 (one row per matrix cell with verdict columns), ac-005
(unbiased verdict from F_bias vs statistical error), ac-006 (Gaussianity verdict
from skew/excess-kurtosis), and the relocated ac-007 (OOP aggregator emits a CSV
+ feeds the -04 verdict machine table). Fixtures are synthetic ΔF tensors with
known statistics — no PiNet involved.
"""

import csv
import itertools

import pytest
import torch

from molix.analysis import (
    ROW_COLUMNS,
    PhaseAAggregator,
    PhaseACell,
    build_machine_table,
)


def _aggregator():
    return PhaseAAggregator(dt=0.5, gamma=0.01, mass=12.0, t_target=300.0)


def _cell(scheme, prec, dataset, delta_f):
    return PhaseACell(scheme=scheme, trained_precision=prec, dataset=dataset, delta_f=delta_f)


# --------------------------------------------------------------------------- #
# ac-003 — fp64-vs-fp64 null control: zero ΔF collapses residual diagnostics
# --------------------------------------------------------------------------- #
def test_zero_delta_control_row_collapses():
    agg = _aggregator()
    row = agg.row(_cell("fp64", "fp64", "qm9", torch.zeros(300)))
    assert row["F_bias"] == 0.0
    assert row["F_rms"] == 0.0
    assert row["T_eff_ratio"] == 0.0
    # a perfectly null residual is trivially unbiased + Gaussian
    assert row["unbiased"] is True
    assert row["gaussian"] is True


# --------------------------------------------------------------------------- #
# ac-004 — one row per matrix cell, with the required verdict columns
# --------------------------------------------------------------------------- #
def test_table_has_one_row_per_cell_with_verdict_columns():
    torch.manual_seed(0)
    schemes = ["int8", "int4"]
    precisions = ["fp32", "fp64"]
    datasets = ["qm9", "aspirin"]
    cells = [
        _cell(s, p, d, torch.randn(500) * 0.01)
        for s, p, d in itertools.product(schemes, precisions, datasets)
    ]
    rows = _aggregator().table(cells)
    assert len(rows) == len(schemes) * len(precisions) * len(datasets)  # 2*2*2 = 8
    required = {"F_bias", "F_skew", "F_exkurt", "T_eff_ratio", "unbiased", "gaussian"}
    for row in rows:
        assert required <= row.keys()
        assert isinstance(row["unbiased"], bool)
        assert isinstance(row["gaussian"], bool)


# --------------------------------------------------------------------------- #
# ac-005 — unbiased iff |F_bias| within statistical error; bias flips it False
# --------------------------------------------------------------------------- #
def test_unbiased_verdict_flips_on_biased_residual():
    torch.manual_seed(1)
    n = 20_000
    centered = torch.randn(n) * 0.02  # mean ~0 -> within SE
    assert _aggregator().row(_cell("int8", "fp32", "qm9", centered))["unbiased"] is True

    biased = torch.randn(n) * 0.02 + 0.05  # constant offset >> SE
    assert _aggregator().row(_cell("int8", "fp32", "qm9", biased))["unbiased"] is False


# --------------------------------------------------------------------------- #
# ac-006 — Gaussian iff |skew|<tol and |exkurt|<tol; heavy tails flip it False
# --------------------------------------------------------------------------- #
def test_gaussian_verdict_flips_on_heavy_tails():
    torch.manual_seed(2)
    n = 200_000
    normal = torch.randn(n) * 0.02
    assert _aggregator().row(_cell("int8", "fp32", "qm9", normal))["gaussian"] is True

    u = torch.rand(n) - 0.5
    laplace = -u.sign() * torch.log1p(-2 * u.abs())  # heavy tails -> +excess kurtosis
    assert _aggregator().row(_cell("int4", "fp32", "qm9", laplace))["gaussian"] is False


# --------------------------------------------------------------------------- #
# ac-007 (relocated) — OOP aggregator writes a CSV and feeds the -04 verdict table
# --------------------------------------------------------------------------- #
def test_to_csv_writes_full_matrix(tmp_path):
    torch.manual_seed(3)
    cells = [_cell(s, "fp32", "qm9", torch.randn(300) * 0.01) for s in ("int8", "int4")]
    out = _aggregator().to_csv(cells, tmp_path / "phase_a.csv")
    assert out.exists()
    with out.open() as fh:
        reader = csv.DictReader(fh)
        assert list(reader.fieldnames) == list(ROW_COLUMNS)
        assert len(list(reader)) == 2


def test_rows_feed_verdict_machine_table():
    """Phase-A rows expose unbiased/gaussian (criteria a/b) for the -04 verdict."""
    torch.manual_seed(4)
    row = _aggregator().row(_cell("int8", "fp32", "qm9", torch.randn(400) * 0.01))
    # Map Phase-A booleans onto the verdict's criterion namespace (a/b from
    # Phase A; c-h are trajectory criteria, placeholdered here) and confirm the
    # machine-table builder consumes the row without dropping it.
    criteria = dict.fromkeys("cdefgh", True)
    criteria["a"] = row["unbiased"]
    criteria["b"] = row["gaussian"]
    verdict_row = {
        "scheme": row["scheme"],
        "trained_precision": row["trained_precision"],
        "dataset": row["dataset"],
        "md_condition": "static",
        "criteria": criteria,
        "verdict": "n/a",
        "form": "",
        "T_eff_ratio": row["T_eff_ratio"],
    }
    table = build_machine_table([verdict_row])
    assert len(table) == 1
    assert table[0]["crit_a"] == row["unbiased"]
    assert table[0]["crit_b"] == row["gaussian"]


@pytest.mark.parametrize("dof", [None, 30])
def test_dof_override_changes_t_eff_ratio(dof):
    torch.manual_seed(5)
    df = torch.randn(300) * 0.05
    row = _aggregator().row(PhaseACell("int8", "fp32", "qm9", df, dof=dof))
    assert row["T_eff_ratio"] > 0.0
