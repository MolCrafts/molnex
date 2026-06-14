"""Synthetic-fixture tests for the thermal-noise verdict classifier (criteria a-h)."""

from molix.analysis.verdict import (
    VERDICT_STRUCTURED,
    VERDICT_THERMAL,
    build_machine_table,
    characterize_failure,
    classify_cell,
    evaluate_criteria,
    run_verdict,
)


def _clean_cell(**over):
    cell = {
        "scheme": "int8",
        "trained_precision": "fp32",
        "dataset": "qm9",
        "F_bias": 0.0,
        "F_std": 1.0,
        "F_skew": 0.0,
        "F_exkurt": 0.0,
        "n": 900,
        "dt": 1.0,
        "tau_c": 1.0,
        "cov_offdiag": 0.01,
        "mean_net_force": 1e-5,
        "energy_drift_slope": 1e-6,
        "stationarity_var_spread": 0.1,
        "observable_delta": 0.01,
    }
    cell.update(over)
    return cell


def test_clean_cell_is_thermal():
    crit = evaluate_criteria(_clean_cell())
    assert all(crit.values())
    assert classify_cell(crit) == VERDICT_THERMAL
    assert characterize_failure(crit) == ""


_SINGLE_VIOLATIONS = {
    "a": {"F_bias": 0.5},
    "b": {"F_exkurt": 2.0},
    "c": {"tau_c": 10.0},
    "d": {"cov_offdiag": 0.5},
    "e": {"mean_net_force": 0.1},
    "f": {"energy_drift_slope": 1e-2},
    "g": {"stationarity_var_spread": 5.0},
    "h": {"observable_delta": 0.5},
}


def test_each_single_violation_is_structured_with_named_form():
    for crit_id, override in _SINGLE_VIOLATIONS.items():
        crit = evaluate_criteria(_clean_cell(**override))
        assert crit[crit_id] is False, f"criterion {crit_id} should fail"
        assert all(crit[c] for c in crit if c != crit_id), f"only {crit_id} should fail"
        assert classify_cell(crit) == VERDICT_STRUCTURED
        assert f"{crit_id}:" in characterize_failure(
            crit
        ) or "PES distortion" in characterize_failure(crit)


def test_biased_plus_correlated_is_pes_distortion():
    crit = evaluate_criteria(_clean_cell(F_bias=0.5, cov_offdiag=0.5))
    assert crit["a"] is False and crit["d"] is False
    form = characterize_failure(crit)
    assert form.startswith("PES distortion, not noise")


def test_machine_table_schema_is_stable():
    crit = evaluate_criteria(_clean_cell())
    verdict = {
        "scheme": "int8",
        "trained_precision": "fp32",
        "dataset": "qm9",
        "md_condition": None,
        "criteria": crit,
        "verdict": classify_cell(crit),
        "form": "",
        "T_eff_ratio": 0.12,
    }
    table = build_machine_table([verdict])
    row = table[0]
    for col in ("scheme", "verdict", "form", "T_eff_ratio", *(f"crit_{c}" for c in "abcdefgh")):
        assert col in row


def test_run_verdict_joins_and_flags_incomplete():
    static = [
        _clean_cell(scheme="int8"),
        _clean_cell(scheme="int4", F_bias=0.5),  # will be biased
        _clean_cell(scheme="fp16"),  # no matching traj -> incomplete
    ]
    traj = [
        {
            "scheme": "int8",
            "trained_precision": "fp32",
            "dataset": "qm9",
            "tau_c": 1.0,
            "dt": 1.0,
            "cov_offdiag": 0.01,
            "mean_net_force": 1e-5,
            "energy_drift_slope": 1e-6,
            "stationarity_var_spread": 0.1,
            "observable_delta": 0.01,
            "t_eff_colored_ratio": 0.2,
        },
        {
            "scheme": "int4",
            "trained_precision": "fp32",
            "dataset": "qm9",
            "tau_c": 1.0,
            "dt": 1.0,
            "cov_offdiag": 0.5,
            "mean_net_force": 1e-5,
            "energy_drift_slope": 1e-6,
            "stationarity_var_spread": 0.1,
            "observable_delta": 0.01,
        },
    ]
    report, table = run_verdict(static, traj)
    by_scheme = {r["scheme"]: r for r in table}
    assert by_scheme["int8"]["verdict"] == VERDICT_THERMAL
    assert by_scheme["int4"]["verdict"] == VERDICT_STRUCTURED
    assert by_scheme["fp16"]["verdict"] == "incomplete"
    assert "int8" in report and "verdict" in report
