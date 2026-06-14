"""Tests for ``molnex check`` (:mod:`molix.cli.doctor` + the Typer command)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from molix.cli import app
from molix.cli.doctor import CheckResult, Doctor, Status, _parse_version

runner = CliRunner()


# -- pure helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2.12.0", (2, 12, 0)),
        ("2.12.0+cpu", (2, 12, 0)),
        ("3.10", (3, 10)),
        ("  1.2.3rc1", (1, 2, 3)),
        ("not-a-version", ()),
        ("", ()),
    ],
)
def test_parse_version(text: str, expected: tuple[int, ...]) -> None:
    assert _parse_version(text) == expected


def test_worst_status_ordering() -> None:
    ok = CheckResult("a", Status.OK, "")
    warn = CheckResult("b", Status.WARN, "")
    fail = CheckResult("c", Status.FAIL, "")
    assert Doctor.worst_status(()) is Status.OK
    assert Doctor.worst_status((ok, ok)) is Status.OK
    assert Doctor.worst_status((ok, warn)) is Status.WARN
    assert Doctor.worst_status((ok, warn, fail)) is Status.FAIL


def test_check_results_are_immutable() -> None:
    result = CheckResult("x", Status.OK, "detail")
    with pytest.raises((AttributeError, TypeError)):
        result.detail = "mutated"  # type: ignore[misc]


# -- the real environment (this venv must be healthy to run the suite) -----


def test_doctor_run_covers_all_checks() -> None:
    results = Doctor().run()
    names = {r.name for r in results}
    assert names == {
        "Python",
        "PyTorch",
        "CUDA",
        "Native ops",
        "cuEquivariance",
        "GPU telemetry",
        "Core deps",
    }


def test_core_invariants_pass_in_this_env() -> None:
    by_name = {r.name: r for r in Doctor().run()}
    # The test venv has Python, torch, native ops and core deps installed.
    for name in ("Python", "PyTorch", "Native ops", "cuEquivariance", "Core deps"):
        assert by_name[name].status is Status.OK, by_name[name]


# -- the Typer command -----------------------------------------------------


def test_check_command_runs() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "MolNex environment check" in result.stdout


def test_check_json_is_valid() -> None:
    result = runner.invoke(app, ["check", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert {item["name"] for item in payload} >= {"Python", "PyTorch", "CUDA"}
    for item in payload:
        assert item["status"] in {"ok", "warn", "fail"}


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0 or "Usage" in result.stdout
