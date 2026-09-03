#!/usr/bin/env python3
"""Run only the unit tests that mirror the files a diff touches.

The source↔test mirror rule (``src/<pkg>/<area>/<module>.py`` →
``tests/test_<pkg>/test_<area>/test_<module>.py``, see
:mod:`scripts.check_test_mirror`) makes the affected test set a pure function
of ``git diff``. This is the on-demand tier of the test policy: mirrored tests
first, the full default suite (``-m "not slow"``) before a push, ``-m slow``
only when the change touches a compile seam.

Usage::

    python scripts/affected_tests.py                 # print the pytest command
    python scripts/affected_tests.py --run           # run it
    python scripts/affected_tests.py --base main --run -- -x -q
    python scripts/affected_tests.py --run --slow    # include slow-marked tests

Mapping rules, in order:

* ``tests/**/test_*.py`` changed → itself.
* ``tests/**/conftest.py`` changed → that directory.
* ``src/<pkg>/**/__init__.py`` changed → ``tests/test_<pkg>/<mirrored dir>``.
* ``src/<pkg>/**/<module>.py`` changed → its mirror (exact, flat or fuzzy
  per :func:`check_test_mirror.find_mirror`); a module with no mirror is
  reported so the gap is visible rather than silently untested.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_test_mirror import ROOT, SRC, TESTS, find_mirror  # noqa: E402


def changed_paths(base: str | None) -> list[Path]:
    """Tracked changes against ``base`` (or the index/worktree) plus untracked files."""
    diff_cmd = ["git", "diff", "--name-only"]
    if base:
        diff_cmd.append(base)
    tracked = subprocess.run(diff_cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    names = set(tracked.stdout.split()) | set(untracked.stdout.split())
    return sorted(ROOT / n for n in names if n.endswith(".py"))


def affected_tests(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Map changed files to test targets.

    Returns:
        ``(targets, unmirrored)`` — test files or directories to run, and the
        changed source modules that have no mirror.
    """
    targets: set[Path] = set()
    unmirrored: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_relative_to(TESTS):
            if path.name.startswith("test_"):
                targets.add(path)
            elif path.name == "conftest.py":
                targets.add(path.parent)
            continue
        if not path.is_relative_to(SRC):
            continue
        rel = path.relative_to(SRC)
        if path.name == "__init__.py":
            pkg, *dirs = rel.parts[:-1]
            tdir = TESTS.joinpath(f"test_{pkg}", *[f"test_{d}" for d in dirs])
            if tdir.is_dir():
                targets.add(tdir)
            else:
                unmirrored.append(path)
            continue
        mirror = find_mirror(path)
        if mirror is None:
            unmirrored.append(path)
        else:
            targets.add(mirror)
    # The cross-cutting repository checks own no mirror; always run them.
    if targets:
        targets.add(TESTS / "test_repository_boundaries.py")
    # Drop files already covered by a selected directory.
    dirs = {t for t in targets if t.is_dir()}
    kept = [t for t in targets if t.is_dir() or not any(t.is_relative_to(d) for d in dirs)]
    return sorted(kept), unmirrored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--base", default=None, help="Git ref to diff against (default: worktree).")
    ap.add_argument("--run", action="store_true", help="Run pytest on the affected set.")
    ap.add_argument("--slow", action="store_true", help="Include slow-marked tests.")
    ap.add_argument("pytest_args", nargs="*", help="Extra pytest arguments after '--'.")
    args = ap.parse_args()

    targets, unmirrored = affected_tests(changed_paths(args.base))
    for src in unmirrored:
        print(f"no mirror: {src.relative_to(ROOT)}", file=sys.stderr)
    if not targets:
        print("no affected tests")
        return 0
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if args.slow:
        cmd += ["-m", ""]
    cmd += [str(t.relative_to(ROOT)) for t in targets]
    cmd += args.pytest_args
    print(" ".join(cmd))
    if not args.run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
