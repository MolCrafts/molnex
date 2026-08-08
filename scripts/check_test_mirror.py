#!/usr/bin/env python3
"""Audit source↔test path mirroring (industrial layout gate).

Rules (soft for legacy; hard for the PiNet spine by default)::

    src/<pkg>/<area>/<module>.py
      → tests/test_<pkg>/test_<area>/test_<module>.py

Usage::

    python scripts/check_test_mirror.py              # report only
    python scripts/check_test_mirror.py --strict-pinet  # exit 1 on PiNet gaps
    python scripts/check_test_mirror.py --strict       # exit 1 on any gap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

# Paths that intentionally have no 1:1 unit test (CLI entry, pure re-export, …).
ALLOW_MISSING: frozenset[str] = frozenset(
    {
        "molix/cli/__main__.py",
        "molix/hooks/_utils.py",
        "molix/profiler/_utils.py",
        "molpot/heads/_common.py",
        "molpot/potentials/elec/_utils.py",
        "molix/datasets/_bond_adapter.py",
        "molix/datasets/_extxyz.py",
    }
)

# Packages / prefixes that must be mirrored under --strict-pinet (the flag
# name is historical; the gate now also guards the MD engine and the shared
# schema/units modules).
PINET_SPINE: tuple[str, ...] = (
    "molrep/interaction/pinet/",
    "molzoo/pinet/",
    "molpot/derivation/force.py",
    "molpot/derivation/energy.py",
    "molix/data/tasks/pad.py",
    "molix/md/",
    "molix/schema.py",
    "molix/units.py",
)


def _iter_src_modules() -> list[Path]:
    out: list[Path] = []
    for p in SRC.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def expected_test(src: Path) -> Path:
    rel = src.relative_to(SRC)
    parts = list(rel.parts)
    pkg = parts[0]
    rest = parts[1:]
    if not rest:
        return TESTS / f"test_{pkg}" / f"test_{src.stem}.py"
    *dirs, name = rest
    name = name.removesuffix(".py")
    tdirs = [f"test_{pkg}"] + [d if d.startswith("test_") else f"test_{d}" for d in dirs]
    return TESTS.joinpath(*tdirs) / f"test_{name}.py"


def find_mirror(src: Path) -> Path | None:
    """Return a reasonable existing test path, if any."""
    expected = expected_test(src)
    if expected.exists():
        return expected
    # Package package/module.py → tests/test_pkg/test_module.py (flat fallback)
    rel = src.relative_to(SRC)
    stem = src.stem
    pkg = rel.parts[0]
    flat = TESTS / f"test_{pkg}" / f"test_{stem}.py"
    if flat.exists():
        return flat
    # Fuzzy: any test_*stem*.py under test_pkg
    pkg_root = TESTS / f"test_{pkg}"
    if pkg_root.is_dir():
        hits = list(pkg_root.rglob(f"test_{stem}.py"))
        if hits:
            return hits[0]
        hits = list(pkg_root.rglob(f"test_{stem}_*.py"))
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="Fail on any missing mirror.")
    ap.add_argument(
        "--strict-pinet",
        action="store_true",
        help="Fail only when the PiNet spine is missing a mirror.",
    )
    args = ap.parse_args()

    missing: list[tuple[str, str]] = []
    present = 0
    for src in _iter_src_modules():
        rel = str(src.relative_to(SRC))
        if rel in ALLOW_MISSING:
            continue
        mirror = find_mirror(src)
        if mirror is None:
            missing.append((rel, str(expected_test(src).relative_to(ROOT))))
        else:
            present += 1

    print(f"mirrored: {present}  missing: {len(missing)}")
    if missing:
        print("\nMissing mirrors:")
        for rel, exp in missing:
            print(f"  {rel}  ->  {exp}")

    if args.strict and missing:
        return 1

    if args.strict_pinet:
        spine_missing = [
            (rel, exp)
            for rel, exp in missing
            if any(rel == p or rel.startswith(p) for p in PINET_SPINE)
        ]
        if spine_missing:
            print("\nPiNet spine gaps:")
            for rel, exp in spine_missing:
                print(f"  {rel}  ->  {exp}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
