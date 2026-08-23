"""Prove every evidence artifact in a run dir records one fingerprint (§0).

Usage:
    python scripts/mm_param_learning/verify_fingerprint.py runs/<run-dir>

Exit 0 when every *present* artifact carries the same hash as
``fingerprint.json``; exit 1 on a missing/empty fingerprint file, a present
artifact without the field, or a mismatch. ``pack_sweep.json`` is CUDA-only
and ``smoke.json`` / ``calibrate.json`` come from different launcher modes,
so an absent artifact is skipped — a present one must agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Evidence artifacts checked by name, all flat under the run dir.
ARTIFACTS: tuple[str, ...] = (
    "split.json",
    "metrics.json",
    "pack_sweep.json",
    "smoke.json",
    "calibrate.json",
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <run-dir>", file=sys.stderr)
        return 2

    root = Path(argv[1])
    fp_file = root / "fingerprint.json"
    if not fp_file.is_file():
        print(f"FAIL: missing fingerprint.json under {root}", file=sys.stderr)
        return 1
    blob = json.loads(fp_file.read_text())
    expected = blob.get("fingerprint")
    if not expected:
        print(f"FAIL: {fp_file} has no fingerprint hash", file=sys.stderr)
        return 1

    failures: list[str] = []
    checked: list[str] = []
    for name in ARTIFACTS:
        path = root / name
        if not path.is_file():
            continue
        checked.append(name)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{name}: unreadable JSON ({exc})")
            continue
        if data.get("fingerprint") != expected:
            failures.append(f"{name}: fingerprint {data.get('fingerprint')!r} != {expected!r}")

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    if not checked:
        print("FAIL: no evidence artifacts found to verify", file=sys.stderr)
        return 1

    print(f"OK: {len(checked)} artifacts match fingerprint {expected[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
