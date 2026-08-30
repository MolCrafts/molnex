#!/usr/bin/env python3
"""Aspirin parity gate for ``pair_style molnex`` — a **developer-run local gate**.

Spec: ``lammps-iface-04-facade`` (criterion ``ac-013``). This tool answers one
question: does the refactored pair style still produce, on one fixed input deck,
the energy and forces recorded from a run of the pre-refactor pair style?

Why ``scripts/`` and not ``regressions/``
-----------------------------------------
``regressions/README.md`` defines that directory as pure-Python public-API
scenario scripts and **forbids subprocess**. This gate must drive an external
LAMMPS binary — driving that binary *is* its job — so it cannot live there. It
also cannot live in CI: no CI runner has LAMMPS or the arch-specific
``molnexplugin.so``. It is therefore a *recorded local debt*: the chain
implementer owes one real run at delivery, and ``ac-013`` may not be marked
passing without the attached output. This file never skips and never fabricates
a value; it exits non-zero when it cannot do its job.

What it does NOT cover
----------------------
The deck is ``atom_style atomic``, so LAMMPS reports ``molecular == 0`` and the
system has **no bond topology and therefore no special-bonds pairs at all**.
This case consequently exercises **none** of ``CutoffGraphImpl::init``'s
``special_bonds`` interception (the defect found while drafting the spec). The
only coverage of that fix is the ctest
``impl_init_rejects_special_bonds_on_molecular``; do not read a green run here
as evidence about it. The same statement is recorded in the spec.

Modes
-----
``--capture``
    Runs the deck on a real LAMMPS build and **prints** a ready-to-paste
    ``Goldens(...)`` block to stdout. Requires ``--model-dir``. Add
    ``--write-goldens`` to have the tool substitute that block into this file
    in place, between the two ``GOLDENS BEGIN`` / ``GOLDENS END`` marker lines.

    The two mechanisms are deliberate and are the *least-magic* pair available:
    the thing printed for a human to paste and the thing written by the flag are
    byte-identical, and the write touches exactly the marked region — so the
    whole effect of a capture is visible as one ``git diff`` hunk. No JSON
    sidecar is emitted: a second on-disk format for the same numbers is a second
    thing that can drift from the literals actually compared against. The
    emitted block is already ``ruff check`` / ``ruff format`` clean, so a
    capture never leaves the repo's lint gate red.

default (verify)
    Re-runs the same deck and compares against the embedded literals.

Validity of the goldens
-----------------------
Goldens are valid **only** for the embedded deck (geometry, atom types, box,
boundary, run commands) together with **one specific exported model**. Both are
pinned by hash — ``deck_sha256`` over the generated deck text and
``model_sha256`` over the export's ``.meta.json`` and ``.pt2`` bytes — plus the
recorded ``units`` style and artifact name. Any mismatch is an error, not a
warning: comparing this deck's numbers against a different model or a mutated
deck would be a fabricated pass.

Tolerance (the exact meaning of "relative 1e-6")
------------------------------------------------
- energy: ``|E - E_gold| / |E_gold|`` (absolute difference if ``E_gold == 0``).
- forces: ``max_i ‖f_i - f_i_gold‖ / max_j ‖f_j_gold‖`` — per-atom vector
  deviation against the largest golden force magnitude. Component-wise relative
  deviation is deliberately *not* used: a component that is ~1e-12 by symmetry
  would demand ~1e-18 absolute agreement, which no summation-order-sensitive
  float sum survives, and the gate would then fail for reasons that have
  nothing to do with the refactor it exists to police.

The bound is 1e-6 rather than bitwise equality because ``lammps-iface-01-meta``
replaced hard-coded conversion floats with pint-evaluated ones (relative shift
~1e-8).

Provenance block schema
-----------------------
The schema is defined here; the captured **values** live in
``GOLDENS.provenance`` so that numbers and the conditions that produced them sit
in one rewrite region and cannot drift apart::

    system            : formula, atom count, what the molecule is
    atom_style        : LAMMPS atom style + the molecular == 0 consequence
    boundary          : LAMMPS boundary flags
    lammps_version    : version banner of the binary that produced the goldens
    lammps_binary     : resolved path to that binary
    plugin            : resolved path to the loaded molnexplugin.so
    plugin_commit     : git HEAD of this repo at capture (+ "-dirty")
    torch_version     : installed torch distribution version
    device            : export target device, from the export's meta.json
    precision         : model compute dtype, from the meta lammps block
    model_units       : the export's native unit system
    contract          : meta lammps contract (cutoff-graph / fixed-graph)
    execution         : meta lammps execution (dynamic / cuda-graph)
    capture_command   : the exact command line that captured the values
    date              : UTC date of capture
    tolerance         : the comparison rule reproduced above

Usage
-----
::

    # capture (needs a LAMMPS build + the plugin + an export directory)
    python scripts/lammps_iface_parity.py --capture \\
        --model-dir /path/to/pinet_aspirin \\
        --lammps /path/to/lmp --plugin /path/to/molnexplugin.so [--write-goldens]

    # verify
    python scripts/lammps_iface_parity.py [--model-dir /path/to/pinet_aspirin]

``--lammps`` falls back to ``$LAMMPS_BIN`` then to ``lmp`` / ``lmp_serial`` /
``lmp_mpi`` / ``lammps`` on ``PATH``. ``--plugin`` falls back to
``$MOLNEX_PLUGIN``; the plugin is always required because ``pair_style molnex``
is a runtime plugin and is never compiled into LAMMPS (see
``interface/lammps/README.md``).

Exit codes
----------
- ``0`` — every quantity within tolerance.
- ``1`` — at least one quantity out of tolerance (a real parity failure).
- ``2`` — the check could not be run: goldens not captured, no LAMMPS build, no
  plugin, hash/units mismatch, or the LAMMPS run itself failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from math import sqrt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

RTOL = 1e-6
"""Relative tolerance for both quantities (see the module docstring)."""

_RUN_TIMEOUT_S = 1800
_BOX_PAD_ANGSTROM = 10.0
#: Mirrors ``[tool.ruff] line-length`` in pyproject.toml; used only when emitting goldens.
_LINE_LIMIT = 100
_LAMMPS_CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lammps")
_ENERGY_TAG = "MOLNEX_ENERGY"

#: LAMMPS atom type -> atomic number, in the order used by ``pair_coeff * *``.
_TYPE_TO_Z: tuple[int, ...] = (6, 1, 8)
#: LAMMPS atom type -> mass (amu).
_TYPE_TO_MASS: tuple[float, ...] = (12.011, 1.008, 15.999)
_TYPE_LABEL: tuple[str, ...] = ("C", "H", "O")

_GOLDENS_BEGIN = "# --- GOLDENS BEGIN ---"
_GOLDENS_END = "# --- GOLDENS END ---"

_CAPTURE_HINT = (
    "goldens not yet captured — run --capture on a LAMMPS build:\n"
    "  python scripts/lammps_iface_parity.py --capture --model-dir <export_dir> \\\n"
    "      --lammps <lmp> --plugin <molnexplugin.so> --write-goldens\n"
    "This tool will not skip and will not invent values."
)


class ParityError(RuntimeError):
    """The parity check could not be performed (exit code 2)."""


@dataclass(frozen=True)
class RunOutcome:
    """One LAMMPS evaluation of the embedded deck.

    Attributes:
        energy: Total potential energy in the deck's LAMMPS unit style.
        forces: Per-atom forces ``(N, 3)`` ordered by LAMMPS atom id.
        lammps_version: The version string from the run banner.
    """

    energy: float
    forces: tuple[tuple[float, float, float], ...]
    lammps_version: str


@dataclass(frozen=True)
class Comparison:
    """The verdict for one compared quantity."""

    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Goldens:
    """Recorded reference values plus everything needed to judge their validity.

    Attributes:
        deck_sha256: Hash of the generated deck text (data file + input template).
        model_sha256: Hash of the export's ``.meta.json`` and ``.pt2`` bytes.
        model_dir: Export directory used at capture (default for verify runs).
        model_name: Artifact basename inside ``model_dir``.
        units: LAMMPS unit style the deck was run under.
        energy: Golden total potential energy.
        forces: Golden per-atom forces ``(N, 3)``, ordered by atom id.
        provenance: The Provenance block, schema per the module docstring.
    """

    deck_sha256: str
    model_sha256: str
    model_dir: str
    model_name: str
    units: str
    energy: float
    forces: tuple[tuple[float, float, float], ...]
    provenance: dict[str, str]

    def compare(self, outcome: RunOutcome) -> list[Comparison]:
        """Judge ``outcome`` against these goldens at :data:`RTOL`."""
        return [self._compare_energy(outcome.energy), self._compare_forces(outcome.forces)]

    def _compare_energy(self, energy: float) -> Comparison:
        delta = abs(energy - self.energy)
        rel = delta / abs(self.energy) if self.energy != 0.0 else delta
        return Comparison(
            "energy",
            rel <= RTOL,
            f"got={energy!r} golden={self.energy!r} rel={rel:.3e} (rtol {RTOL:g})",
        )

    def _compare_forces(self, forces: tuple[tuple[float, float, float], ...]) -> Comparison:
        if len(forces) != len(self.forces):
            return Comparison(
                "forces",
                False,
                f"atom count changed: got {len(forces)}, golden {len(self.forces)}",
            )
        scale = max(_norm(f) for f in self.forces)
        if scale == 0.0:
            return Comparison(
                "forces",
                False,
                "golden forces are all zero — that is not a usable parity reference; re-capture",
            )
        worst_rel = 0.0
        worst_atom = 0
        for index, (got, gold) in enumerate(zip(forces, self.forces, strict=True)):
            rel = _norm(tuple(a - b for a, b in zip(got, gold, strict=True))) / scale
            if rel > worst_rel:
                worst_rel, worst_atom = rel, index
        label = ASPIRIN_ATOMS[worst_atom][0] if worst_atom < len(ASPIRIN_ATOMS) else "?"
        return Comparison(
            "forces",
            worst_rel <= RTOL,
            f"max_i |df_i| / max_j |f_j| = {worst_rel:.3e} at atom {worst_atom + 1} "
            f"({label}) (rtol {RTOL:g})",
        )


#: Embedded aspirin conformer: ``(label, lammps_type, x, y, z)`` in Å.
#:
#: Acetylsalicylic acid, C9H8O4, 21 atoms — a benzene ring (C1..C6) carrying a
#: twisted carboxyl at C1 and an out-of-plane acetoxy at C2. Built from standard
#: bond lengths and angles; it is a *reasonable* conformer, not a minimum. Only
#: its reproducibility matters: capture and verify use these same literals, and
#: any edit here invalidates every golden (enforced by ``deck_sha256``).
ASPIRIN_ATOMS: tuple[tuple[str, int, float, float, float], ...] = (
    ("C1", 1, 1.390, 0.000, 0.000),
    ("C2", 1, 0.695, 1.204, 0.000),
    ("C3", 1, -0.695, 1.204, 0.000),
    ("C4", 1, -1.390, 0.000, 0.000),
    ("C5", 1, -0.695, -1.204, 0.000),
    ("C6", 1, 0.695, -1.204, 0.000),
    ("H3", 2, -1.235, 2.139, 0.000),
    ("H4", 2, -2.470, 0.000, 0.000),
    ("H5", 2, -1.235, -2.139, 0.000),
    ("H6", 2, 1.235, -2.139, 0.000),
    ("C7", 1, 2.880, 0.000, 0.000),
    ("O1", 3, 3.485, 0.358, 0.985),
    ("O2", 3, 3.550, -0.397, -1.090),
    ("HO", 2, 4.495, -0.322, -0.885),
    ("O3", 3, 1.385, 2.399, 0.000),
    ("C8", 1, 1.694, 2.934, 1.212),
    ("O4", 3, 1.367, 2.368, 2.218),
    ("C9", 1, 2.443, 4.231, 1.290),
    ("HM1", 2, 2.597, 4.499, 2.336),
    ("HM2", 2, 1.867, 5.013, 0.796),
    ("HM3", 2, 3.408, 4.123, 0.796),
)

# ``atom_style atomic`` is load-bearing, not incidental: it is what makes
# ``molecular == 0``, which is what keeps this deck out of the special_bonds
# path (see the module docstring). ``sort 0 0.0`` disables spatial re-ordering so
# the edge set is assembled in a fixed order run to run.
_INPUT_TEMPLATE = """\
# molnex aspirin parity deck - generated by scripts/lammps_iface_parity.py
# atom_style atomic => molecular == 0 => no special-bonds pairs exist, so this
# deck exercises none of CutoffGraphImpl::init's special_bonds interception.
units           {units}
atom_style      atomic
boundary        f f f
newton          on
atom_modify     sort 0 0.0

plugin          load {plugin}

read_data       {data_file}

pair_style      molnex {model_dir} {model_name}
pair_coeff      * * {z_map}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check no

thermo          1
thermo_style    custom step pe
thermo_modify   norm no

run             0

print           "{energy_tag} $(pe:%.17g)"
write_dump      all custom {dump_file} id fx fy fz modify sort id format float %.17g
"""


class AspirinDeck:
    """The embedded single-molecule deck: its data file, its input script, its hash.

    The deck is the unit the goldens are pinned to. Its fingerprint is taken over
    the *unsubstituted* input template plus the data-file text, so it is
    independent of paths, of the model, and of the unit style — those are
    recorded and checked as separate fields, which gives a specific diagnostic
    instead of one opaque hash mismatch.
    """

    def data_text(self) -> str:
        """Render the LAMMPS data file for the embedded conformer."""
        lo, hi = self._box()
        lines = [
            "LAMMPS data file: molnex aspirin parity deck (single molecule)",
            "",
            f"{len(ASPIRIN_ATOMS)} atoms",
            f"{len(_TYPE_TO_Z)} atom types",
            "",
        ]
        for axis, (low, high) in zip("xyz", zip(lo, hi, strict=True), strict=True):
            lines.append(f"{low:.6f} {high:.6f} {axis}lo {axis}hi")
        lines += ["", "Masses", ""]
        for index, mass in enumerate(_TYPE_TO_MASS):
            lines.append(f"{index + 1} {mass:.6f}  # {_TYPE_LABEL[index]}")
        lines += ["", "Atoms # atomic", ""]
        for index, (label, type_id, x, y, z) in enumerate(ASPIRIN_ATOMS):
            lines.append(f"{index + 1} {type_id} {x:.6f} {y:.6f} {z:.6f}  # {label}")
        return "\n".join(lines) + "\n"

    def input_text(
        self,
        *,
        plugin: Path,
        model_dir: Path,
        model_name: str,
        units: str,
        data_file: str,
        dump_file: str,
    ) -> str:
        """Render the LAMMPS input script that evaluates the deck once."""
        return _INPUT_TEMPLATE.format(
            units=units,
            plugin=plugin,
            data_file=data_file,
            model_dir=model_dir,
            model_name=model_name,
            z_map=" ".join(str(z) for z in _TYPE_TO_Z),
            energy_tag=_ENERGY_TAG,
            dump_file=dump_file,
        )

    def fingerprint(self) -> str:
        """SHA-256 over the deck's data text and its unsubstituted input template."""
        digest = hashlib.sha256()
        digest.update(self.data_text().encode())
        digest.update(_INPUT_TEMPLATE.encode())
        return digest.hexdigest()

    def _box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        x = [atom[2] for atom in ASPIRIN_ATOMS]
        y = [atom[3] for atom in ASPIRIN_ATOMS]
        z = [atom[4] for atom in ASPIRIN_ATOMS]
        pad = _BOX_PAD_ANGSTROM
        lo = (min(x) - pad, min(y) - pad, min(z) - pad)
        hi = (max(x) + pad, max(y) + pad, max(z) + pad)
        return lo, hi


class LammpsRun:
    """One binding of binary + plugin + export + unit style, evaluated on a deck."""

    def __init__(
        self,
        *,
        lammps_bin: Path,
        plugin: Path,
        model_dir: Path,
        model_name: str,
        units: str,
        deck: AspirinDeck,
    ) -> None:
        self.lammps_bin = lammps_bin
        self.plugin = plugin
        self.model_dir = model_dir
        self.model_name = model_name
        self.units = units
        self.deck = deck

    def execute(self) -> RunOutcome:
        """Run LAMMPS once in a temp directory and read back energy and forces."""
        with tempfile.TemporaryDirectory(prefix="molnex-parity-") as tmp:
            workdir = Path(tmp)
            (workdir / "aspirin.data").write_text(self.deck.data_text())
            (workdir / "in.parity").write_text(
                self.deck.input_text(
                    plugin=self.plugin,
                    model_dir=self.model_dir,
                    model_name=self.model_name,
                    units=self.units,
                    data_file="aspirin.data",
                    dump_file="forces.dump",
                )
            )
            completed = self._invoke(workdir)
            dump_path = workdir / "forces.dump"
            if not dump_path.is_file():
                raise ParityError(
                    "LAMMPS produced no force dump; last output:\n" + _tail(completed)
                )
            return RunOutcome(
                energy=_parse_energy(completed),
                forces=_parse_dump(dump_path.read_text()),
                lammps_version=_parse_version(completed),
            )

    def _invoke(self, workdir: Path) -> str:
        argv = [str(self.lammps_bin), "-in", "in.parity", "-log", "log.parity"]
        try:
            proc = subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParityError(
                f"LAMMPS timed out after {_RUN_TIMEOUT_S}s: {shlex.join(argv)}"
            ) from exc
        except OSError as exc:
            raise ParityError(f"could not execute {self.lammps_bin}: {exc}") from exc
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise ParityError(
                f"LAMMPS exited {proc.returncode} ({shlex.join(argv)}); last output:\n"
                + _tail(output)
            )
        return output


def _norm(vector: tuple[float, ...]) -> float:
    return sqrt(sum(component * component for component in vector))


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _parse_energy(output: str) -> float:
    matches = re.findall(rf"^{_ENERGY_TAG}\s+(\S+)\s*$", output, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ParityError(
            f"expected exactly one '{_ENERGY_TAG} <value>' line in the LAMMPS output, "
            f"found {len(matches)}; last output:\n" + _tail(output)
        )
    try:
        return float(matches[0])
    except ValueError as exc:
        raise ParityError(f"unparseable energy literal {matches[0]!r}") from exc


def _parse_version(output: str) -> str:
    match = re.search(r"^LAMMPS \(([^)]*)\)", output, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def _parse_dump(text: str) -> tuple[tuple[float, float, float], ...]:
    """Read ``id fx fy fz`` from a single-snapshot LAMMPS custom dump."""
    lines = text.splitlines()
    try:
        count = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
    except (ValueError, IndexError) as exc:
        raise ParityError("force dump has no 'ITEM: NUMBER OF ATOMS' section") from exc
    header = next((i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")), None)
    if header is None:
        raise ParityError("force dump has no 'ITEM: ATOMS' section")
    columns = lines[header].split()[2:]
    try:
        indices = [columns.index(name) for name in ("id", "fx", "fy", "fz")]
    except ValueError as exc:
        raise ParityError(f"force dump lacks an id/fx/fy/fz column: {columns}") from exc
    rows = lines[header + 1 : header + 1 + count]
    if len(rows) != count:
        raise ParityError(f"force dump declares {count} atoms but holds {len(rows)} rows")
    id_at, fx_at, fy_at, fz_at = indices
    by_id: dict[int, tuple[float, float, float]] = {}
    for row in rows:
        fields = row.split()
        by_id[int(fields[id_at])] = (
            float(fields[fx_at]),
            float(fields[fy_at]),
            float(fields[fz_at]),
        )
    missing = sorted(set(range(1, count + 1)) - by_id.keys())
    if missing:
        raise ParityError(f"force dump is missing atom ids {missing}")
    return tuple(by_id[atom_id] for atom_id in range(1, count + 1))


def _resolve_lammps(explicit: str | None) -> Path:
    given = explicit or os.environ.get("LAMMPS_BIN")
    if given:
        path = Path(given).expanduser()
        resolved = path if path.is_file() else (shutil.which(given) or "")
        if not resolved or not os.access(resolved, os.X_OK):
            raise ParityError(f"LAMMPS binary {given!r} is not an executable file")
        return Path(resolved).resolve()
    for candidate in _LAMMPS_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
    raise ParityError(
        "no LAMMPS build found — pass --lammps <path>, set $LAMMPS_BIN, or put one of "
        f"{', '.join(_LAMMPS_CANDIDATES)} on PATH. This gate needs a real LAMMPS with the "
        "molnex plugin; it will not skip."
    )


def _resolve_plugin(explicit: str | None) -> Path:
    given = explicit or os.environ.get("MOLNEX_PLUGIN")
    if not given:
        raise ParityError(
            "no molnex plugin given — pass --plugin <molnexplugin.so> or set $MOLNEX_PLUGIN. "
            "pair_style molnex is a runtime plugin and is never compiled into LAMMPS; build it "
            "against the same LAMMPS (see interface/lammps/README.md)."
        )
    path = Path(given).expanduser()
    if not path.is_file():
        raise ParityError(f"molnex plugin {path} does not exist")
    return path.resolve()


def _model_files(model_dir: Path, model_name: str) -> tuple[Path, Path]:
    meta = model_dir / f"{model_name}.meta.json"
    package = model_dir / f"{model_name}.pt2"
    for path in (meta, package):
        if not path.is_file():
            raise ParityError(
                f"{path} not found — --model-dir must be an export directory produced by "
                "molix.engine.export_for_lammps (use --model-name for a non-default basename)"
            )
    return meta, package


def _model_fingerprint(meta: Path, package: Path) -> str:
    digest = hashlib.sha256()
    for path in (meta, package):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_provenance(meta: Path) -> dict[str, str]:
    """Read the export's self-description; absent keys are reported as unknown."""
    try:
        document = json.loads(meta.read_text())
    except (OSError, ValueError) as exc:
        raise ParityError(f"could not read {meta}: {exc}") from exc
    block = document.get("lammps", {})
    return {
        "device": str(document.get("device", "unknown")),
        "precision": str(block.get("model_dtype", "unknown")),
        "model_units": str(block.get("units", "unknown")),
        "contract": str(block.get("contract", "unknown")),
        "execution": str(block.get("execution", "unknown")),
    }


def _torch_version() -> str:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "not installed"


def _plugin_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def _literal(text: str) -> str:
    """Render ``text`` as a double-quoted Python string literal.

    ``json.dumps`` rather than ``repr``: ruff format normalises string quotes to
    double, so a ``repr``-rendered block would leave this file failing
    ``ruff format --check`` the moment a capture wrote it. Emitting what the
    formatter already wants keeps ``--write-goldens`` gate-clean by construction
    instead of by a follow-up instruction someone will forget.
    """
    return json.dumps(text)


def _emitted(line: str) -> str:
    """Suffix an emitted data line with ``noqa`` only when it must exceed the limit.

    A captured absolute path or capture command can be longer than the project's
    line limit, and neither ruff format nor any wrapping can shorten a string
    literal that must stay byte-exact. Truncating provenance to satisfy a
    formatting rule would be the dishonest fix, so the overlong line carries a
    narrow, per-line suppression instead — and only when it is actually over.
    """
    return f"{line}  # noqa: E501" if len(line) > _LINE_LIMIT else line


def _render_goldens(goldens: Goldens) -> str:
    """Render the pasteable / writable ``GOLDENS`` assignment (marker lines included)."""
    # Force rows cannot overflow the limit (three float reprs plus a short label
    # is under 95 columns at the widest), so they carry the atom label instead.
    forces = "\n".join(
        f"        ({fx!r}, {fy!r}, {fz!r}),  # {ASPIRIN_ATOMS[i][0]}"
        for i, (fx, fy, fz) in enumerate(goldens.forces)
    )
    provenance = "\n".join(
        _emitted(f"        {_literal(key)}: {_literal(value)},")
        for key, value in goldens.provenance.items()
    )
    fields = "\n".join(
        _emitted(f"    {name}={value},")
        for name, value in (
            ("deck_sha256", _literal(goldens.deck_sha256)),
            ("model_sha256", _literal(goldens.model_sha256)),
            ("model_dir", _literal(goldens.model_dir)),
            ("model_name", _literal(goldens.model_name)),
            ("units", _literal(goldens.units)),
            ("energy", repr(goldens.energy)),
        )
    )
    return (
        f"{_GOLDENS_BEGIN}\n"
        "GOLDENS: Goldens | None = Goldens(\n"
        f"{fields}\n"
        "    forces=(\n"
        f"{forces}\n"
        "    ),\n"
        "    provenance={\n"
        f"{provenance}\n"
        "    },\n"
        ")\n"
        f"{_GOLDENS_END}\n"
    )


def _write_goldens(block: str) -> None:
    """Substitute ``block`` between this file's two marker lines, in place."""
    lines = THIS_FILE.read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\n") == _GOLDENS_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.rstrip("\n") == _GOLDENS_END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ParityError(
            f"{THIS_FILE} must contain exactly one {_GOLDENS_BEGIN!r} line followed by exactly "
            f"one {_GOLDENS_END!r} line; found {len(starts)} and {len(ends)}"
        )
    updated = lines[: starts[0]] + [block] + lines[ends[0] + 1 :]
    THIS_FILE.write_text("".join(updated))


def _capture(args: argparse.Namespace, deck: AspirinDeck) -> int:
    if not args.model_dir:
        raise ParityError("--capture requires --model-dir <export_dir>")
    lammps_bin = _resolve_lammps(args.lammps)
    plugin = _resolve_plugin(args.plugin)
    model_dir = Path(args.model_dir).expanduser().resolve()
    meta, package = _model_files(model_dir, args.model_name)
    outcome = LammpsRun(
        lammps_bin=lammps_bin,
        plugin=plugin,
        model_dir=model_dir,
        model_name=args.model_name,
        units=args.units,
        deck=deck,
    ).execute()
    provenance = {
        "system": f"aspirin (acetylsalicylic acid), C9H8O4, {len(ASPIRIN_ATOMS)} atoms",
        "atom_style": "atomic (molecular == 0: no bonds, so no special_bonds pairs)",
        "boundary": "f f f",
        "lammps_version": outcome.lammps_version,
        "lammps_binary": str(lammps_bin),
        "plugin": str(plugin),
        "plugin_commit": _plugin_commit(),
        "torch_version": _torch_version(),
        **_model_provenance(meta),
        "capture_command": shlex.join(sys.argv),
        "date": datetime.now(UTC).date().isoformat(),
        "tolerance": f"rel {RTOL:g}; energy |dE|/|E|, forces max_i |df_i| / max_j |f_j|",
    }
    block = _render_goldens(
        Goldens(
            deck_sha256=deck.fingerprint(),
            model_sha256=_model_fingerprint(meta, package),
            model_dir=str(model_dir),
            model_name=args.model_name,
            units=args.units,
            energy=outcome.energy,
            forces=outcome.forces,
            provenance=provenance,
        )
    )
    print(f"# captured on LAMMPS {outcome.lammps_version} via {lammps_bin}")
    print(block, end="")
    if args.write_goldens:
        _write_goldens(block)
        print(f"# written in place into {THIS_FILE}", file=sys.stderr)
        print("# review the diff, then commit it with the run output", file=sys.stderr)
    else:
        print(
            "# not written: paste the block above between the GOLDENS marker lines, "
            "or re-run with --write-goldens",
            file=sys.stderr,
        )
    return 0


def _verify(args: argparse.Namespace, deck: AspirinDeck) -> int:
    # Goldens first, deliberately: without them a LAMMPS build proves nothing, so
    # reporting a missing binary would send the reader after the wrong thing.
    if GOLDENS is None:
        raise ParityError(_CAPTURE_HINT)
    goldens = GOLDENS
    lammps_bin = _resolve_lammps(args.lammps)
    plugin = _resolve_plugin(args.plugin)
    model_dir = Path(args.model_dir or goldens.model_dir).expanduser().resolve()
    meta, package = _model_files(model_dir, args.model_name)
    _assert_goldens_apply(goldens, deck, args, meta, package)
    outcome = LammpsRun(
        lammps_bin=lammps_bin,
        plugin=plugin,
        model_dir=model_dir,
        model_name=args.model_name,
        units=args.units,
        deck=deck,
    ).execute()
    print(f"deck      : {deck.fingerprint()[:16]}  model: {model_dir}")
    print(
        f"captured  : {goldens.provenance.get('date', '?')} on LAMMPS "
        f"{goldens.provenance.get('lammps_version', '?')}"
    )
    print(f"now       : LAMMPS {outcome.lammps_version} via {lammps_bin}")
    comparisons = goldens.compare(outcome)
    for comparison in comparisons:
        print(f"{comparison.label:<9}: {'OK  ' if comparison.ok else 'FAIL'}  {comparison.detail}")
    if all(comparison.ok for comparison in comparisons):
        print("PARITY OK")
        return 0
    print("PARITY FAILED", file=sys.stderr)
    return 1


def _assert_goldens_apply(
    goldens: Goldens,
    deck: AspirinDeck,
    args: argparse.Namespace,
    meta: Path,
    package: Path,
) -> None:
    """Refuse to compare unless the run reproduces the captured conditions."""
    fingerprint = deck.fingerprint()
    if fingerprint != goldens.deck_sha256:
        raise ParityError(
            f"deck changed since capture ({fingerprint} != {goldens.deck_sha256}); the embedded "
            "geometry or input template was edited, so the goldens no longer describe this run — "
            "re-capture rather than compare"
        )
    if args.units != goldens.units:
        raise ParityError(
            f"unit style {args.units!r} does not match the captured {goldens.units!r}"
        )
    if args.model_name != goldens.model_name:
        raise ParityError(
            f"model name {args.model_name!r} does not match the captured {goldens.model_name!r}"
        )
    model_sha = _model_fingerprint(meta, package)
    if model_sha != goldens.model_sha256:
        raise ParityError(
            f"model changed since capture ({model_sha} != {goldens.model_sha256}); goldens are "
            f"valid only for the export captured at {goldens.model_dir} — re-capture"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lammps_iface_parity.py",
        description=(
            "Aspirin parity gate for pair_style molnex (spec lammps-iface-04-facade, ac-013). "
            "Developer-run: needs a real LAMMPS build plus the molnex plugin. Never skips."
        ),
        epilog="exit codes: 0 parity OK, 1 parity FAILED, 2 cannot run the check",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run the deck and emit a Goldens(...) block instead of verifying",
    )
    parser.add_argument(
        "--write-goldens",
        action="store_true",
        help="with --capture, substitute the emitted block into this file in place",
    )
    parser.add_argument("--model-dir", help="export directory (required with --capture)")
    parser.add_argument("--model-name", default="model", help="artifact basename (default: model)")
    parser.add_argument("--lammps", help="LAMMPS binary (default: $LAMMPS_BIN, then PATH)")
    parser.add_argument("--plugin", help="molnexplugin.so (default: $MOLNEX_PLUGIN)")
    parser.add_argument(
        "--units",
        default="real",
        choices=("real", "metal"),
        help="LAMMPS unit style for the deck (default: real)",
    )
    args = parser.parse_args(argv)
    if args.write_goldens and not args.capture:
        parser.error("--write-goldens is only meaningful with --capture")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    deck = AspirinDeck()
    try:
        return _capture(args, deck) if args.capture else _verify(args, deck)
    except ParityError as exc:
        print(f"lammps_iface_parity: {exc}", file=sys.stderr)
        return 2


# The block below is the only machine-written region of this file. --capture
# --write-goldens replaces everything between the two marker lines and nothing
# else, so a capture shows up as exactly one reviewable diff hunk. Populate it by
# running a capture on a LAMMPS build; leaving it None keeps the gate red, which
# is the intended state until the chain implementer attaches a real run.
# --- GOLDENS BEGIN ---
GOLDENS: Goldens | None = None
# --- GOLDENS END ---


if __name__ == "__main__":
    raise SystemExit(main())
