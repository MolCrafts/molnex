#!/usr/bin/env python3
"""Parity gate for ``pair_style molnex`` — a **developer-run local gate**.

Specs: ``lammps-iface-04-facade`` (``ac-013``) and ``lammps-iface-05-pbc``
(``ac-014``). Two cases, and they answer different questions **by different
means** — the distinction is load-bearing, so it is stated before anything else:

``aspirin`` — *captured literals.*
    A single open-boundary molecule, compared against energies and forces
    recorded from a run of the pre-refactor pair style and embedded in this file
    as goldens. The reference is a **past run**, pinned by deck and model hashes.

``periodic`` — *live three-way agreement.* **Nothing is captured.**
    A small periodic water box evaluated three ways in one invocation: on one
    MPI rank, on several MPI ranks, and by a direct in-process Python evaluation
    of the same configuration. The reference is not a literal — the **Python
    evaluation is the reference**, computed now. There is nothing to embed,
    because what is being asserted is that three ways of computing the same
    number agree today, not that one of them reproduces a number from last week.

Why the periodic case has no goldens: its whole content is a *disagreement*
between execution paths. A captured literal would freeze one of the three paths
and turn a three-way agreement into a two-way comparison against whichever path
happened to run at capture time — including, possibly, a broken one.

What the periodic case proves (and only it can)
-----------------------------------------------
* **local-only site-energy summation** — a rank that summed its ghosts' site
  energies too would disagree with the Python evaluation, which sums each site
  exactly once. Visible already at one rank, since a periodic box has ghosts.
* **single reduction of ghost forces** — a pair style that reverse-communicates
  ghost forces *and* leaves them for the integrator sends each home twice. Only
  a decomposed run has an inter-rank reduction to double, so the one-rank versus
  many-rank leg is the only end-to-end evidence for it.
* **edge-order invariance, at full scale** — the C++ side builds edges in the
  LAMMPS neighbour list's raw order while the Python reference builds them in
  sorted ``nonzero()`` order. Agreement on a real model is the full-size version
  of a shuffle experiment.

Dependencies differ between the cases, deliberately
---------------------------------------------------
The aspirin case is **standard-library only**: it shells out to LAMMPS and
compares numbers. The periodic case **imports torch and molix**, because its
reference is an actual forward pass of the exported model. That import happens
inside the periodic path and nowhere else, so ``--case aspirin`` still runs on a
machine with no torch.

Why ``scripts/`` and not ``regressions/``
-----------------------------------------
``regressions/README.md`` defines that directory as pure-Python public-API
scenario scripts and **forbids subprocess**. This gate must drive an external
LAMMPS binary — driving that binary *is* its job — so it cannot live there. It
also cannot live in CI: no CI runner has LAMMPS or the arch-specific
``molnexplugin.so``. It is therefore a *recorded local debt*: the chain
implementer owes one real run at delivery, and ``ac-013`` / ``ac-014`` may not be
marked passing without the attached output. This file never skips and never
fabricates a value; it exits non-zero when it cannot do its job.

What the decks do NOT cover
---------------------------
Both decks are ``atom_style atomic``, so LAMMPS reports ``molecular == 0`` and
neither system has **any bond topology, hence no special-bonds pairs at all**.
They consequently exercise **none** of ``CutoffGraphImpl::init``'s
``special_bonds`` interception (the defect found while drafting the spec). The
only coverage of that fix is the ctest
``impl_init_rejects_special_bonds_on_molecular``; do not read a green run here
as evidence about it. The same statement is recorded in the spec.

Modes
-----
``--capture`` (aspirin only)
    Runs the aspirin deck on a real LAMMPS build and **prints** a ready-to-paste
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

    ``--capture`` is meaningless for the periodic case and is refused there
    rather than quietly ignored: that case has no literals by design.

default (verify)
    Runs both cases (choose one with ``--case``): the aspirin deck against the
    embedded literals, and the periodic deck as a live three-way comparison.

Validity of the goldens (aspirin only)
--------------------------------------
Goldens are valid **only** for the embedded deck (geometry, atom types, box,
boundary, run commands) together with **one specific exported model**. Both are
pinned by hash — ``deck_sha256`` over the generated deck text and
``model_sha256`` over the export's ``.meta.json`` and ``.pt2`` bytes — plus the
recorded ``units`` style and artifact name. Any mismatch is an error, not a
warning: comparing this deck's numbers against a different model or a mutated
deck would be a fabricated pass.

Tolerance (the exact meaning of "relative 1e-6")
------------------------------------------------
One rule, used by both cases; ``reference`` is the golden for aspirin and the
other evaluation for the periodic three-way:

- energy: ``|E - E_ref| / |E_ref|`` (absolute difference if ``E_ref == 0``).
- forces: ``max_i ‖f_i - f_i_ref‖ / max_j ‖f_j_ref‖`` — per-atom vector
  deviation against the largest reference force magnitude. Component-wise
  relative deviation is deliberately *not* used: a component that is ~1e-12 by
  symmetry would demand ~1e-18 absolute agreement, which no
  summation-order-sensitive float sum survives, and the gate would then fail for
  reasons that have nothing to do with the defect it exists to police.

The bound is 1e-6 rather than bitwise equality because ``lammps-iface-01-meta``
replaced hard-coded conversion floats with pint-evaluated ones (relative shift
~1e-8), and because the periodic case compares three different summation orders
over the same terms.

Provenance
----------
The aspirin case *stores* its provenance (the run that produced the literals is
gone, so its conditions must travel with them); the periodic case *prints* its
own at the end of every run, because there is nothing stored for it to describe
— a periodic result is only ever the run you just did, and the block below is
what gets pasted into the runbook entry in ``interface/lammps/README.md``.

The schema is defined here; for aspirin the captured **values** live in
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

The periodic block adds the fields that case turns on, and which ``ac-014``
requires to be recorded::

    ranks             : rank counts actually run (single and multi)
    mpirun            : the launcher template used for the multi-rank leg
    comm_modify       : the ghost cutoff the deck set, and the T * cutoff + skin
                        it was derived from
    receptive_field   : T, from the export meta
    ghost_atoms       : ghost image count the Python reference built
    box               : the periodic cell edge in Å

Usage
-----
::

    # capture the aspirin goldens (LAMMPS build + plugin + export directory)
    python scripts/lammps_iface_parity.py --capture \\
        --model-dir /path/to/pinet_aspirin \\
        --lammps /path/to/lmp --plugin /path/to/molnexplugin.so [--write-goldens]

    # verify both cases
    python scripts/lammps_iface_parity.py --model-dir /path/to/export \\
        --lammps /path/to/lmp --plugin /path/to/molnexplugin.so

    # only the periodic three-way, on 4 ranks
    python scripts/lammps_iface_parity.py --case periodic --ranks 4 \\
        --model-dir /path/to/export --mpirun "mpirun -np {ranks}"

``--lammps`` falls back to ``$LAMMPS_BIN`` then to ``lmp`` / ``lmp_serial`` /
``lmp_mpi`` / ``lammps`` on ``PATH``. ``--plugin`` falls back to
``$MOLNEX_PLUGIN``; the plugin is always required because ``pair_style molnex``
is a runtime plugin and is never compiled into LAMMPS (see
``interface/lammps/README.md``). ``--mpirun`` falls back to ``$LAMMPS_MPIRUN``
and then to ``mpirun -np {ranks}``; it is a template that **must** contain
``{ranks}``, so an ``srun -n {ranks}`` site works without a second flag. A
missing launcher is exit 2 naming it — the multi-rank leg is never skipped,
because skipping it is exactly the same as not testing the defect it exists for.

Exit codes
----------
- ``0`` — every quantity within tolerance.
- ``1`` — at least one quantity out of tolerance (a real parity failure).
- ``2`` — the check could not be run: goldens not captured, no LAMMPS build, no
  plugin, no MPI launcher, hash/units mismatch, an export that cannot serve the
  periodic case, a missing torch, or the LAMMPS run itself failed.
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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from itertools import product
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # the periodic case imports torch at call time; see PeriodicReference
    import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

RTOL = 1e-6
"""Relative tolerance for both quantities (see the module docstring)."""

_RUN_TIMEOUT_S = 1800
_BOX_PAD_ANGSTROM = 10.0
#: ``neighbor <skin> bin`` in both decks, in Å.
#:
#: Written here rather than left to LAMMPS' per-unit-style default because the
#: periodic deck's ``comm_modify cutoff`` is ``T * cutoff + skin`` and the Python
#: reference builds its ghost shell to the same width: three places need the same
#: number, so one of them must own it.
_SKIN_ANGSTROM = 2.0
#: Largest extended system (locals + ghost images) the Python reference will build.
#:
#: A guard, not a tuning knob. The ghost shell is ``T * cutoff + skin`` wide, so
#: its atom count grows as that cubed against a fixed box: a deep enough model
#: turns the reference into an out-of-memory crash with no diagnostic. Exceeding
#: this is exit 2 with the arithmetic spelled out, never a silent truncation of
#: the shell — a truncated shell is a WRONG reference, which is worse than none.
_MAX_REFERENCE_ATOMS = 250_000
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

#: Default MPI launcher template. ``{ranks}`` is mandatory; see ``--mpirun``.
_MPIRUN_DEFAULT = "mpirun -np {ranks}"

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
    """The verdict for one compared quantity.

    The two alternate constructors ARE the tolerance rule of the module
    docstring, in one place. They are constructors rather than free functions
    because the rule and the verdict it produces are the same concept, and they
    are shared rather than duplicated because both cases apply the identical
    bound to different references — the golden literals for aspirin, the other
    two evaluations for the periodic three-way.
    """

    label: str
    ok: bool
    detail: str

    @classmethod
    def energy(cls, label: str, got: float, reference: float) -> Comparison:
        """Judge one energy against ``reference`` at :data:`RTOL`."""
        delta = abs(got - reference)
        rel = delta / abs(reference) if reference != 0.0 else delta
        return cls(
            label,
            rel <= RTOL,
            f"got={got!r} ref={reference!r} rel={rel:.3e} (rtol {RTOL:g})",
        )

    @classmethod
    def forces(
        cls,
        label: str,
        got: tuple[tuple[float, float, float], ...],
        reference: tuple[tuple[float, float, float], ...],
        atom_labels: tuple[str, ...],
    ) -> Comparison:
        """Judge per-atom forces against ``reference`` at :data:`RTOL`."""
        if len(got) != len(reference):
            return cls(label, False, f"atom count differs: got {len(got)}, ref {len(reference)}")
        scale = max(_norm(f) for f in reference)
        if scale == 0.0:
            return cls(
                label,
                False,
                "reference forces are all zero — that is not a usable parity reference",
            )
        worst_rel = 0.0
        worst_atom = 0
        for index, (one, other) in enumerate(zip(got, reference, strict=True)):
            rel = _norm(tuple(a - b for a, b in zip(one, other, strict=True))) / scale
            if rel > worst_rel:
                worst_rel, worst_atom = rel, index
        name = atom_labels[worst_atom] if worst_atom < len(atom_labels) else "?"
        return cls(
            label,
            worst_rel <= RTOL,
            f"max_i |df_i| / max_j |f_j| = {worst_rel:.3e} at atom {worst_atom + 1} "
            f"({name}) (rtol {RTOL:g})",
        )


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
        labels = tuple(atom[0] for atom in ASPIRIN_ATOMS)
        return [
            Comparison.energy("energy", outcome.energy, self.energy),
            Comparison.forces("forces", outcome.forces, self.forces, labels),
        ]


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


class Deck(Protocol):
    """What ``LammpsRun`` needs of a deck: its data, its script, its fingerprint.

    A protocol rather than a base class because the two decks share no
    implementation — the whole of what they have in common is this interface,
    and inheriting one from the other would say they are variants of one system
    when they are two different systems (an open-boundary molecule and a
    periodic box) that happen to be run the same way.

    Anything a deck needs beyond these arguments — the periodic deck's
    ``comm_modify`` cutoff, say — is a **constructor** parameter of that deck, so
    that this signature stays the small set every deck genuinely shares.
    """

    def data_text(self) -> str:
        """Render the LAMMPS data file."""
        ...

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
        ...

    def fingerprint(self) -> str:
        """SHA-256 over the deck's data text and its unsubstituted input template."""
        ...


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


#: Embedded periodic water box: ``(label, lammps_type, x, y, z)`` in Å.
#:
#: 27 rigid water molecules (81 atoms) on a 3x3x3 lattice inside a cubic cell of
#: :data:`_WATER_BOX_ANGSTROM`, which is the edge length of 27 waters at ~1
#: g/cm³. Each molecule carries the experimental gas-phase geometry (O-H 0.9572
#: Å, H-O-H 104.52°), a deterministic random orientation and a deterministic
#: positional jitter; the closest inter-molecular contact is 1.63 Å, which is a
#: short hydrogen bond rather than a clash. Like the aspirin conformer this is a
#: *reasonable* snapshot, not a minimum — only its reproducibility matters.
#:
#: ``atom_style atomic`` with O and H as bare types is deliberate for a neural
#: potential: the model reads only ``Z`` and positions, so bond topology would
#: add nothing but a special-bonds requirement. The types reuse
#: :data:`_TYPE_TO_Z` (C, H, O), so both decks take the same ``pair_coeff`` line
#: and need the same exported species set; the water box simply never uses type 1.
#:
#: Every coordinate lies inside ``[0, L)``, so LAMMPS reads the deck without
#: remapping and the Python reference can build periodic images by plain
#: translation.
WATER_ATOMS: tuple[tuple[str, int, float, float, float], ...] = (
    ("O1", 3, 1.349521, 1.555499, 1.594660),
    ("H1a", 2, 2.166202, 1.209579, 1.954666),
    ("H1b", 2, 1.248287, 1.106897, 0.755173),
    ("O2", 3, 1.314460, 1.348488, 4.814288),
    ("H2a", 2, 0.854975, 0.772735, 4.203050),
    ("H2b", 2, 2.086599, 1.643629, 4.331664),
    ("O3", 3, 1.737493, 1.333652, 7.950607),
    ("H3a", 2, 2.667601, 1.327303, 8.176639),
    ("H3b", 2, 1.506377, 2.261704, 7.911404),
    ("O4", 3, 1.637552, 4.598015, 1.396978),
    ("H4a", 2, 1.785987, 3.791259, 0.903680),
    ("H4b", 2, 1.775515, 4.349288, 2.310943),
    ("O5", 3, 1.646194, 4.599281, 4.668626),
    ("H5a", 2, 2.203201, 3.965389, 4.216790),
    ("H5b", 2, 1.657034, 4.318884, 5.583772),
    ("O6", 3, 1.746320, 4.744277, 7.856464),
    ("H6a", 2, 2.277632, 4.902261, 7.076092),
    ("H6b", 2, 1.327065, 3.899196, 7.694300),
    ("O7", 3, 1.670689, 7.515778, 1.545776),
    ("H7a", 2, 1.280738, 6.759975, 1.985017),
    ("H7b", 2, 2.592370, 7.281240, 1.437476),
    ("O8", 3, 1.447335, 7.833503, 4.422025),
    ("H8a", 2, 1.092126, 8.403196, 3.739744),
    ("H8b", 2, 0.684649, 7.371505, 4.770021),
    ("O9", 3, 1.492063, 7.768466, 7.963219),
    ("H9a", 2, 1.674440, 8.688828, 7.773735),
    ("H9b", 2, 1.911345, 7.288722, 7.248880),
    ("O10", 3, 4.882766, 1.577103, 1.583534),
    ("H10a", 2, 5.036316, 2.054809, 2.398672),
    ("H10b", 2, 5.749213, 1.266952, 1.320269),
    ("O11", 3, 4.489933, 1.682213, 4.857592),
    ("H11a", 2, 5.390301, 1.387117, 4.993562),
    ("H11b", 2, 4.578482, 2.531730, 4.425490),
    ("O12", 3, 4.723016, 1.412829, 7.844771),
    ("H12a", 2, 4.203360, 0.610887, 7.900259),
    ("H12b", 2, 4.082644, 2.102022, 7.668225),
    ("O13", 3, 4.488430, 4.867475, 1.486109),
    ("H13a", 2, 5.280981, 5.352558, 1.256352),
    ("H13b", 2, 3.895758, 5.530212, 1.840725),
    ("O14", 3, 4.615249, 4.831654, 4.694814),
    ("H14a", 2, 4.330552, 4.143636, 5.296321),
    ("H14b", 2, 5.496000, 4.568075, 4.428290),
    ("O15", 3, 4.849612, 4.699421, 7.660212),
    ("H15a", 2, 4.427805, 5.438277, 8.098850),
    ("H15b", 2, 5.470137, 4.359902, 8.305121),
    ("O16", 3, 4.766386, 7.672967, 1.673291),
    ("H16a", 2, 5.192567, 8.374642, 2.165484),
    ("H16b", 2, 4.250226, 7.200794, 2.326638),
    ("O17", 3, 4.861168, 7.917199, 4.649686),
    ("H17a", 2, 4.882542, 8.141781, 5.579921),
    ("H17b", 2, 5.367488, 7.107325, 4.586632),
    ("O18", 3, 4.410784, 8.012743, 7.878632),
    ("H18a", 2, 4.791777, 7.239467, 8.294709),
    ("H18b", 2, 3.476038, 7.816079, 7.816961),
    ("O19", 3, 7.595160, 1.522324, 1.345474),
    ("H19a", 2, 7.534284, 1.920474, 0.477141),
    ("H19b", 2, 7.821409, 0.608017, 1.174930),
    ("O20", 3, 7.964976, 1.716760, 4.533577),
    ("H20a", 2, 8.854961, 1.924634, 4.249067),
    ("H20b", 2, 8.049238, 0.882293, 4.994876),
    ("O21", 3, 7.895656, 1.524317, 7.816120),
    ("H21a", 2, 8.352842, 1.469713, 6.976935),
    ("H21b", 2, 8.230022, 2.325183, 8.219908),
    ("O22", 3, 7.984441, 4.597760, 1.617903),
    ("H22a", 2, 8.459087, 3.785463, 1.794304),
    ("H22b", 2, 8.615464, 5.148703, 1.154758),
    ("O23", 3, 7.743218, 4.430976, 4.789370),
    ("H23a", 2, 7.749589, 5.007165, 5.553698),
    ("H23b", 2, 7.701227, 5.025906, 4.040686),
    ("O24", 3, 7.652503, 4.515254, 7.869291),
    ("H24a", 2, 8.445246, 4.191506, 8.297049),
    ("H24b", 2, 7.045227, 4.688295, 8.588668),
    ("O25", 3, 7.860639, 7.720413, 1.471078),
    ("H25a", 2, 8.308119, 7.829419, 0.631964),
    ("H25b", 2, 7.144543, 8.355062, 1.445405),
    ("O26", 3, 7.688751, 7.919718, 4.588244),
    ("H26a", 2, 8.471468, 7.498210, 4.233397),
    ("H26b", 2, 7.008485, 7.750115, 3.936550),
    ("O27", 3, 7.629571, 7.589109, 7.919317),
    ("H27a", 2, 6.756489, 7.766071, 8.269523),
    ("H27b", 2, 7.880892, 6.753262, 8.312290),
)

#: Cubic cell edge of the water deck in Å (27 waters at ~1 g/cm³).
_WATER_BOX_ANGSTROM = 9.318

# `comm_modify cutoff` is substituted per model (T * cutoff + skin), which is why
# it is a constructor argument of the deck rather than a literal here: the value
# is a property of the EXPORT, and hard-coding one would either over-communicate
# for shallow models or, far worse, under-communicate for deep ones — the pair
# style would then refuse the run, which is the fail-closed behaviour, but the
# gate would look broken rather than the deck.
_PERIODIC_INPUT_TEMPLATE = """\
# molnex periodic water parity deck - generated by scripts/lammps_iface_parity.py
# atom_style atomic => molecular == 0 => no special-bonds pairs exist, so this
# deck exercises none of CutoffGraphImpl::init's special_bonds interception.
# newton on is REQUIRED by a supports_pbc export: the integrator's global reverse
# communication is the only thing that carries ghost forces to their owners.
units           {units}
atom_style      atomic
boundary        p p p
newton          on
atom_modify     sort 0 0.0

plugin          load {plugin}

read_data       {data_file}

pair_style      molnex {model_dir} {model_name}
pair_coeff      * * {z_map}

neighbor        {skin} bin
neigh_modify    delay 0 every 1 check no
comm_modify     cutoff {comm_cutoff}

thermo          1
thermo_style    custom step pe
thermo_modify   norm no

run             0

print           "{energy_tag} $(pe:%.17g)"
write_dump      all custom {dump_file} id fx fy fz modify sort id format float %.17g
"""


class PeriodicWaterDeck:
    """The embedded periodic deck: a water box under ``p p p``, run once.

    Constructed with the ghost cutoff its ``comm_modify`` line must set, because
    that number belongs to the model (``T * cutoff + skin``) and not to the
    geometry. Everything else — the cell, the coordinates, the atom types — is
    fixed, so two runs of the same export compare like for like.

    No goldens are pinned to this deck. Its fingerprint exists for the
    provenance block only: what this case asserts is agreement between three
    evaluations of one configuration, all produced in the same invocation, so
    there is nothing for a hash to protect against.
    """

    def __init__(self, comm_cutoff: float) -> None:
        self.comm_cutoff = comm_cutoff

    def data_text(self) -> str:
        """Render the LAMMPS data file for the embedded water box."""
        lines = [
            "LAMMPS data file: molnex periodic water parity deck",
            "",
            f"{len(WATER_ATOMS)} atoms",
            f"{len(_TYPE_TO_Z)} atom types",
            "",
        ]
        for axis in "xyz":
            lines.append(f"0.000000 {_WATER_BOX_ANGSTROM:.6f} {axis}lo {axis}hi")
        lines += ["", "Masses", ""]
        for index, mass in enumerate(_TYPE_TO_MASS):
            lines.append(f"{index + 1} {mass:.6f}  # {_TYPE_LABEL[index]}")
        lines += ["", "Atoms # atomic", ""]
        for index, (label, type_id, x, y, z) in enumerate(WATER_ATOMS):
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
        return _PERIODIC_INPUT_TEMPLATE.format(
            units=units,
            plugin=plugin,
            data_file=data_file,
            model_dir=model_dir,
            model_name=model_name,
            z_map=" ".join(str(z) for z in _TYPE_TO_Z),
            skin=f"{_SKIN_ANGSTROM:g}",
            comm_cutoff=f"{self.comm_cutoff:.6f}",
            energy_tag=_ENERGY_TAG,
            dump_file=dump_file,
        )

    def fingerprint(self) -> str:
        """SHA-256 over the deck's data text and its unsubstituted input template."""
        digest = hashlib.sha256()
        digest.update(self.data_text().encode())
        digest.update(_PERIODIC_INPUT_TEMPLATE.encode())
        return digest.hexdigest()


class LammpsRun:
    """One binding of binary + plugin + export + unit style, evaluated on a deck.

    ``launcher`` is the MPI prefix, already resolved to argv — empty for a serial
    run, ``["mpirun", "-np", "2"]`` for a decomposed one. It is a plain prefix
    rather than a rank count because how a site starts an MPI job is not
    something this tool can know (``mpirun -np`` / ``srun -n`` / a wrapper), and
    guessing would fail on the machines that most need the multi-rank leg.
    """

    def __init__(
        self,
        *,
        lammps_bin: Path,
        plugin: Path,
        model_dir: Path,
        model_name: str,
        units: str,
        deck: Deck,
        launcher: tuple[str, ...] = (),
    ) -> None:
        self.lammps_bin = lammps_bin
        self.plugin = plugin
        self.model_dir = model_dir
        self.model_name = model_name
        self.units = units
        self.deck = deck
        self.launcher = launcher

    def execute(self) -> RunOutcome:
        """Run LAMMPS once in a temp directory and read back energy and forces."""
        with tempfile.TemporaryDirectory(prefix="molnex-parity-") as tmp:
            workdir = Path(tmp)
            (workdir / "system.data").write_text(self.deck.data_text())
            (workdir / "in.parity").write_text(
                self.deck.input_text(
                    plugin=self.plugin,
                    model_dir=self.model_dir,
                    model_name=self.model_name,
                    units=self.units,
                    data_file="system.data",
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
        argv = [*self.launcher, str(self.lammps_bin), "-in", "in.parity", "-log", "log.parity"]
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


@dataclass(frozen=True)
class ExportMeta:
    """The export's own description of itself, read from ``<name>.meta.json``.

    The aspirin case needs five strings for its provenance block; the periodic
    case needs the numbers that decide how it must be run — the cutoff and ``T``
    that set ``comm_modify``, the ``energy_conv`` entry that maps model units
    onto the deck's, and the capabilities that say whether this export can serve
    a periodic run at all. One reader for one file, rather than two that could
    disagree about what it says.

    Nothing here re-derives a conversion factor. ``energy_conv`` is the
    pint-evaluated table the export stamped in, and reading it is the only way
    the Python reference and the C++ side are guaranteed to scale by the same
    number (``interface/lammps/README.md``, "Units").
    """

    device: str
    model_dtype: str
    units: str
    contract: str
    execution: str
    supports_pbc: bool
    cutoff: float
    receptive_field_layers: int
    energy_conv: dict[str, float]

    @classmethod
    def from_file(cls, meta: Path) -> ExportMeta:
        """Parse ``meta``; absent strings become ``"unknown"``, absent numbers 0."""
        try:
            document = json.loads(meta.read_text())
        except (OSError, ValueError) as exc:
            raise ParityError(f"could not read {meta}: {exc}") from exc
        block = document.get("lammps", {})
        raw_conv = block.get("energy_conv", {})
        return cls(
            device=str(document.get("device", "unknown")),
            model_dtype=str(block.get("model_dtype", "unknown")),
            units=str(block.get("units", "unknown")),
            contract=str(block.get("contract", "unknown")),
            execution=str(block.get("execution", "unknown")),
            supports_pbc=bool(block.get("supports_pbc", False)),
            cutoff=float(block.get("cutoff", 0.0)),
            receptive_field_layers=int(block.get("receptive_field_layers", 0)),
            energy_conv={str(k): float(v) for k, v in raw_conv.items()},
        )

    def provenance(self) -> dict[str, str]:
        """The five self-description fields both provenance blocks carry."""
        return {
            "device": self.device,
            "precision": self.model_dtype,
            "model_units": self.units,
            "contract": self.contract,
            "execution": self.execution,
        }

    def ghost_cutoff(self) -> float:
        """``T * cutoff + skin`` — what ``comm_modify cutoff`` must be set to.

        The same expression ``CutoffGraphImpl::init`` checks. Stated once, on the
        object that knows ``T`` and the cutoff, so the deck's ``comm_modify``
        line and the Python reference's ghost shell cannot be widened
        independently of each other.
        """
        return self.receptive_field_layers * self.cutoff + _SKIN_ANGSTROM

    def energy_factor(self, lammps_units: str) -> float:
        """One model energy unit expressed in ``lammps_units``."""
        if lammps_units not in self.energy_conv:
            raise ParityError(
                f"the export's energy_conv table has no entry for unit style "
                f"{lammps_units!r} (it serves {sorted(self.energy_conv) or 'nothing'}); "
                "the Python reference would have to invent the conversion the pair style "
                "reads from this table, which is exactly the drift the table exists to "
                "prevent"
            )
        return self.energy_conv[lammps_units]

    def require_periodic(self) -> None:
        """Refuse an export that cannot serve the periodic case, saying why."""
        if self.contract != "cutoff-graph":
            raise ParityError(
                f"the periodic case needs a cutoff-graph export; this one declares "
                f"contract {self.contract!r}"
            )
        if not self.supports_pbc:
            raise ParityError(
                "the periodic case needs an export with supports_pbc=true — a "
                "supports_pbc=false export refuses ghost atoms outright, so it cannot be "
                "run on a periodic box at all. Re-export with supports_pbc=True."
            )
        if self.execution != "dynamic":
            raise ParityError(
                f"the periodic case needs execution 'dynamic'; this export declares "
                f"{self.execution!r}. A cuda-graph export fixes N, while the ghost count "
                "varies — the pair style refuses that combination too."
            )
        if self.receptive_field_layers < 1 or self.cutoff <= 0.0:
            raise ParityError(
                "the export meta lacks a usable cutoff / receptive_field_layers "
                f"(cutoff={self.cutoff!r}, receptive_field_layers="
                f"{self.receptive_field_layers!r}); it is not a schema v3 export"
            )


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of a deck, whatever produced it.

    Attributes:
        label: How this evaluation was produced, for the verdict lines.
        energy: Total potential energy in the deck's LAMMPS unit style.
        forces: Per-atom forces ``(N, 3)`` on the REAL atoms, ordered by atom id
            (ghost contributions already folded onto their owners).
    """

    label: str
    energy: float
    forces: tuple[tuple[float, float, float], ...]

    @classmethod
    def of_run(cls, label: str, outcome: RunOutcome) -> Evaluation:
        """Wrap a LAMMPS run outcome as an evaluation."""
        return cls(label, outcome.energy, outcome.forces)


class PeriodicReference:
    """Direct in-process evaluation of the periodic deck — the live reference.

    This is the only part of the tool that imports torch, and it does so inside
    :meth:`evaluate` so that the aspirin case stays standard-library only.

    It reproduces what a single LAMMPS rank does, in Python, from first
    principles:

    1. build every periodic image of every atom that falls within the ghost
       shell ``T * cutoff + skin`` of the cell — the same width the deck's
       ``comm_modify cutoff`` sets, so this rank's copy of the system is the
       same one LAMMPS assembles;
    2. build the cutoff graph over locals + images with **plain** displacements,
       exactly as the C++ side does: the images already carry their shift, so no
       minimum-image convention is applied anywhere and the model differentiates
       the same coordinates LAMMPS hands it;
    3. evaluate the export with ``local_mask`` selecting the real atoms, so the
       energy is ``Σ_{i∈local} E_i``;
    4. fold each image's force onto the atom it is a copy of — which is what the
       integrator's reverse communication does, expressed as a sum.

    Why a ghost shell rather than a minimum-image graph over N atoms: the model
    recomputes ``edge_diff`` from the positions it is given. Handing it N wrapped
    atoms and a minimum-image edge list would make it differentiate displacement
    vectors that do not match the coordinates — correct distances, wrong
    gradients. Explicit images are the same trick LAMMPS uses, for the same
    reason.

    A ghost shell WIDER than ``T * cutoff`` would give the same answer (sites
    beyond the receptive field cannot reach a local one), which is what makes
    this a legitimate reference for a run whose shell also carries the skin.
    """

    def __init__(self, *, meta: ExportMeta, package: Path, lammps_units: str) -> None:
        self.meta = meta
        self.package = package
        self.lammps_units = lammps_units
        self.ghost_count = 0

    def evaluate(self) -> Evaluation:
        """Run the exported model once over locals + ghost images."""
        _require_torch()
        import torch

        from molix.export import Exporter

        device = self.meta.device if self.meta.device in ("cpu", "cuda") else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ParityError(
                "the export was produced for CUDA and this machine has none; an AOTI "
                "package is device-specific, so the Python reference cannot run it on the "
                "CPU. Run the gate on the export's device, or re-export for CPU."
            )
        dtype = getattr(torch, self.meta.model_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ParityError(
                f"the export declares model_dtype {self.meta.model_dtype!r}, which is not a "
                "torch dtype"
            )

        local_pos = torch.tensor(
            [[atom[2], atom[3], atom[4]] for atom in WATER_ATOMS], dtype=torch.float64
        )
        local_z = torch.tensor([_TYPE_TO_Z[atom[1] - 1] for atom in WATER_ATOMS], dtype=torch.int64)
        pos, owner = self._with_images(local_pos)
        self.ghost_count = int(pos.shape[0] - local_pos.shape[0])

        n_local = local_pos.shape[0]
        z = local_z[owner]
        local_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
        local_mask[:n_local] = True
        edge_index = self._edges(pos)

        runner = Exporter.load(self.package)
        outputs = runner(
            z.to(device),
            pos.to(device=device, dtype=dtype),
            edge_index.to(device),
            local_mask.to(device),
        )
        if not isinstance(outputs, tuple | list) or len(outputs) < 3:
            raise ParityError(
                "the exported package did not return the schema-v3 triple "
                "(energy, forces, atomic_energy); this gate cannot judge a v2 export"
            )
        energy = float(outputs[0].double().sum().item())
        forces = outputs[1].double().cpu()

        # The reverse communication, as a sum: every image's force belongs to the
        # atom it copies. index_add_ over `owner` is that statement — locals map
        # to themselves, so the first n_local rows land where they already were.
        folded = torch.zeros((n_local, 3), dtype=torch.float64)
        folded.index_add_(0, owner, forces)

        factor = self.meta.energy_factor(self.lammps_units)
        return Evaluation(
            "python",
            energy * factor,
            tuple((float(row[0]), float(row[1]), float(row[2])) for row in folded * factor),
        )

    def _with_images(self, local_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positions of locals + ghost images, and each row's owning atom."""
        import torch

        shell = self.meta.ghost_cutoff()
        box = _WATER_BOX_ANGSTROM
        reach = ceil(shell / box)
        n_local = int(local_pos.shape[0])
        estimate = n_local * (2 * reach + 1) ** 3
        if estimate > _MAX_REFERENCE_ATOMS:
            raise ParityError(
                f"this export needs a ghost shell of {shell:.2f} A "
                f"(T={self.meta.receptive_field_layers} * cutoff={self.meta.cutoff:.2f} + "
                f"skin={_SKIN_ANGSTROM:g}) around a {box:.3f} A cell, i.e. up to "
                f"{estimate} atoms in the reference evaluation — over the "
                f"{_MAX_REFERENCE_ATOMS} guard. The embedded deck is too small for a model "
                "with this receptive field; the shell is NOT truncated to fit, because a "
                "truncated shell is a wrong reference rather than a slow one"
            )
        blocks = [local_pos]
        owners = [torch.arange(n_local, dtype=torch.int64)]
        span = range(-reach, reach + 1)
        for offset in product(span, repeat=3):
            if offset == (0, 0, 0):
                continue
            shift = torch.tensor([o * box for o in offset], dtype=torch.float64)
            shifted = local_pos + shift
            # Keep only images inside the shell around the cell — LAMMPS' own
            # ghost criterion for a single rank, whose subdomain IS the cell.
            inside = ((shifted > -shell) & (shifted < box + shell)).all(dim=1)
            if not bool(inside.any()):
                continue
            blocks.append(shifted[inside])
            owners.append(torch.arange(n_local, dtype=torch.int64)[inside])
        return torch.cat(blocks, dim=0), torch.cat(owners, dim=0)

    def _edges(self, pos: torch.Tensor) -> torch.Tensor:
        """Full bidirectional cutoff graph over ``pos``, strictly inside the cutoff.

        ``d2 < cutoff^2`` — strict, matching ``graph_bridge.cpp``'s refilter, so a
        pair sitting exactly on the cutoff is included or excluded identically on
        both sides. Chunked over rows because the extended system is large enough
        that a full pairwise matrix is not.

        The rows come out in ``nonzero()`` order, i.e. sorted by ``(source,
        target)``, while the C++ side emits raw neighbour-list order for an export
        that does not require sorting. That difference is deliberate: agreement
        between the two IS the full-scale edge-order-invariance evidence.
        """
        import torch

        cutsq = self.meta.cutoff * self.meta.cutoff
        n = int(pos.shape[0])
        # 1 << 22 elements bounds the O(chunk*n) pairwise delta tensor's memory per chunk.
        chunk = max(1, (1 << 22) // max(n, 1))
        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            delta = pos[start:stop, None, :] - pos[None, :, :]
            d2 = (delta * delta).sum(dim=2)
            hit = d2 < cutsq
            hit[torch.arange(stop - start), torch.arange(start, stop)] = False
            rows, cols = hit.nonzero(as_tuple=True)
            sources.append(rows + start)
            targets.append(cols)
        edge = torch.stack([torch.cat(sources), torch.cat(targets)], dim=1)
        if edge.shape[0] == 0:
            raise ParityError(
                f"the cutoff graph is empty at cutoff {self.meta.cutoff:.2f} A — the export's "
                "cutoff cannot be reconciled with the embedded water box"
            )
        return edge


def _require_torch() -> None:
    """Fail with a precise diagnostic when torch is not importable.

    Called once at the top of the periodic evaluation; every helper below it then
    imports torch plainly, since by then the import is known to work. Keeping the
    check at one entry point is what lets the aspirin case remain
    standard-library only without scattering try/except around.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ParityError(
            "the periodic case evaluates the exported model in-process and therefore needs "
            f"torch, which is not importable ({exc}). The aspirin case does not: run "
            "--case aspirin on a machine without torch."
        ) from exc


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
        **ExportMeta.from_file(meta).provenance(),
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


def _resolve_launcher(explicit: str | None, ranks: int) -> tuple[str, ...]:
    """Resolve the MPI launcher argv for ``ranks``, or refuse loudly.

    A template rather than a rank flag: ``mpirun -np {ranks}`` and
    ``srun -n {ranks}`` are both one string here, and neither needs this tool to
    know which MPI it is talking to. The ``{ranks}`` placeholder is mandatory —
    a template without it would silently launch one rank and the multi-rank leg
    would pass by not being a multi-rank leg.
    """
    template = explicit or os.environ.get("LAMMPS_MPIRUN") or _MPIRUN_DEFAULT
    if "{ranks}" not in template:
        raise ParityError(
            f"the MPI launcher template {template!r} does not contain '{{ranks}}'; it must, "
            "because this tool substitutes the rank count into it. Example: "
            f"--mpirun {_MPIRUN_DEFAULT!r}"
        )
    argv = shlex.split(template.format(ranks=ranks))
    if not argv:
        raise ParityError(f"the MPI launcher template {template!r} is empty")
    resolved = shutil.which(argv[0])
    if not resolved:
        raise ParityError(
            f"MPI launcher {argv[0]!r} is not on PATH. The multi-rank leg is the ONLY "
            "end-to-end evidence that ghost forces are reduced exactly once, so it is not "
            "skipped: install an MPI runtime, or point --mpirun / $LAMMPS_MPIRUN at the "
            "launcher this machine uses (it must contain '{ranks}')."
        )
    return (resolved, *argv[1:])


def _periodic(args: argparse.Namespace) -> int:
    """The live three-way periodic comparison; nothing here is captured."""
    # The aspirin goldens record an export directory, so --model-dir may be
    # omitted once they exist; the periodic case has no goldens of its own and
    # therefore nothing else to fall back to.
    captured_dir = GOLDENS.model_dir if GOLDENS is not None else ""
    if not args.model_dir and not captured_dir:
        raise ParityError(
            "the periodic case needs --model-dir <export_dir>: it has no goldens to take a "
            "default export from"
        )
    model_dir = Path(args.model_dir or captured_dir).expanduser().resolve()
    meta_path, package = _model_files(model_dir, args.model_name)
    meta = ExportMeta.from_file(meta_path)
    meta.require_periodic()

    lammps_bin = _resolve_lammps(args.lammps)
    plugin = _resolve_plugin(args.plugin)
    launcher = _resolve_launcher(args.mpirun, args.ranks)
    deck = PeriodicWaterDeck(meta.ghost_cutoff())

    def run(label: str, prefix: tuple[str, ...]) -> Evaluation:
        return Evaluation.of_run(
            label,
            LammpsRun(
                lammps_bin=lammps_bin,
                plugin=plugin,
                model_dir=model_dir,
                model_name=args.model_name,
                units=args.units,
                deck=deck,
                launcher=prefix,
            ).execute(),
        )

    reference = PeriodicReference(meta=meta, package=package, lammps_units=args.units)
    # Python first: it is the reference, and if the export cannot be evaluated
    # in-process there is no point spending two LAMMPS runs to find that out.
    evaluations = [
        reference.evaluate(),
        run("1 rank", ()),
        run(f"{args.ranks} ranks", launcher),
    ]

    print(
        f"deck      : periodic water, {len(WATER_ATOMS)} atoms in a "
        f"{_WATER_BOX_ANGSTROM:.3f} A cube, {deck.fingerprint()[:16]}"
    )
    print(f"model     : {model_dir} (T={meta.receptive_field_layers}, cutoff={meta.cutoff:.3f} A)")
    print(
        f"comm      : comm_modify cutoff {meta.ghost_cutoff():.6f}  "
        f"(= T * cutoff + skin {_SKIN_ANGSTROM:g})"
    )
    print(f"ghosts    : {reference.ghost_count} periodic images in the Python reference")
    for evaluation in evaluations:
        print(f"energy    : {evaluation.label:<9} {evaluation.energy!r}")

    labels = tuple(atom[0] for atom in WATER_ATOMS)
    comparisons: list[Comparison] = []
    # Every pair, not just each against the reference: naming the two evaluations
    # that disagree is what turns a failure into a diagnosis, and the one-rank
    # versus many-rank pair is the double-reduction check specifically.
    for one, other in ((0, 1), (0, 2), (1, 2)):
        got, ref = evaluations[other], evaluations[one]
        tag = f"{ref.label} vs {got.label}"
        comparisons.append(Comparison.energy(f"E  {tag}", got.energy, ref.energy))
        comparisons.append(Comparison.forces(f"F  {tag}", got.forces, ref.forces, labels))
    for comparison in comparisons:
        print(f"{comparison.label:<26}: {'OK  ' if comparison.ok else 'FAIL'}  {comparison.detail}")

    _print_periodic_provenance(args, meta, lammps_bin, plugin, launcher, reference.ghost_count)
    if all(comparison.ok for comparison in comparisons):
        print("PERIODIC PARITY OK")
        return 0
    print("PERIODIC PARITY FAILED", file=sys.stderr)
    return 1


def _print_periodic_provenance(
    args: argparse.Namespace,
    meta: ExportMeta,
    lammps_bin: Path,
    plugin: Path,
    launcher: tuple[str, ...],
    ghost_atoms: int,
) -> None:
    """Print the block that goes into the README runbook entry.

    Printed rather than stored: a periodic result IS the run just performed, so
    there is no captured value for it to travel with (see the module docstring).
    """
    fields = {
        "system": f"water box, {len(WATER_ATOMS) // 3} molecules, {len(WATER_ATOMS)} atoms",
        "atom_style": "atomic (molecular == 0: no bonds, so no special_bonds pairs)",
        "boundary": "p p p",
        "box": f"{_WATER_BOX_ANGSTROM:.3f} A cubic",
        "ranks": f"1 and {args.ranks}",
        "mpirun": shlex.join(launcher),
        "comm_modify": (
            f"cutoff {meta.ghost_cutoff():.6f} "
            f"(T {meta.receptive_field_layers} * cutoff {meta.cutoff:.3f} + skin "
            f"{_SKIN_ANGSTROM:g})"
        ),
        "receptive_field": str(meta.receptive_field_layers),
        "ghost_atoms": str(ghost_atoms),
        "lammps_binary": str(lammps_bin),
        "plugin": str(plugin),
        "plugin_commit": _plugin_commit(),
        "torch_version": _torch_version(),
        **meta.provenance(),
        "capture_command": shlex.join(sys.argv),
        "date": datetime.now(UTC).date().isoformat(),
        "tolerance": f"rel {RTOL:g}; energy |dE|/|E|, forces max_i |df_i| / max_j |f_j|",
    }
    print("\n# Provenance (paste into the runbook entry in interface/lammps/README.md)")
    for key, value in fields.items():
        print(f"#   {key:<16}: {value}")


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
            "Parity gate for pair_style molnex: an aspirin deck against captured goldens "
            "(lammps-iface-04-facade, ac-013) and a periodic water box as a live "
            "single-rank / multi-rank / Python three-way comparison (lammps-iface-05-pbc, "
            "ac-014). Developer-run: needs a real LAMMPS build plus the molnex plugin, and "
            "the periodic case also needs torch. Never skips."
        ),
        epilog="exit codes: 0 parity OK, 1 parity FAILED, 2 cannot run the check",
    )
    parser.add_argument(
        "--case",
        default="all",
        choices=("all", "aspirin", "periodic"),
        help="which case to run (default: all)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run the aspirin deck and emit a Goldens(...) block instead of verifying",
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
        "--mpirun",
        help=(
            "MPI launcher template containing '{ranks}' for the periodic multi-rank leg "
            f"(default: $LAMMPS_MPIRUN, then {_MPIRUN_DEFAULT!r})"
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=2,
        help="rank count for the periodic multi-rank leg (default: 2, minimum 2)",
    )
    parser.add_argument(
        "--units",
        default="real",
        choices=("real", "metal"),
        help="LAMMPS unit style for the deck (default: real)",
    )
    args = parser.parse_args(argv)
    if args.write_goldens and not args.capture:
        parser.error("--write-goldens is only meaningful with --capture")
    if args.capture and args.case == "periodic":
        parser.error(
            "--capture does not apply to the periodic case: it is a live three-way "
            "comparison and has no literals to capture (see the module docstring)"
        )
    if args.ranks < 2:
        parser.error(
            f"--ranks must be at least 2 (got {args.ranks}): one rank cannot show a "
            "double reduction, which is the whole point of the multi-rank leg"
        )
    return args


def _run_case(name: str, run: Callable[[], int]) -> int:
    """Run one case, turning its refusal into exit code 2 without losing the others.

    Each case is caught SEPARATELY on purpose: with ``--case all``, an aspirin
    run that cannot proceed (no goldens captured yet) must not suppress the
    periodic evidence, which is the one thing ``ac-014`` is waiting for.
    """
    print(f"== {name} ==")
    try:
        return run()
    except ParityError as exc:
        print(f"lammps_iface_parity [{name}]: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.capture:
        try:
            return _capture(args, AspirinDeck())
        except ParityError as exc:
            print(f"lammps_iface_parity: {exc}", file=sys.stderr)
            return 2
    codes = []
    if args.case in ("all", "aspirin"):
        codes.append(_run_case("aspirin (captured goldens)", lambda: _verify(args, AspirinDeck())))
    if args.case in ("all", "periodic"):
        codes.append(
            _run_case(
                "periodic (live three-way: 1 rank / N ranks / python)", lambda: _periodic(args)
            )
        )
    # A case that could not run (2) outranks a real parity failure (1): "the
    # evidence does not exist" is the more urgent statement, and the failure it
    # might be hiding is still printed above.
    return 2 if 2 in codes else max(codes)


# The block below is the only machine-written region of this file. --capture
# --write-goldens replaces everything between the two marker lines and nothing
# else, so a capture shows up as exactly one reviewable diff hunk. Populate it by
# running a capture on a LAMMPS build; leaving it None keeps the gate red, which
# is the intended state until the chain implementer attaches a real run.
# --- GOLDENS BEGIN ---
GOLDENS: Goldens | None = Goldens(
    deck_sha256="6521b1a5464f2c3dcde3d59dbfa0c124565450f7bd6a330bb5c18efadd65fdc4",
    model_sha256="6cabdf804019c1e37c5cdfb4d341d98746f89f83d7dd81a30170786fcf62fb12",
    model_dir="/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/exports/pinet_open",
    model_name="model",
    units="real",
    energy=-4.071392419652929,
    forces=(
        (0.001617516530299541, -0.00039194644115346543, -0.00048092655174468814),  # C1
        (0.0005795073130391902, 0.0026888653598164418, -3.856508204858706e-05),  # C2
        (-0.0005439981843239172, 0.001382104695150323, -0.0007116704422125822),  # C3
        (-0.0018389857860910306, 0.00048540394915417656, -0.00011479657863502026),  # C4
        (-0.0003093452662640833, -0.0018016195730161163, -1.8703653402518157e-05),  # C5
        (0.000451710338197586, -0.001417854860830212, -0.00014731574496826733),  # C6
        (-0.00031501662180477303, -0.00039267428478711444, -0.0005859439429179189),  # H3
        (-0.00037641056298971104, 7.329077323706743e-05, -2.4676567310954035e-05),  # H4
        (-8.166023044405479e-05, -0.0003555020785389975, 6.645064194658579e-06),  # H5
        (-0.0008649193975827742, -0.00010822159433353153, -0.00011149364800500466),  # H6
        (0.0006211607132034546, -0.0008879850682874705, -0.0011129938076485286),  # C7
        (0.001156131974947077, -0.0009277201638423037, -0.0008841326450261144),  # O1
        (0.003303322400884454, 0.0007228575003867649, 0.0028379079912723955),  # O2
        (-0.0021204357398143104, -0.0003788848633171089, -0.0015402913139857112),  # HO
        (-0.0029226374114664395, -0.002851489515290777, 0.0026153909776558994),  # O3
        (-0.0006369258257633481, -0.0006590335618036039, -0.0009040269469598114),  # C8
        (-0.0006698696530169741, -0.0005274118715940003, 0.0016031119776230205),  # O4
        (-0.00029660401987812595, -0.0004087260505031768, -3.684440638783914e-05),  # C9
        (0.0008060123547453404, 0.0014320442797168822, 0.0008443636330927344),  # HM1
        (0.000702069039599166, 0.0024046902502433253, -0.0005941951418595943),  # HM2
        (0.001739378034523734, 0.0019198131195928994, -0.0006008431707255696),  # HM3
    ),
    provenance={
        "system": "aspirin (acetylsalicylic acid), C9H8O4, 21 atoms",
        "atom_style": "atomic (molecular == 0: no bonds, so no special_bonds pairs)",
        "boundary": "f f f",
        "lammps_version": "22 Jul 2025",
        "lammps_binary": "/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/build-lammps/lmp",  # noqa: E501
        "plugin": "/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/build-plugin/aarch64/molnexplugin.so",  # noqa: E501
        "plugin_commit": "a6b0a03143e45db195a31278980332c367246ccd-dirty",
        "torch_version": "2.13.0",
        "device": "cuda",
        "precision": "float64",
        "model_units": "real",
        "contract": "cutoff-graph",
        "execution": "dynamic",
        "capture_command": "scripts/lammps_iface_parity.py --capture --write-goldens --case aspirin --model-dir /nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/exports/pinet_open --lammps /nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/build-lammps/lmp --plugin /nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200/build-plugin/aarch64/molnexplugin.so",  # noqa: E501
        "date": "2026-08-30",
        "tolerance": "rel 1e-06; energy |dE|/|E|, forces max_i |df_i| / max_j |f_j|",
    },
)
# --- GOLDENS END ---


if __name__ == "__main__":
    raise SystemExit(main())
