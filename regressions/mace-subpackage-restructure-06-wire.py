"""Public import surface and parameter inventory for chain step 06-wire.

Chain step 06 flips `molzoo` over to the `molzoo.mace` sub-package: it deletes
the flat `molzoo/mace_matpes.py` and `molzoo/mace_omol.py`, turns
`MACEMatpes` / `MACEOMol` / `load_matpes_state_dict` / `load_omol_state_dict`
into thin aliases over `molzoo.mace.variants`, and puts the top level on **one**
PEP 562 lazy policy so that `import molzoo` no longer drags in the
cuEquivariance stack. Everything a user can see is supposed to be unchanged
except that one import cost. This file is the standalone check of that claim
(spec `.claude/specs/mace-subpackage-restructure-06-wire.md`, ac-009).

Scenario (public API only), in one process, in this order — the order *is* part
of the test:

1. **Lazy policy.** The script is itself the clean interpreter the policy has to
   be measured in, so the probe runs before anything else pulls torch: after
   `import molzoo`, neither `torch` nor `cuequivariance_torch` may be in
   `sys.modules`; after `import molzoo.mace` (whose config models are imported
   eagerly) both must *still* be absent; `from molix import config` then brings
   torch but **not** cuEquivariance, which is what makes step 4 attributable;
   touching `molzoo.MACE` finally pulls the equivariance stack in. A unit test
   needs a subprocess for this (`tests/test_molzoo/test_imports.py`) because a
   pytest session has imported cuEquivariance long before collection; here the
   process is fresh by construction. It is also asserted to *be* fresh, so
   importing this file from an already-warm interpreter fails loudly instead of
   passing vacuously.
2. **Import surface.** The ten top-level names and the whole
   `molzoo.mace.__all__` resolve; the top-level aliases are the *same objects*
   as the sub-package's, which is what "thin alias" has to mean; `dir()` on both
   modules reports exactly `sorted(__all__)`, and an unknown name still raises
   `AttributeError` rather than an `ImportError` from inside the package. The
   `set(__all__) <= set(dir())` subset form of the ac-001 gate is asserted
   verbatim alongside the stronger equality that in fact holds.
3. **Parameter inventory.** Tiny MatPES and OMol variants are built through the
   keyword surface the scripts bind and their parameter tensor count and element
   total are compared to the hard-coded pre-cutover baselines below. The same
   two models are then built through the `MACEPotential(spec)` form and required
   to agree name-for-name and shape-for-shape with the keyword form.

Why a parameter inventory and not an energy. Energies are the natural golden but
the wrong one here: this restructure reorders module construction, which
reorders the draws from torch's global RNG, so a freshly initialised model's
energy moves for reasons that have nothing to do with correctness. The
parameter inventory is invariant to that and is the offline-checkable proxy for
the claim that matters — an unchanged tensor inventory is exactly the condition
under which the official MatPES / OMol checkpoints still load strictly, and the
strict load is what carries the external numerical verdicts (mace-omol-port-01
ac-006, 7e-7 eV / 4.3e-6 eV·A) across the cutover. Re-running those verdicts is
impossible in-tree by policy: mace-torch and e3nn are not importable here.

Goldens
-------
Integers, so there is no floating-point tolerance anywhere in this file — every
comparison below is exact equality.

    capture command : git worktree add --detach <tmp> 0e05959 && \\
                      PYTHONPATH=<tmp>/src python -c '
                        import torch
                        from molix import config
                        config.set_precision("fp64")
                        from molzoo.mace_matpes import MACEMatpes
                        from molzoo.mace_omol import MACEOMol
                        torch.manual_seed(0)
                        m = MACEMatpes(atomic_numbers=[1, 6, 8],
                                       atomic_energies=torch.tensor([-13.6, -1029.0, -2041.0]),
                                       **TINY_MATPES_KWARGS)
                        print(len(list(m.parameters())),
                              sum(p.numel() for p in m.parameters()))'
                      (and the same for MACEOMol / TINY_OMOL_KWARGS)
    commit          : 0e05959 (0e059591b923833986344a936e60941588588de7) — the
                      last commit at which the flat `molzoo/mace_matpes.py` and
                      `molzoo/mace_omol.py` still existed. The 06-wire working
                      tree that deletes them is uncommitted on top of it.
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-09
    device / dtype  : CPU, float64 (`config.set_precision("fp64")`),
                      `use_fallback=True` for MatPES; MACEOMol takes no such
                      keyword and its spec defaults to the fused path.
    oracle          : this repository at the pre-cutover commit. No mace-torch,
                      no e3nn, no ASE, no network, no subprocess. The dtype and
                      the fallback switch do not enter the counts; they are
                      pinned so the capture is reproducible, not because the
                      numbers depend on them.
    observed        : the flat pre-cutover classes and the post-cutover
                      `molzoo.mace.variants` classes produce not merely the same
                      two totals but the identical `named_parameters()` list —
                      same names, same shapes, same order — for both variants,
                      and the `MACEPotential(spec)` form matches both. Only the
                      two totals are pinned as literals here; the keyword-vs-spec
                      list equality is re-checked live (`check_construction_paths`)
                      since it needs no golden.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-06-wire.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lazy-policy probe. Nothing above this may import torch, so the imports below
# are deliberately interleaved with the snapshots they are being measured by --
# moving any of them breaks the measurement rather than merely reordering it.
# ---------------------------------------------------------------------------
import sys

#: `sys.modules` before this file touched anything. Recorded so that running
#: the file from a warm interpreter (an `import` rather than `python <file>`)
#: is reported as a broken measurement instead of a silent pass.
FRESH_INTERPRETER = "torch" not in sys.modules and "molzoo" not in sys.modules

import molzoo  # noqa: E402

#: Third-party module whose presence marks "the equivariance stack is loaded".
#: The expensive import the lazy policy exists to defer; same probe as
#: `tests/test_molzoo/test_imports.py`.
CUEQ = "cuequivariance_torch"

#: Snapshot after `import molzoo`: the top level must cost `typing` and nothing
#: else, so torch has not been imported either.
AT_MOLZOO = {"torch": "torch" in sys.modules, "cueq": CUEQ in sys.modules}

import molzoo.mace  # noqa: E402

#: Snapshot after `import molzoo.mace`. This one imports `molzoo.mace.spec`
#: eagerly; the sub-package docstring's claim is that the config models are
#: torch-free, so even torch must still be absent here.
AT_MOLZOO_MACE = {"torch": "torch" in sys.modules, "cueq": CUEQ in sys.modules}

from molix import config  # noqa: E402

config.set_precision("fp64")  # before any module construction: cuEq bakes in dtype

#: Snapshot after the `molix` import that the rest of the file needs. torch is
#: now loaded and cuEquivariance still is not -- without this leg, "cuEq
#: appeared when we touched `molzoo.MACE`" would rest on the assumption that
#: nothing in between could have brought it.
AT_MOLIX = {"torch": "torch" in sys.modules, "cueq": CUEQ in sys.modules}

#: The lazy resolution itself. `molzoo.__getattr__` does not cache onto the
#: module, so this is an honest first touch; it is kept for the identity check
#: in `check_alias_identity`.
LAZILY_RESOLVED_MACE = molzoo.MACE

#: Snapshot after the first model-symbol access: lazy, not absent.
AFTER_ATTRIBUTE_ACCESS = {"torch": "torch" in sys.modules, "cueq": CUEQ in sys.modules}

import torch  # noqa: E402

from molzoo import (  # noqa: E402
    MACE,
    Allegro,
    AllegroSpec,
    MACEMatpes,
    MACEOMol,
    MACESpec,
    PiNet,
    PiNetSpec,
    load_matpes_state_dict,
    load_omol_state_dict,
)
from molzoo.mace import MACE as MACE_FROM_SUBPACKAGE  # noqa: E402
from molzoo.mace import (  # noqa: E402
    CheckpointRemap,
    MACEMatpesSpec,
    MACEOMolSpec,
    MACEPotential,
)

# ---------------------------------------------------------------------------
# Configurations. Literal copies of `TINY_MATPES_KWARGS` / `TINY_OMOL_KWARGS`
# in `tests/test_molzoo/test_mace/conftest.py` -- copied rather than imported,
# because a regression script must not depend on the test suite and because a
# golden whose configuration can be edited elsewhere is not a golden.
# ---------------------------------------------------------------------------

#: Element table (z-table), strictly ascending as `torch.searchsorted` needs.
ATOMIC_NUMBERS: list[int] = [1, 6, 8]

#: Per-element reference energies `E0` in eV/atom, in `ATOMIC_NUMBERS` order.
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]

#: Tiny MACE-MatPES hyper-parameters (l_max=1, 16 channels): structurally the
#: shipped model, small enough to build in a second on CPU.
TINY_MATPES_KWARGS: dict[str, object] = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 1,
    "num_features": 16,
    "max_hidden_l": 1,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "radial_mlp": [8],
    "use_fallback": True,  # CPU: the fused kernels need a GPU + the ops wheel
}

#: Tiny MACE-OMol hyper-parameters (l_max=1, 16 channels). `use_fallback` is
#: absent on purpose: `MACEOMol` takes no such keyword, so the spec's default
#: (`False`, the fused path) is what both construction paths must agree on.
TINY_OMOL_KWARGS: dict[str, object] = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 1,
    "num_features": 16,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "edge_channels": 8,
}

#: Construction seed. The counts below are seed-independent -- initialisation
#: fills tensors, it does not decide how many there are -- so this is hygiene,
#: not a load-bearing golden input.
SEED = 0

# ---------------------------------------------------------------------------
# Goldens -- see the module docstring for provenance.
# ---------------------------------------------------------------------------

#: `len(list(m.parameters()))` and `sum(p.numel() for p in m.parameters())` for
#: `MACEMatpes(**TINY_MATPES_KWARGS)`, captured from the flat pre-cutover
#: `molzoo.mace_matpes` at commit 0e05959.
MATPES_PARAM_TENSORS = 21
MATPES_PARAM_ELEMENTS = 6852

#: The same two totals for `MACEOMol(**TINY_OMOL_KWARGS)`, captured from the
#: flat pre-cutover `molzoo.mace_omol` at the same commit. Larger than MatPES
#: because of the charge/spin conditioning and the per-l edge channels.
OMOL_PARAM_TENSORS = 73
OMOL_PARAM_ELEMENTS = 16795

#: The names `scripts/matpes_port/run_nve.py:45`,
#: `benchmarks/bench_mace_matpes.py:38` and `benchmarks/bench_trainer_throughput.py:26`
#: bind, plus the Allegro / PiNet pair that joined the same lazy table in 06.
#: Compared against `molzoo.__all__` as a set, so a name silently dropped from
#: the export list fails here even though the `from molzoo import ...` above
#: would still have succeeded through `__getattr__`.
EXPECTED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "Allegro",
        "AllegroSpec",
        "MACE",
        "MACEMatpes",
        "MACEOMol",
        "MACESpec",
        "PiNet",
        "PiNetSpec",
        "load_matpes_state_dict",
        "load_omol_state_dict",
    }
)

#: The six names ac-001 requires `molzoo.mace.__all__` to contain *at least*.
#: The rest of that list is checked for resolvability, not for membership, so
#: adding a symbol to the sub-package does not fail this file.
REQUIRED_SUBPACKAGE_NAMES: frozenset[str] = frozenset(
    {
        "MACE",
        "MACESpec",
        "MACEMatpes",
        "MACEOMol",
        "load_matpes_state_dict",
        "load_omol_state_dict",
    }
)

#: A name in neither lazy table: the probe for "the export tables are closed".
UNKNOWN_NAME = "NotAThing"


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def count(self, name: str, got: int, want: int) -> None:
        """Assert an integer golden exactly -- no tolerance applies to a count."""
        if got != want:
            self.failures.append(f"{name}: got {got}, want {want}")
        print(f"  {name:<46} {got:>6}  {'ok' if got == want else 'FAILED'}")

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (a name resolves, a module is absent, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        print(f"  {name:<46} {'ok' if holds else 'FAILED':>6}")


def check_lazy_policy(checker: Checker) -> None:
    """The four `sys.modules` snapshots taken while this file was importing.

    The whole point of the ladder is attribution: each snapshot narrows what
    could have loaded the equivariance stack, so the final `cueq is present`
    can only be blamed on the `molzoo.MACE` access.

    Args:
        checker: Failure collector.
    """
    print("Lazy policy (snapshots taken during this file's own import)")
    checker.truth(
        "lazy.interpreter_was_fresh",
        FRESH_INTERPRETER,
        "torch or molzoo was already imported -- run this file as the entry "
        "point (`python regressions/...py`), not as an imported module; the "
        "snapshots below measure nothing otherwise",
    )
    checker.truth(
        "lazy.import_molzoo_is_torch_free",
        AT_MOLZOO["torch"] is False,
        "`import molzoo` pulled torch -- the top level is supposed to import nothing but `typing`",
    )
    checker.truth(
        "lazy.import_molzoo_is_cueq_free",
        AT_MOLZOO["cueq"] is False,
        f"`import molzoo` pulled {CUEQ} -- the lazy policy is not in force",
    )
    checker.truth(
        "lazy.import_molzoo_mace_is_torch_free",
        AT_MOLZOO_MACE["torch"] is False,
        "`import molzoo.mace` pulled torch -- its eagerly imported config "
        "models are documented as torch-free",
    )
    checker.truth(
        "lazy.import_molzoo_mace_is_cueq_free",
        AT_MOLZOO_MACE["cueq"] is False,
        f"`import molzoo.mace` pulled {CUEQ} -- the sub-package re-export "
        "surface must stay lazy too",
    )
    checker.truth(
        "lazy.molix_brings_torch_not_cueq",
        AT_MOLIX["torch"] is True and AT_MOLIX["cueq"] is False,
        f"expected torch and no {CUEQ} after `from molix import config`, got "
        f"{AT_MOLIX} -- the next leg cannot attribute the stack to the "
        "attribute access",
    )
    checker.truth(
        "lazy.attribute_access_brings_cueq",
        AFTER_ATTRIBUTE_ACCESS["cueq"] is True,
        f"`molzoo.MACE` did not pull {CUEQ} -- lazy has become absent, and the "
        "symbol is resolving to something other than the real encoder",
    )
    checker.truth(
        "lazy.resolved_symbol_is_a_class",
        isinstance(LAZILY_RESOLVED_MACE, type),
        f"`molzoo.MACE` resolved to {LAZILY_RESOLVED_MACE!r}, not a class",
    )


def check_import_surface(checker: Checker) -> None:
    """Both export tables: complete, closed, and reported by `dir()`.

    Args:
        checker: Failure collector.
    """
    print("\nImport surface (molzoo and molzoo.mace export tables)")
    checker.truth(
        "surface.top_level_all_is_the_expected_set",
        set(molzoo.__all__) == EXPECTED_TOP_LEVEL,
        f"molzoo.__all__ is {sorted(molzoo.__all__)}, expected {sorted(EXPECTED_TOP_LEVEL)}",
    )
    unresolved_top = [name for name in molzoo.__all__ if getattr(molzoo, name, None) is None]
    checker.truth(
        "surface.every_top_level_name_resolves",
        not unresolved_top,
        f"lazy table advertises names that do not resolve: {unresolved_top}",
    )
    checker.truth(
        "surface.top_level_symbols_are_bound",
        all(
            symbol is not None
            for symbol in (
                MACE,
                MACESpec,
                MACEMatpes,
                MACEOMol,
                load_matpes_state_dict,
                load_omol_state_dict,
                Allegro,
                AllegroSpec,
                PiNet,
                PiNetSpec,
            )
        ),
        "one of the ten `from molzoo import ...` bindings is None",
    )
    checker.truth(
        "surface.subpackage_all_covers_required",
        REQUIRED_SUBPACKAGE_NAMES <= set(molzoo.mace.__all__),
        f"molzoo.mace.__all__ is missing "
        f"{sorted(REQUIRED_SUBPACKAGE_NAMES - set(molzoo.mace.__all__))}",
    )
    unresolved_sub = [
        name for name in molzoo.mace.__all__ if getattr(molzoo.mace, name, None) is None
    ]
    checker.truth(
        "surface.every_subpackage_name_resolves",
        not unresolved_sub,
        f"molzoo.mace advertises names that do not resolve: {unresolved_sub}",
    )
    checker.truth(
        "surface.subpackage_classes_are_bound",
        isinstance(MACE_FROM_SUBPACKAGE, type)
        and isinstance(MACEPotential, type)
        and isinstance(CheckpointRemap, type)
        and isinstance(MACEMatpesSpec, type)
        and isinstance(MACEOMolSpec, type),
        "`from molzoo.mace import MACE, MACEPotential, CheckpointRemap, "
        "MACEMatpesSpec, MACEOMolSpec` did not all bind classes",
    )

    # ac-001's gate is the subset form; the equality is what actually holds,
    # because both `__dir__`s return `sorted(__all__)`. Pinning the stronger
    # one keeps `dir()` and `import *` from drifting apart in either direction.
    checker.truth(
        "surface.ac001_subpackage_all_within_dir",
        set(molzoo.mace.__all__) <= set(dir(molzoo.mace)),
        f"dir(molzoo.mace) is missing {sorted(set(molzoo.mace.__all__) - set(dir(molzoo.mace)))}",
    )
    checker.truth(
        "surface.top_level_dir_is_sorted_all",
        dir(molzoo) == sorted(molzoo.__all__),
        f"dir(molzoo) is {dir(molzoo)}, expected {sorted(molzoo.__all__)}",
    )
    checker.truth(
        "surface.subpackage_dir_is_sorted_all",
        dir(molzoo.mace) == sorted(molzoo.mace.__all__),
        f"dir(molzoo.mace) is {dir(molzoo.mace)}, expected {sorted(molzoo.mace.__all__)}",
    )

    for module in (molzoo, molzoo.mace):
        raised = False
        try:
            getattr(module, UNKNOWN_NAME)
        except AttributeError:
            raised = True
        checker.truth(
            f"surface.{module.__name__}_table_is_closed",
            raised,
            f"{module.__name__}.{UNKNOWN_NAME} did not raise AttributeError -- a "
            "typo must fail as a missing attribute, not as an ImportError from "
            "inside the package",
        )


def check_alias_identity(checker: Checker) -> None:
    """The top-level names are the sub-package's objects, not copies of them.

    "Thin alias" is only meaningful as an identity claim: a second definition
    that happened to behave the same would satisfy every other check in this
    file and still be the duplication 06-wire exists to remove.

    Args:
        checker: Failure collector.
    """
    print("\nAlias identity (top level is molzoo.mace, not a second definition)")
    for name, top_level in (
        ("MACE", MACE),
        ("MACESpec", MACESpec),
        ("MACEMatpes", MACEMatpes),
        ("MACEOMol", MACEOMol),
        ("load_matpes_state_dict", load_matpes_state_dict),
        ("load_omol_state_dict", load_omol_state_dict),
    ):
        checker.truth(
            f"alias.{name}",
            top_level is getattr(molzoo.mace, name),
            f"molzoo.{name} is not molzoo.mace.{name}",
        )
    checker.truth(
        "alias.MACE_binding_matches_attribute",
        MACE is MACE_FROM_SUBPACKAGE is LAZILY_RESOLVED_MACE,
        "`from molzoo import MACE`, `from molzoo.mace import MACE` and "
        "`molzoo.MACE` gave different objects",
    )
    checker.truth(
        "alias.variants_are_potentials",
        issubclass(MACEMatpes, MACEPotential) and issubclass(MACEOMol, MACEPotential),
        "the named variants no longer specialise MACEPotential",
    )
    checker.truth(
        "alias.variant_specs_share_the_base",
        issubclass(MACEMatpesSpec, MACESpec) and issubclass(MACEOMolSpec, MACESpec),
        "MACESpec is no longer the shared foundation-variant configuration base",
    )


def build_variants() -> tuple[MACEMatpes, MACEOMol]:
    """The two tiny foundation models, built through the keyword surface.

    `atomic_energies` is passed as a `torch.Tensor`, the way
    `scripts/matpes_port/run_nve.py:152` passes it, even though the spec holds a
    plain list -- the keyword adapters have to accept both.

    Returns:
        `(matpes, omol)`, fp64 on CPU.
    """
    energies = torch.tensor(ATOMIC_ENERGIES)
    torch.manual_seed(SEED)
    matpes = MACEMatpes(
        atomic_numbers=list(ATOMIC_NUMBERS), atomic_energies=energies, **TINY_MATPES_KWARGS
    )
    torch.manual_seed(SEED)
    omol = MACEOMol(
        atomic_numbers=list(ATOMIC_NUMBERS), atomic_energies=energies, **TINY_OMOL_KWARGS
    )
    return matpes, omol


def build_from_specs() -> tuple[MACEPotential, MACEPotential]:
    """The same two models, built the way new code is told to build them.

    Returns:
        `(matpes, omol)` as plain `MACEPotential`s over the two variant specs.
    """
    torch.manual_seed(SEED)
    matpes = MACEPotential(
        MACEMatpesSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_MATPES_KWARGS
        )
    )
    torch.manual_seed(SEED)
    omol = MACEPotential(
        MACEOMolSpec(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            use_fallback=False,  # MACEOMol takes no such keyword; the spec defaults
            **TINY_OMOL_KWARGS,
        )
    )
    return matpes, omol


def check_parameter_inventory(
    checker: Checker, name: str, model: torch.nn.Module, tensors: int, elements: int
) -> None:
    """One model's parameter tensor count and element total, against the goldens.

    Args:
        checker: Failure collector.
        name: Label for the printed rows.
        model: The constructed variant.
        tensors: Golden `len(list(model.parameters()))`.
        elements: Golden `sum(p.numel() for p in model.parameters())`.
    """
    parameters = list(model.parameters())
    checker.count(f"{name}.param_tensors", len(parameters), tensors)
    checker.count(f"{name}.param_elements", sum(p.numel() for p in parameters), elements)


def check_construction_paths(
    checker: Checker, name: str, keyword: torch.nn.Module, spec: torch.nn.Module
) -> None:
    """The keyword adapter and the spec form build the same module graph.

    Not a golden -- a relation, so it needs no captured literal and cannot rot.
    It is the assertion that `variants.py` really is an adapter: if it re-derived
    a hyper-parameter, an irreps string or an `E0` table of its own (which its
    docstring forbids), the two inventories would part company here.

    Args:
        checker: Failure collector.
        name: Label for the printed rows.
        keyword: Model from the keyword constructor.
        spec: Model from `MACEPotential(spec)`.
    """
    from_keyword = [(key, tuple(value.shape)) for key, value in keyword.named_parameters()]
    from_spec = [(key, tuple(value.shape)) for key, value in spec.named_parameters()]
    differing = [
        f"{left} != {right}" for left, right in zip(from_keyword, from_spec) if left != right
    ]
    checker.truth(
        f"{name}.keyword_and_spec_agree",
        from_keyword == from_spec,
        f"{len(differing)} entr(y/ies) differ: {differing[:5]}; "
        f"keyword has {len(from_keyword)} tensors, spec {len(from_spec)}",
    )


def main() -> int:
    """Check the import surface, the lazy policy and the parameter goldens."""
    checker = Checker()
    check_lazy_policy(checker)
    check_import_surface(checker)
    check_alias_identity(checker)

    print("\nParameter inventory (tiny variants, fp64 CPU, vs pre-cutover goldens)")
    keyword_matpes, keyword_omol = build_variants()
    check_parameter_inventory(
        checker, "matpes", keyword_matpes, MATPES_PARAM_TENSORS, MATPES_PARAM_ELEMENTS
    )
    check_parameter_inventory(
        checker, "omol", keyword_omol, OMOL_PARAM_TENSORS, OMOL_PARAM_ELEMENTS
    )

    print("\nConstruction paths (keyword adapter vs MACEPotential(spec))")
    spec_matpes, spec_omol = build_from_specs()
    check_construction_paths(checker, "matpes", keyword_matpes, spec_matpes)
    check_construction_paths(checker, "omol", keyword_omol, spec_omol)
    check_parameter_inventory(
        checker, "matpes.spec", spec_matpes, MATPES_PARAM_TENSORS, MATPES_PARAM_ELEMENTS
    )
    check_parameter_inventory(
        checker, "omol.spec", spec_omol, OMOL_PARAM_TENSORS, OMOL_PARAM_ELEMENTS
    )

    if checker.failures:
        print("\nFAILED — the molzoo public wiring no longer matches the 06-wire contract:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
