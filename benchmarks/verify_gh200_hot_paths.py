"""GH200 verification of the 2026-10-16 hot-path changes on CUDA (fp64).

The CPU unit tests exercise these paths only through cuEquivariance's naive
fallback; this script runs the same checks on the fused kernels. One
``PASS`` / ``FAIL`` / ``SKIP`` line per check with the measured number; exit
status is non-zero when any check fails.

1. Allegro
   * ``WeightedChannels(basis, weights, index=source, num_rows=N)`` (one
     ``SegmentedPolynomial`` with ``output_indices`` / ``output_shapes``)
     equals ``scatter_sum_compile_safe(WeightedChannels(basis, weights),
     source, N)`` — fp64 and fp32.
   * ``AllegroPotential`` on the tiny spec (CUDA eager vs CPU eager) and on
     the official packages (CUDA eager vs the upstream goldens at the test's
     tolerance; Allegro-OAM-L is an fp32 package with fp32 goldens, gated at
     the recorded fp32-rounding level, notes 2026-10-15).
   * ``Compiler.explain(forward)`` → 1 graph / 0 breaks;
     ``Compiler(cuda_graphs=True)`` under ``torch.no_grad()`` vs eager.
2. MACE (tiny MatPES / OMol / Polar fixtures)
   * CUDA eager vs the CPU reference ``_gh200_verify/step7_ref.pt``.
   * explain → 0 breaks for MatPES / OMol ``forward``, Polar's ``forward``
     on its periodic path and ``MACEEncoder.forward``;
     ``Compiler(cuda_graphs=True)`` vs eager for the same four. Polar's
     non-periodic path is reported as SKIP: its real-space Hartree pair list
     (``molpot/potentials/gto/hartree.py::cached_complete_directed_pairs``)
     reads ``.item()`` for a cache key and masks pairs data-dependently (34
     graphs / 33 breaks, pre-existing; notes 2026-10-17).

Every ``cuda-graphs vs eager`` check also asserts that inductor recorded a
CUDA graph (``cudagraph_trees.get_manager``), so a silently skipped capture
cannot pass as parity.

Run (on a GH200 node, through ``benchmarks/run_gh200_hot_paths.sbatch``)::

    "${PY}" benchmarks/verify_gh200_hot_paths.py

``--device cpu --quick`` runs the eager-only subset on a login node as a
syntax / API smoke; the kernel checks are reported as SKIP there.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import torch

from molix import config

config.set_precision("fp64")

from tests.conftest import make_graph_batch  # noqa: E402
from tests.test_molzoo.test_allegro.conftest import TINY, WEIGHTS_DIR  # noqa: E402
from tests.test_molzoo.test_allegro.test_potential import _golden_batch  # noqa: E402
from tests.test_molzoo.test_mace.conftest import (  # noqa: E402
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    PAIR_BATCH,
    PAIR_POS,
    PAIR_Z,
    SINGLE_GRAPH,
    TINY_MATPES_KWARGS,
    TINY_OMOL_KWARGS,
    TINY_POLAR_KWARGS,
    full_edge_index,
    wake_zero_init_readout,
)
from tests.test_molzoo.test_mace.test_polar import _periodic_cluster  # noqa: E402
from torch._inductor.cudagraph_trees import get_manager  # noqa: E402

from molix.compile import Compiler  # noqa: E402
from molix.F.scatter import scatter_sum_compile_safe  # noqa: E402
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY, VIRIAL_KEY  # noqa: E402
from molzoo.allegro import AllegroPotential, AllegroSpec  # noqa: E402
from molzoo.allegro.layers import WeightedChannels  # noqa: E402
from molzoo.mace.encoder import MACEEncoder  # noqa: E402
from molzoo.mace.polar import MacePolar  # noqa: E402
from molzoo.mace.potential import MaceMatpes, MaceOmol  # noqa: E402
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACEPolarSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
STEP7_REF = HERE / "_gh200_verify" / "step7_ref.pt"

#: Tolerances (abs unless stated).
TOL_WEIGHTED_FP64 = 1e-12
TOL_WEIGHTED_FP32 = 1e-4
TOL_TINY_CUDA_VS_CPU = 1e-10  # the Allegro tests' ATOL
TOL_SMALL_PKG_E = 1e-8  # eV/atom, test_load_matches_the_official_package
TOL_SMALL_PKG_F = 1e-7  # eV/Å (forces and virial)
TOL_OAM_E = 1e-5  # eV/atom: fp32 goldens; recorded fp64-eval parity 5e-7
TOL_OAM_F = 1e-3  # eV/Å: recorded 4e-5
TOL_MACE_REF_REL = 1e-10
TOL_GRAPH_F = 2e-13  # eV/Å, CUDA graphs vs eager
TOL_GRAPH_E = 1e-11  # eV, |E| ~ 1e3-1e4 so 2e-13 is below one ulp
#: Polar's periodic path adds a reciprocal-space Hartree sum to ``graphs.energy``
#: whose terms inductor reorders: per-atom energies replay to 0.0, the k-space
#: term to 2.3e-12 eV and its force gradient to 1.3e-12 eV/A (job 3978557).
TOL_GRAPH_F_KSPACE = 5e-12
#: Why Polar's non-periodic path is not a compile seam here (notes 2026-10-17).
POLAR_NON_PERIODIC = (
    "non-periodic path: hartree.py cached_complete_directed_pairs reads .item() and "
    "masks pairs data-dependently (34 graphs / 33 breaks, pre-existing)"
)


class Report:
    """Collects PASS / FAIL / SKIP lines; ``failed`` drives the exit status."""

    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0
        self.skipped = 0
        self.t0 = time.time()

    def _stamp(self) -> str:
        return f"[{time.time() - self.t0:7.1f}s]"

    def check(self, name: str, value: float, limit: float, unit: str = "") -> bool:
        ok = math.isfinite(value) and value <= limit
        self._tally(ok)
        print(
            f"{'PASS' if ok else 'FAIL'} {self._stamp()} {name}: {value:.3e} <= {limit:.1e} {unit}"
        )
        return ok

    def equal(self, name: str, got: object, want: object, detail: str = "") -> bool:
        ok = got == want
        self._tally(ok)
        extra = f" {detail}" if (detail and not ok) else ""
        print(
            f"{'PASS' if ok else 'FAIL'} {self._stamp()} {name}: got {got!r} want {want!r}{extra}"
        )
        return ok

    def skip(self, name: str, why: str) -> None:
        self.skipped += 1
        print(f"SKIP {self._stamp()} {name}: {why}")

    def error(self, name: str, exc: BaseException) -> None:
        self.failed += 1
        print(f"FAIL {self._stamp()} {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)

    def _tally(self, ok: bool) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1


report = Report()


def guarded(name: str):
    """Run a check function; an exception is one FAIL line, not an abort."""

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            report.error(name, exc)
            return None

    return run


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(b.device) - b.detach()).abs().max())


def outputs(frame, keys) -> dict:
    return {k: frame[k].detach().clone() for k in keys if k in frame.keys(include_nested=True)}


def eager(model, batch, keys):
    with torch.no_grad():
        return outputs(model(batch.clone()), keys)


def explain_one_graph(name: str, target, batch) -> None:
    with torch.no_grad():
        summary = Compiler.explain(target, batch.clone())
    report.equal(
        f"{name} explain (graphs, breaks)",
        (summary.graph_count, summary.graph_break_count),
        (1, 0),
        detail=f"break_sites={summary.break_sites} reasons={summary.break_reasons}",
    )


def cuda_graph_vs_eager(
    name: str, model, batch, *, want_virial: bool = False, tol_f: float = TOL_GRAPH_F
) -> None:
    """``Compiler(cuda_graphs=True)(model)`` under no_grad, replayed, vs eager."""
    keys = [ENERGY_KEY, ATOMIC_ENERGY_KEY, FORCES_KEY] + ([VIRIAL_KEY] if want_virial else [])
    ref = eager(model, batch, keys)
    compiled = Compiler(cuda_graphs=True)(model)
    got = None
    with torch.no_grad():
        for _ in range(3):  # compile, record, replay
            torch.compiler.cudagraph_mark_step_begin()
            got = outputs(compiled(batch.clone()), keys)
    torch.cuda.synchronize()
    assert got is not None
    manager = get_manager(torch.cuda.current_device(), create_if_none_exists=False)
    recorded = manager is not None and any(manager.roots.values())
    report.equal(f"{name} inductor recorded a CUDA graph", recorded, True)
    report.check(
        f"{name} cuda-graphs vs eager |dE|",
        max_abs(got[ENERGY_KEY], ref[ENERGY_KEY]),
        TOL_GRAPH_E,
        "eV",
    )
    report.check(
        f"{name} cuda-graphs vs eager |dE_atom|",
        max_abs(got[ATOMIC_ENERGY_KEY], ref[ATOMIC_ENERGY_KEY]),
        TOL_GRAPH_E,
        "eV",
    )
    report.check(
        f"{name} cuda-graphs vs eager |dF|",
        max_abs(got[FORCES_KEY], ref[FORCES_KEY]),
        tol_f,
        "eV/A",
    )
    if want_virial:
        report.check(
            f"{name} cuda-graphs vs eager |dW|",
            max_abs(got[VIRIAL_KEY], ref[VIRIAL_KEY]),
            TOL_GRAPH_F,
            "eV",
        )


# --------------------------------------------------------------------------- Allegro


def allegro_cluster():
    """The ``cluster`` fixture of tests/test_molzoo/test_allegro/conftest.py."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.4, 0.9, 0.2], [0.2, -0.3, 1.3], [0.9, -0.6, 1.8]],
        dtype=torch.float64,
    )
    pairs = [[i, j] for i in range(5) for j in range(5) if i != j]
    return make_graph_batch(
        pos, torch.tensor([6, 1, 1, 8, 1]), torch.tensor(pairs), torch.zeros(5, dtype=torch.long)
    )


def check_weighted_channels(device: torch.device) -> None:
    """Indexed output of the segmented polynomial == explicit scatter of the per-pair result."""
    E, N = 4096, 128
    for dims, channels in (([1, 3, 5], 4), ([1, 3, 5, 7], 64)):  # tiny spec; OAM-L shape
        for dtype, tol in ((torch.float64, TOL_WEIGHTED_FP64), (torch.float32, TOL_WEIGHTED_FP32)):
            name = f"allegro.WeightedChannels indexed-vs-scatter dims={dims} u={channels} {dtype}"
            try:
                torch.manual_seed(0)
                op = WeightedChannels(irrep_dims=dims, channels=channels, dtype=dtype).to(device)
                basis = torch.randn(E, sum(dims), dtype=dtype, device=device)
                weights = torch.randn(E, channels * len(dims), dtype=dtype, device=device)
                index = torch.randint(0, N, (E,), device=device)
                fused = op(basis, weights, index=index, num_rows=N)
                ref = scatter_sum_compile_safe(op(basis, weights), index, N)
                report.equal(f"{name} shape", tuple(fused.shape), (N, sum(dims) * channels))
                report.check(name, max_abs(fused, ref), tol)
            except Exception as exc:  # noqa: BLE001
                report.error(name, exc)


def check_allegro_models(device: torch.device, *, quick: bool) -> None:
    keys = [ENERGY_KEY, ATOMIC_ENERGY_KEY, FORCES_KEY]
    on_cuda = device.type == "cuda"

    # tiny spec: CUDA eager vs CPU eager (the path the unit tests validated)
    name = "allegro.tiny"
    tiny_model = tiny_batch = None
    try:
        torch.manual_seed(0)
        model = AllegroPotential(AllegroSpec(**TINY), wrt=("atoms.pos",)).eval()
        cluster = allegro_cluster()
        cpu = eager(model, cluster, keys)
        model = model.to(device)
        batch = cluster.to(device)
        gpu = eager(model, batch, keys)
        report.check(
            f"{name} eager {device.type} vs cpu |dE|",
            max_abs(gpu[ENERGY_KEY], cpu[ENERGY_KEY]),
            TOL_TINY_CUDA_VS_CPU,
            "eV",
        )
        report.check(
            f"{name} eager {device.type} vs cpu |dF|",
            max_abs(gpu[FORCES_KEY], cpu[FORCES_KEY]),
            TOL_TINY_CUDA_VS_CPU,
            "eV/A",
        )
        tiny_model, tiny_batch = model, batch
    except Exception as exc:  # noqa: BLE001
        report.error(name, exc)

    # official packages vs the upstream goldens (the test_load_matches_the_official_package path)
    packages = [
        (
            "allegro_small_default.nequip.zip",
            "goldens.pt",
            TOL_SMALL_PKG_E,
            TOL_SMALL_PKG_F,
            ("molecule", "periodic"),
        ),
        (
            "allegro_small_zbl.nequip.zip",
            "goldens_zbl.pt",
            TOL_SMALL_PKG_E,
            TOL_SMALL_PKG_F,
            ("molecule", "periodic"),
        ),
        (
            "Allegro-OAM-L-0.1.nequip.zip",
            "goldens_oam_l.pt",
            TOL_OAM_E,
            TOL_OAM_F,
            ("molecule", "periodic"),
        ),
    ]
    #: Which official cases also get the CUDA-graph compile (bounded job).
    compile_cases = {
        ("allegro_small_default.nequip.zip", "molecule"),
        ("allegro_small_default.nequip.zip", "periodic"),
        ("allegro_small_zbl.nequip.zip", "molecule"),
        ("Allegro-OAM-L-0.1.nequip.zip", "molecule"),
        ("Allegro-OAM-L-0.1.nequip.zip", "periodic"),
    }
    loaded = []
    if not (WEIGHTS_DIR / "goldens.pt").exists():
        report.skip(
            "allegro.official", f"MOLNEX_ALLEGRO_WEIGHTS_DIR has no goldens.pt ({WEIGHTS_DIR})"
        )
    else:
        for package, goldens, tol_e, tol_f, structures in packages:
            if quick and package.startswith("Allegro-OAM"):
                report.skip(f"allegro.{package}", "--quick")
                continue
            try:
                records = torch.load(WEIGHTS_DIR / goldens, weights_only=True)
            except Exception as exc:  # noqa: BLE001
                report.error(f"allegro.{package} load goldens", exc)
                continue
            for structure in structures:
                name = f"allegro.{package}:{structure}"
                try:
                    record = records[structure]
                    periodic = record.get("cell") is not None
                    wrt = ("atoms.pos", "graphs.strain") if periodic else ("atoms.pos",)
                    model = AllegroPotential.load(WEIGHTS_DIR / package, wrt=wrt).to(device).eval()
                    batch = _golden_batch(record).to(device)
                    out = model(batch.clone())
                    n = record["pos"].shape[0]
                    energy = out[ENERGY_KEY].detach().cpu()
                    report.check(
                        f"{name} |dE|/atom vs golden",
                        float((energy - record["total_energy"].double()).abs().max()) / n,
                        tol_e,
                        "eV/atom",
                    )
                    report.check(
                        f"{name} max|dF| vs golden",
                        max_abs(out[FORCES_KEY].cpu(), record["forces"].double()),
                        tol_f,
                        "eV/A",
                    )
                    if periodic:  # nequip's virial is -dE/deps; molnex graphs.virial is +dE/deps
                        report.check(
                            f"{name} max|dW| vs golden",
                            max_abs(
                                out[VIRIAL_KEY].detach().cpu().view(3, 3),
                                -record["virial"].double().view(3, 3),
                            ),
                            tol_f,
                            "eV",
                        )
                    if (package, structure) in compile_cases:
                        loaded.append((name, model, batch, periodic))
                except Exception as exc:  # noqa: BLE001
                    report.error(name, exc)

    if quick:
        report.skip("allegro explain / cuda-graphs", "--quick")
        return

    # explain: the op is the compile boundary; one eager forward spent the element gate above
    if tiny_model is not None:
        guarded("allegro.tiny explain")(
            explain_one_graph, "allegro.tiny forward", tiny_model, tiny_batch
        )

    if not on_cuda:
        report.skip("allegro cuda-graphs", f"device {device}")
        return
    if tiny_model is not None:
        guarded("allegro.tiny cuda-graphs")(
            cuda_graph_vs_eager, "allegro.tiny", tiny_model, tiny_batch
        )
    for name, model, batch, periodic in loaded:
        guarded(f"{name} cuda-graphs")(
            cuda_graph_vs_eager, name, model, batch, want_virial=periodic
        )


# --------------------------------------------------------------------------- MACE


def mace_cluster(conditioned: bool):
    graphs = (
        {
            "total_charge": torch.zeros(1, dtype=torch.long),
            "total_spin": torch.ones(1, dtype=torch.long),
        }
        if conditioned
        else None
    )
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs=graphs,
    )


def mace_pair_batch():
    return make_graph_batch(
        pos=torch.tensor(PAIR_POS, dtype=torch.float64),
        Z=torch.tensor(PAIR_Z, dtype=torch.long),
        pair_index=full_edge_index(PAIR_BATCH),
        batch=torch.tensor(PAIR_BATCH, dtype=torch.long),
    )


def mace_models():
    """Exactly ``_gh200_verify/step7_ref.py::models`` (seeds, wake, eval)."""
    torch.manual_seed(0)
    matpes = MaceMatpes(
        MACEMatpesSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_MATPES_KWARGS
        ),
        wrt=("atoms.pos",),
    ).eval()
    torch.manual_seed(0)
    omol = MaceOmol(
        MACEOMolSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_OMOL_KWARGS
        ),
        wrt=("atoms.pos",),
    ).eval()
    wake_zero_init_readout(omol)
    torch.manual_seed(0)
    polar = MacePolar(
        MACEPolarSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_POLAR_KWARGS
        ),
        wrt=("atoms.pos",),
    ).eval()
    return {
        "matpes": (matpes, mace_cluster(False)),
        "matpes_pair": (matpes, mace_pair_batch()),
        "omol": (omol, mace_cluster(True)),
        "polar": (polar, mace_cluster(True)),
    }


def check_mace_reference(device: torch.device) -> dict:
    """CUDA eager vs the CPU values of step7_ref.pt (max rel per key)."""
    ref = torch.load(STEP7_REF, weights_only=True)
    cases = mace_models()
    key_names = {
        "graphs.energy": ENERGY_KEY,
        "atoms.energy": ATOMIC_ENERGY_KEY,
        "atoms.forces": FORCES_KEY,
    }
    moved = {}
    for name, (model, batch) in cases.items():
        label = f"mace.{name}"
        try:
            model = model.to(device)
            batch = batch.to(device)
            frame = model(batch.clone())
            for key, tkey in key_names.items():
                want = ref[name][key]
                got = frame[tkey].detach().cpu()
                denom = float(want.abs().max().clamp_min(1e-300))
                report.check(
                    f"{label} eager {device.type} vs cpu-ref {key} max_rel",
                    max_abs(got, want) / denom,
                    TOL_MACE_REF_REL,
                )
            moved[name] = (model, batch)
        except Exception as exc:  # noqa: BLE001
            report.error(label, exc)
    return moved


def check_mace_compile(device: torch.device, moved: dict) -> None:
    on_cuda = device.type == "cuda"
    # explain: products' forward (eager forward already spent the element gate)
    for name in ("matpes", "omol"):
        if name in moved:
            model, batch = moved[name]
            guarded(f"mace.{name} explain")(explain_one_graph, f"mace.{name} forward", model, batch)
    report.skip("mace.polar forward explain", POLAR_NON_PERIODIC)
    # Polar periodic seam (bind_cell + k-space grids), as test_polar does
    name = "mace.polar periodic forward"
    try:
        torch.manual_seed(0)
        polar = (
            MacePolar(
                MACEPolarSpec(
                    atomic_numbers=ATOMIC_NUMBERS,
                    atomic_energies=ATOMIC_ENERGIES,
                    **TINY_POLAR_KWARGS,
                ),
                wrt=("atoms.pos",),
            )
            .eval()
            .to(device)
        )
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).to(device)
        polar.bind_cell(cell)
        periodic = _periodic_cluster(cell.cpu()).to(device)
        polar(periodic.clone())
        explain_one_graph(name, polar, periodic)
        if on_cuda:
            guarded("mace.polar periodic cuda-graphs")(
                cuda_graph_vs_eager,
                "mace.polar periodic",
                polar,
                periodic,
                tol_f=TOL_GRAPH_F_KSPACE,
            )
    except Exception as exc:  # noqa: BLE001
        report.error(name, exc)
    # MACEEncoder.forward
    name = "mace.MACEEncoder forward"
    try:
        encoder = (
            MACEEncoder(
                MACEMatpesSpec(
                    atomic_numbers=ATOMIC_NUMBERS,
                    atomic_energies=ATOMIC_ENERGIES,
                    **TINY_MATPES_KWARGS,
                )
            )
            .eval()
            .to(device)
        )
        cluster = mace_cluster(False).to(device)
        encoder(cluster.clone())
        explain_one_graph(name, encoder, cluster)
    except Exception as exc:  # noqa: BLE001
        report.error(name, exc)
    if not on_cuda:
        report.skip("mace cuda-graphs", f"device {device}")
        return
    for name in ("matpes", "omol"):
        if name in moved:
            model, batch = moved[name]
            guarded(f"mace.{name} cuda-graphs")(cuda_graph_vs_eager, f"mace.{name}", model, batch)
    report.skip("mace.polar cuda-graphs", POLAR_NON_PERIODIC)


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="eager-only subset (login-node smoke)")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False")
        return 2
    print(
        f"=== verify_gh200_hot_paths device={device} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )
    if device.type == "cuda":
        print(f"=== gpu {torch.cuda.get_device_name(0)} sm{torch.cuda.get_device_capability(0)}")
    print(
        "=== molix::scatter_sum native op loaded: "
        f"{getattr(torch.ops.molix, 'scatter_sum', None) is not None}"
    )
    print(
        f"=== config.ftype={config['ftype']} weights_dir={WEIGHTS_DIR} "
        f"step7_ref={STEP7_REF.exists()}"
    )

    print("--- 1. Allegro")
    check_weighted_channels(device)
    check_allegro_models(device, quick=args.quick)

    print("--- 2. MACE")
    moved = check_mace_reference(device)
    if args.quick:
        report.skip("mace explain / cuda-graphs", "--quick")
    else:
        check_mace_compile(device, moved)

    print(
        f"=== SUMMARY passed={report.passed} failed={report.failed} skipped={report.skipped} "
        f"({time.time() - report.t0:.0f}s)"
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
