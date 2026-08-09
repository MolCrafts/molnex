"""PiNet rank=3 train / inference speed microbench.

Measures wall time for the real operating regimes with the current API:

* energy-only inference (``compute_forces=False`` at init)
* force inference (``compute_forces=True`` at init, eval + enable_grad)
* force training (train, energy + force loss + ``loss.backward``)
* encode-only forward

``forces`` / method knobs are **init-only** — ``forward(batch)`` has no
flags and no runtime branching.

Usage::

    PYTHONPATH=src python benchmarks/bench_pinet_rank3_speed.py --device cuda --steps 50
    PYTHONPATH=src python benchmarks/bench_pinet_rank3_speed.py --device cpu --steps 20
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

# PiNet is pure-PyTorch; the arch-tagged C++ op .so may be linked against a
# different torch ABI (e.g. cpu build vs this CUDA wheel). Soft-skip so the
# microbench still runs — scatter/PME native ops are not on the PiNet path.
_orig_load_library = torch.ops.load_library


def _soft_load_library(path):  # noqa: ANN001
    try:
        return _orig_load_library(path)
    except OSError as exc:
        print(f"[bench] skip native op library ({path}): {exc}")


torch.ops.load_library = _soft_load_library  # type: ignore[method-assign]

from molix.profiler import MockBatch  # noqa: E402
from molpot.derivation import EnergyReadout, ForceReadout  # noqa: E402
from molzoo.pinet import PiNet, PiNetPotential  # noqa: E402

ATOM_TYPES = list(range(1, 8))


@dataclass
class Row:
    name: str
    ms: float
    atoms_per_s: float
    notes: str = ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_it(
    fn: Callable[[], None],
    *,
    device: torch.device,
    n_warmup: int,
    n_steps: int,
) -> float:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0 / n_steps


def _factory(
    device: str,
    *,
    n_atoms: int = 512,
    n_edges: int = 4096,
    n_graphs: int = 16,
) -> Callable[[], object]:
    return MockBatch(
        n_atoms=n_atoms,
        n_edges=n_edges,
        n_graphs=n_graphs,
        atomic_numbers=7,
        device=device,
        seed=0,
    )


def _ef_loss(batch) -> torch.Tensor:
    return (
        batch["graphs", "energy"].float().pow(2).mean()
        + batch["atoms", "forces"].float().pow(2).mean()
    )


def _energy_loss(batch) -> torch.Tensor:
    return batch["graphs", "energy"].float().pow(2).mean()


def build_potential(
    *,
    rank: int = 3,
    depth: int = 3,
    hidden: int = 64,
    emit_property_features: bool | None = None,
    compute_forces: bool = True,
    method: str = "func",
) -> PiNetPotential:
    kwargs: dict = dict(
        atom_types=ATOM_TYPES,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[hidden, hidden],
        pi_nodes=[hidden, hidden],
        ii_nodes=[hidden, hidden],
        depth=depth,
        rank=rank,
        hidden_dim=hidden,
        compute_forces=compute_forces,
    )
    if emit_property_features is not None:
        kwargs["emit_property_features"] = emit_property_features
    pot = PiNetPotential(**kwargs)
    # Rebuild pipeline when method differs from the hard-coded default ("func").
    if method != "func":
        pot.deriv_method = method
        if compute_forces:
            energy_ro = EnergyReadout(pot, method=method, backward=True)  # type: ignore[arg-type]
            force_ro = ForceReadout(pot, method=method)  # type: ignore[arg-type]

            def _pipeline(batch, e=energy_ro, f=force_ro):
                return f(e(batch))

            pot._pipeline = _pipeline
        else:
            pot._pipeline = EnergyReadout(pot, method=method, backward=False)  # type: ignore[arg-type]
    return pot


def build_encoder(*, rank: int = 3, hidden: int = 64, emit: bool = False) -> PiNet:
    return PiNet(
        atom_types=ATOM_TYPES,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[hidden, hidden],
        pi_nodes=[hidden, hidden],
        ii_nodes=[hidden, hidden],
        depth=3,
        rank=rank,
        emit_property_features=emit,
    )


def _row(name: str, ms: float, n_atoms: int, notes: str = "") -> Row:
    return Row(name=name, ms=ms, atoms_per_s=n_atoms / (ms / 1000.0), notes=notes)


def print_table(rows: list[Row], baseline: str | None = None) -> None:
    base_ms = next((r.ms for r in rows if r.name == baseline), None) if baseline else None
    print(f"\n{'name':<42} {'ms/step':>10} {'atoms/s':>12} {'rel':>8}  notes")
    print("-" * 90)
    for r in rows:
        rel = f"{r.ms / base_ms:5.2f}x" if base_ms and base_ms > 0 and r.ms == r.ms else "  -"
        print(f"{r.name:<42} {r.ms:10.3f} {r.atoms_per_s:12,.0f} {rel:>8}  {r.notes}")


def run_matrix(
    *,
    device: torch.device,
    n_steps: int,
    n_warmup: int,
    n_atoms: int,
    n_edges: int,
    n_graphs: int,
    skip_compile: bool = False,
) -> list[Row]:
    dev = str(device)
    factory = _factory(dev, n_atoms=n_atoms, n_edges=n_edges, n_graphs=n_graphs)
    rows: list[Row] = []

    # ---- encoder-only -------------------------------------------------------
    for rank, emit, tag in (
        (3, False, "enc rank3 emit=0"),
        (3, True, "enc rank3 emit=1"),
        (1, False, "enc rank1 emit=0"),
    ):
        enc = build_encoder(rank=rank, emit=emit).to(device)
        enc.eval()

        def step(e=enc):
            with torch.no_grad():
                e(factory())

        ms = _time_it(step, device=device, n_warmup=n_warmup, n_steps=n_steps)
        rows.append(_row(tag, ms, n_atoms))

    # ---- energy-only inference ---------------------------------------------
    pot_e = build_potential(rank=3, emit_property_features=False, compute_forces=False).to(device)
    pot_e.eval()

    def energy_inf():
        with torch.no_grad():
            pot_e(factory())

    ms = _time_it(energy_inf, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("energy-inf rank3 (no forces)", ms, n_atoms, "baseline"))

    # compiled energy (inference only)
    if skip_compile:
        rows.append(_row("energy-inf + torch.compile", float("nan"), n_atoms, "skipped"))
    else:
        pot_ec = build_potential(rank=3, emit_property_features=False, compute_forces=False).to(
            device
        )
        pot_ec.eval()
        try:
            pot_ec = torch.compile(pot_ec, backend="inductor", fullgraph=False)
            for _ in range(3):
                with torch.no_grad():
                    pot_ec(factory())
            _sync(device)

            def energy_compiled():
                with torch.no_grad():
                    pot_ec(factory())

            ms = _time_it(
                energy_compiled, device=device, n_warmup=max(5, n_warmup), n_steps=n_steps
            )
            rows.append(_row("energy-inf + torch.compile", ms, n_atoms))
        except Exception as e:  # noqa: BLE001
            rows.append(
                _row(
                    "energy-inf + torch.compile",
                    float("nan"),
                    n_atoms,
                    f"FAIL {type(e).__name__}",
                )
            )
        finally:
            torch.compiler.reset()

    # ---- force inference (eval fused path, method=func) --------------------
    for emit, tag in ((False, "force-inf eval emit=0"), (True, "force-inf eval emit=1")):
        pot = build_potential(
            rank=3, emit_property_features=emit, compute_forces=True, method="func"
        ).to(device)
        pot.eval()
        assert pot.encoder.emit_property_features is emit

        def step_eval(m=pot):
            m.eval()
            with torch.enable_grad():
                m(factory())

        ms = _time_it(step_eval, device=device, n_warmup=n_warmup, n_steps=n_steps)
        rows.append(_row(tag, ms, n_atoms, "func has_aux 1-pass"))

    # rank-1 force inf for ratio
    pot_r1 = build_potential(
        rank=1, emit_property_features=False, compute_forces=True, method="func"
    ).to(device)
    pot_r1.eval()

    def force_r1():
        with torch.enable_grad():
            pot_r1(factory())

    ms = _time_it(force_r1, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("force-inf eval rank1", ms, n_atoms))

    # force-inf grad method (autograd 1-pass)
    pot_ig = build_potential(
        rank=3, emit_property_features=False, compute_forces=True, method="grad"
    ).to(device)
    pot_ig.eval()

    def force_inf_grad():
        with torch.enable_grad():
            pot_ig(factory())

    ms = _time_it(force_inf_grad, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("force-inf eval grad", ms, n_atoms, "autograd 1-pass"))

    # ---- force training ----------------------------------------------------
    for method, tag, note in (
        ("func", "force-train func emit=0", "func has_aux + loss.backward"),
        ("grad", "force-train grad emit=0", "autograd create_graph + loss.backward"),
    ):
        pot = build_potential(
            rank=3,
            emit_property_features=False,
            compute_forces=True,
            method=method,
        ).to(device)
        pot.train()
        opt = torch.optim.Adam(pot.parameters(), lr=1e-3)

        def step(m=pot, o=opt):
            o.zero_grad(set_to_none=True)
            out = m(factory())
            _ef_loss(out).backward()
            o.step()

        ms = _time_it(step, device=device, n_warmup=n_warmup, n_steps=n_steps)
        rows.append(_row(tag, ms, n_atoms, note))

    # emit=1 force train (func default path)
    pot = build_potential(
        rank=3, emit_property_features=True, compute_forces=True, method="func"
    ).to(device)
    pot.train()
    opt = torch.optim.Adam(pot.parameters(), lr=1e-3)

    def step_emit1(m=pot, o=opt):
        o.zero_grad(set_to_none=True)
        _ef_loss(m(factory())).backward()
        o.step()

    ms = _time_it(step_emit1, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("force-train func emit=1", ms, n_atoms, "func has_aux + loss.backward"))

    # energy-only train (no forces)
    pot_et = build_potential(rank=3, emit_property_features=False, compute_forces=False).to(device)
    pot_et.train()
    opt_et = torch.optim.Adam(pot_et.parameters(), lr=1e-3)

    def energy_train():
        opt_et.zero_grad(set_to_none=True)
        out = pot_et(factory())
        _energy_loss(out).backward()
        opt_et.step()

    ms = _time_it(energy_train, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("energy-train (no forces)", ms, n_atoms))

    # ---- forward-only (no loss.backward) train vs eval ---------------------
    pot = build_potential(
        rank=3, emit_property_features=False, compute_forces=True, method="func"
    ).to(device)
    b = factory()

    def just_forward_train():
        pot.train()
        with torch.enable_grad():
            pot(b.clone())

    ms_fwd = _time_it(just_forward_train, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(_row("force-train forward only", ms_fwd, n_atoms, "no loss.backward"))

    def just_forward_eval():
        pot.eval()
        with torch.enable_grad():
            pot(b.clone())

    ms_ev = _time_it(just_forward_eval, device=device, n_warmup=n_warmup, n_steps=n_steps)
    rows.append(
        _row(
            "force-eval forward only",
            ms_ev,
            n_atoms,
            f"train/eval fwd ratio={ms_fwd / ms_ev:.2f}x" if ms_ev > 0 else "",
        )
    )

    print(f"\n# batch: atoms={n_atoms} edges={n_edges} graphs={n_graphs}  device={device}")
    print(f"# torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"# GPU: {torch.cuda.get_device_name(0)}")

    return rows


def profile_cpu_ops(n_steps: int, n_atoms: int, n_edges: int, n_graphs: int) -> None:
    from torch.profiler import ProfilerActivity, profile

    factory = _factory("cpu", n_atoms=n_atoms, n_edges=n_edges, n_graphs=n_graphs)
    pot = build_potential(rank=3, emit_property_features=False, compute_forces=True)
    pot.train()
    for _ in range(3):
        _ef_loss(pot(factory())).backward()
        pot.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as p:
        for _ in range(n_steps):
            _ef_loss(pot(factory())).backward()
            pot.zero_grad(set_to_none=True)
    print("\n# Top CPU ops — force-train step (self time):")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--atoms", type=int, default=512)
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--graphs", type=int, default=16)
    ap.add_argument("--profile-ops", action="store_true")
    ap.add_argument("--skip-compile", action="store_true", help="Skip torch.compile cells")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but torch.cuda.is_available() is False "
            f"(torch={torch.__version__}). Install a CUDA build or use --device cpu."
        )

    device = torch.device(args.device)
    rows = run_matrix(
        device=device,
        n_steps=args.steps,
        n_warmup=args.warmup,
        n_atoms=args.atoms,
        n_edges=args.edges,
        n_graphs=args.graphs,
        skip_compile=args.skip_compile,
    )
    print_table(rows, baseline="energy-inf rank3 (no forces)")

    by_name = {r.name: r.ms for r in rows if r.ms == r.ms}

    def ratio(a: str, b: str) -> str:
        if a in by_name and b in by_name and by_name[b] > 0:
            return f"{by_name[a] / by_name[b]:.2f}x"
        return "n/a"

    print("\n# Key ratios")
    print(
        "  force-train-func / energy-train = "
        f"{ratio('force-train func emit=0', 'energy-train (no forces)')}"
    )
    print(
        "  force-train-grad / energy-train = "
        f"{ratio('force-train grad emit=0', 'energy-train (no forces)')}"
    )
    print(
        "  force-train-func / force-inf    = "
        f"{ratio('force-train func emit=0', 'force-inf eval emit=0')}"
    )
    print(
        "  force-train-grad / force-inf    = "
        f"{ratio('force-train grad emit=0', 'force-inf eval emit=0')}"
    )
    print(
        "  force-train-grad / func         = "
        f"{ratio('force-train grad emit=0', 'force-train func emit=0')}"
    )
    print(
        "  force-inf-grad / func           = "
        f"{ratio('force-inf eval grad', 'force-inf eval emit=0')}"
    )
    print(
        "  force-inf train-fwd / eval-fwd  = "
        f"{ratio('force-train forward only', 'force-eval forward only')}"
    )
    print(f"  enc emit=1 / emit=0             = {ratio('enc rank3 emit=1', 'enc rank3 emit=0')}")
    print(f"  enc rank3 / rank1               = {ratio('enc rank3 emit=0', 'enc rank1 emit=0')}")
    print(
        "  force-inf rank3 / rank1         = "
        f"{ratio('force-inf eval emit=0', 'force-inf eval rank1')}"
    )

    if args.profile_ops and device.type == "cpu":
        profile_cpu_ops(min(args.steps, 15), args.atoms, args.edges, args.graphs)


if __name__ == "__main__":
    main()
