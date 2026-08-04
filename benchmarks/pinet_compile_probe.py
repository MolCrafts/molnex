"""Probe how torch.compile can wrap current PiNetPotential.

Run on a GPU node (GH200)::

    PYTHONPATH=src python benchmarks/pinet_compile_probe.py
"""

from __future__ import annotations

import traceback

import torch

_orig = torch.ops.load_library


def soft(p):  # noqa: ANN001
    try:
        return _orig(p)
    except OSError as e:
        print(f"[probe] skip op: {e}")


torch.ops.load_library = soft  # type: ignore[method-assign]

from molix.compile import Compiler  # noqa: E402
from molix.profiler import MockBatch  # noqa: E402
from molpot.derivation import EnergyReadout, ForceReadout  # noqa: E402
from molzoo.pinet import PiNet, PiNetPotential  # noqa: E402

ATOM = list(range(1, 8))
device = "cuda" if torch.cuda.is_available() else "cpu"


def factory(n=64, e=256, g=4):
    return MockBatch(
        n_atoms=n, n_edges=e, n_graphs=g, atomic_numbers=7, device=device, seed=0
    )


def make_pot(*, forces=False, method="func", rank=3, depth=2, hidden=32):
    pot = PiNetPotential(
        atom_types=ATOM,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[hidden, hidden],
        pi_nodes=[hidden, hidden],
        ii_nodes=[hidden, hidden],
        depth=depth,
        rank=rank,
        hidden_dim=hidden,
        compute_forces=forces,
        emit_property_features=False,
    )
    if method != "func":
        pot.deriv_method = method
        if forces:
            er = EnergyReadout(pot, method=method, backward=True)
            fr = ForceReadout(pot, method=method)
            pot._pipeline = lambda b, e=er, f=fr: f(e(b))
        else:
            pot._pipeline = EnergyReadout(pot, method=method, backward=False)
    if device == "cuda":
        pot = pot.cuda()
    return pot


def try_call(label, fn):
    print(f"\n=== {label} ===", flush=True)
    try:
        out = fn()
        print("OK", out, flush=True)
        return True
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:220]}", flush=True)
        lines = traceback.format_exc().strip().splitlines()
        for line in lines[-25:]:
            print(line, flush=True)
        return False
    finally:
        torch.compiler.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    print(
        "torch",
        torch.__version__,
        "device",
        device,
        "gpu",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        flush=True,
    )

    def a():
        pot = make_pot(forces=False).eval()
        c = torch.compile(pot, backend="inductor", fullgraph=False)
        with torch.no_grad():
            b = c(factory()())
        return ("energy", float(b["graphs", "energy"].sum()))

    try_call("A energy-only torch.compile(fullgraph=False)", a)

    def b():
        pot = make_pot(forces=False).eval()
        c = torch.compile(pot, backend="inductor", fullgraph=True)
        with torch.no_grad():
            b = c(factory()())
        return ("energy", float(b["graphs", "energy"].sum()))

    try_call("B energy-only torch.compile(fullgraph=True)", b)

    def c():
        enc = PiNet(
            atom_types=ATOM,
            r_max=5.0,
            n_basis=5,
            pp_nodes=[32, 32],
            pi_nodes=[32, 32],
            ii_nodes=[32, 32],
            depth=2,
            rank=3,
            emit_property_features=False,
        )
        if device == "cuda":
            enc = enc.cuda()
        enc = enc.eval()
        cenc = torch.compile(enc, backend="inductor", fullgraph=False)
        with torch.no_grad():
            b = cenc(factory()())
        return ("nf", tuple(b["atoms", "node_features"].shape))

    try_call("C encoder torch.compile(fullgraph=False)", c)

    def d():
        pot = make_pot(forces=False).eval()

        class Core(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, batch):
                return self.p._write_energy(batch)

        core = torch.compile(Core(pot), backend="inductor", fullgraph=False)
        with torch.no_grad():
            b = core(factory()())
        return ("energy", float(b["graphs", "energy"].sum()))

    try_call("D Core(_write_energy) compile fullgraph=False", d)

    def e():
        pot = make_pot(forces=False).eval()
        batch = factory()()
        expl = torch._dynamo.explain(pot)(batch)
        print("graph_count", expl.graph_count, flush=True)
        print("graph_break_count", expl.graph_break_count, flush=True)
        if expl.break_reasons:
            for i, r in enumerate(expl.break_reasons[:8]):
                print(f"  break[{i}] {str(r)[:240]}", flush=True)
        return ("breaks", expl.graph_break_count)

    try_call("E dynamo.explain energy-only", e)

    def f():
        pot = make_pot(forces=True, method="grad").eval()
        c = torch.compile(pot, backend="inductor", fullgraph=False)
        with torch.enable_grad():
            b = c(factory()())
        return (
            "E",
            float(b["graphs", "energy"].sum()),
            "F",
            float(b["atoms", "forces"].abs().mean()),
        )

    try_call("F force grad torch.compile(fullgraph=False)", f)

    def g():
        pot = make_pot(forces=True, method="func").eval()
        c = torch.compile(pot, backend="inductor", fullgraph=False)
        with torch.enable_grad():
            b = c(factory()())
        return (
            "E",
            float(b["graphs", "energy"].sum()),
            "F",
            float(b["atoms", "forces"].abs().mean()),
        )

    try_call("G force func torch.compile(fullgraph=False)", g)

    def h():
        pot = make_pot(forces=False).eval()
        pot = Compiler(cuda_graphs=True)(pot)
        fac = factory(n=64, e=256, g=4)
        with torch.no_grad():
            for _ in range(3):
                pot(fac())
            if device == "cuda":
                torch.cuda.synchronize()
            b = pot(fac())
        return ("energy", float(b["graphs", "energy"].sum()))

    try_call("H Compiler(cuda_graphs=True) energy-only", h)

    def i():
        pot = make_pot(forces=False).eval()
        pot.encoder = torch.compile(pot.encoder, backend="inductor", fullgraph=False)
        pot._pipeline = EnergyReadout(pot, method="func", backward=False)
        with torch.no_grad():
            b = pot(factory()())
        return ("energy", float(b["graphs", "energy"].sum()))

    try_call("I compile encoder submodule, energy readout eager", i)

    def j():
        pot = make_pot(forces=True, method="grad").train()
        pot = torch.compile(pot, backend="inductor", fullgraph=False)
        opt = torch.optim.Adam(pot.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        b = pot(factory()())
        loss = b["graphs", "energy"].pow(2).mean() + b["atoms", "forces"].pow(2).mean()
        loss.backward()
        opt.step()
        return ("loss", float(loss.detach()))

    try_call("J force-train grad compiled fullgraph=False one step", j)

    # K: fullgraph force with grad (old production claim)
    def k():
        pot = make_pot(forces=True, method="grad").eval()
        pot = torch.compile(pot, backend="inductor", fullgraph=True, dynamic=False)
        with torch.enable_grad():
            b = pot(factory()())
        return (
            "E",
            float(b["graphs", "energy"].sum()),
            "F",
            float(b["atoms", "forces"].abs().mean()),
        )

    try_call("K force grad torch.compile(fullgraph=True)", k)

    # L: plain tensors path — extract energy as function of (Z, pos, edge_index, batch)
    #    (future compile surface if TensorDict is the problem)
    def l():
        pot = make_pot(forces=False).eval()
        batch = factory()()

        class FlatEnergy(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.encoder = p.encoder
                self.out_layers = p.out_layers
                self.energy_aggregation = p.energy_aggregation

            def forward(self, Z, pos, edge_index, atom_batch, num_graphs: int):
                from tensordict import TensorDict

                td = TensorDict(
                    {
                        "atoms": TensorDict(
                            {"Z": Z, "pos": pos, "batch": atom_batch},
                            batch_size=[Z.shape[0]],
                        ),
                        "edges": TensorDict(
                            {"edge_index": edge_index},
                            batch_size=[edge_index.shape[0]],
                        ),
                        "graphs": TensorDict({}, batch_size=[num_graphs]),
                    },
                    batch_size=[],
                )
                td = self.encoder(td)
                block_outputs = td["atoms", "p1_block_outputs"]
                output = block_outputs.new_zeros(block_outputs.shape[0], 1)
                for i, out_layer in enumerate(self.out_layers):
                    output = out_layer(block_outputs[:, i, :], output)
                atom_energy = output.squeeze(-1)
                return self.energy_aggregation(
                    atom_energy, atom_batch, num_graphs=num_graphs
                )

        flat = FlatEnergy(pot)
        if device == "cuda":
            flat = flat.cuda()
        flat = flat.eval()
        cflat = torch.compile(flat, backend="inductor", fullgraph=True)
        Z = batch["atoms", "Z"]
        pos = batch["atoms", "pos"]
        ei = batch["edges", "edge_index"]
        ab = batch["atoms", "batch"]
        ng = int(batch["graphs"].batch_size[0])
        with torch.no_grad():
            e = cflat(Z, pos, ei, ab, ng)
        return ("energy", float(e.sum()), "shape", tuple(e.shape))

    try_call("L flat (Z,pos,edge_index) energy fullgraph=True", l)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
