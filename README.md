<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;MolNex
</h1>

<p><strong>A layered molecular machine-learning framework — train, represent, compose, and assemble.</strong></p>

<p>
  <a href="https://github.com/MolCrafts/molnex/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molnex/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/molnex/"><img src="https://img.shields.io/pypi/v/molnex?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/molnex/"><img src="https://img.shields.io/pypi/pyversions/molnex?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff"></a>
</p>

<p>
  <a href="https://molcrafts.github.io/molnex/"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

MolNex is a molecular machine-learning framework organized as four cooperating
packages — `molix`, `molrep`, `molpot`, and `molzoo` — that can be used together
or independently.

> **Under active development.** Public APIs may change between minor releases.

## Vision

Molecular ML projects rarely need just one model. They accumulate training
infrastructure, representation modules, compositional output layers, and several
reference architectures — and when all of that collapses into a single library
built around one preferred model, the codebase becomes hard to extend, test, and
reason about.

MolNex exists to keep those concerns apart so they can grow at their own pace. It
aspires to be a framework where a contributor always knows which layer they are
working in, what that layer is allowed to own, and how it connects to the rest of
the stack — rather than a trainer wrapped around one favorite architecture.

That separation is what unlocks the goal: multiple model families living in one
codebase, training infrastructure and representation modules reused freely across
them, and reference models that stay replaceable examples of assembly instead of
becoming the framework itself.

## Capabilities

| Package | Capability |
|---------|------------|
| `molix`   | Training and execution — `Trainer`, `TrainState`, step protocol, hook lifecycle, packed-cache data pipeline, dataset loaders (QM9, RevMD17, 3BPA, Water-LES, MolRec), checkpointing, `torch.compile` / CUDA-graph and AOT-Inductor export, in-process Langevin MD, engine/LAMMPS export bridge, and arch-tagged native C++ ops |
| `molrep`  | Representation learning — node/radial/angular embeddings, cutoffs, equivariant interaction blocks (tensor products, symmetric contraction, PiNet GC blocks), pooling and readout heads |
| `molpot`  | Potentials and composition — classical potential terms (LJ, harmonic bonds/angles/dihedrals, electrostatics / PME), prediction heads, force/stress derivation (`ForceDerivation`), `PotentialComposer`, and assembled models such as Sonata |
| `molzoo`  | Assembled reference encoders — MACE, Allegro, PiNet (plus full energy/force façades `PiNetPotential` / `MACEOMol`); each family should ship a paper-traceable spec under `src/molzoo/specs/` |

## Install

```bash
pip install molnex
```

Requires Python >= 3.10 and PyTorch >= 2.10. The package builds native C++ ops
via scikit-build-core and CMake >= 4.0; an editable install is
`pip install -e ".[dev]"`. See [Installation](https://molcrafts.github.io/molnex/installation/)
for the full build setup.

## Quick start

Train a model with `molix.Trainer`:

```python
import torch
from molix.core.trainer import Trainer

trainer = Trainer(
    model=model,                       # any nn.Module
    loss_fn=loss_fn,                   # (predictions, batch) -> scalar
    optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1),
)

state = trainer.train(datamodule, max_epochs=5)
print(state["train/loss"])
```

See the [Molix Quick Start](https://molcrafts.github.io/molnex/molix/tutorials/quick-start/)
for the runnable end-to-end version, and
[Train a Graph Model](https://molcrafts.github.io/molnex/molix/tutorials/train-a-graph-model/)
for molecular graph batches.

## Documentation

- [Documentation home](https://molcrafts.github.io/molnex/)
- [Installation](https://molcrafts.github.io/molnex/installation/)
- [Molix](https://molcrafts.github.io/molnex/molix/) — training, hooks, data, and execution
- [MolRep](https://molcrafts.github.io/molnex/molrep/) — representation learning modules
- [MolPot](https://molcrafts.github.io/molnex/molpot/) — potential composition and physical outputs
- [MolZoo](https://molcrafts.github.io/molnex/molzoo/) — reference encoder families
- [API Reference](https://molcrafts.github.io/molnex/api/)

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| [molpy](https://github.com/MolCrafts/molpy)     | Python toolkit — the shared molecular data model & workflow layer |
| [molrs](https://github.com/MolCrafts/molrs)     | Rust core — molecular data structures & compute kernels (native + WASM) |
| [molpack](https://github.com/MolCrafts/molpack) | Packmol-grade molecular packing (Rust + Python) |
| [molvis](https://github.com/MolCrafts/molvis)   | WebGL molecular visualization & editing |
| [molexp](https://github.com/MolCrafts/molexp)   | Workflow & experiment-management platform |
| **molnex** | Molecular machine-learning framework — this repo |
| [molq](https://github.com/MolCrafts/molq)       | Unified job queue — local / SLURM / PBS / LSF |
| [molcfg](https://github.com/MolCrafts/molcfg)   | Layered configuration library |
| [mollog](https://github.com/MolCrafts/mollog)   | Structured logging, stdlib-compatible |
| [molhub](https://github.com/MolCrafts/molhub)   | Molecular dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp)   | MCP server for the ecosystem |
| [molrec](https://github.com/MolCrafts/molrec)   | Atomistic record specification |

## Contributing

Contributions are welcome — see the [documentation](https://molcrafts.github.io/molnex/)
to get started.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>
