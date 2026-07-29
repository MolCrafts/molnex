# molzoo

Molecular model zoo. Provides encoder architectures (MACE, Allegro, PiNet).

PiNet is a **package** (`molzoo/pinet/`), not a single file:

| Module | Role |
|--------|------|
| `spec` | `PiNetSpec` config |
| `geometry` | PBC-safe edge displacement helpers |
| `encoder` | `PiNet` feature encoder (molrep GC blocks only) |
| `potential` | `PiNetPotential` energy + functorch forces (composition; long-term home molpot) |
| `properties` | `PiNetDipole` / `PiNetPolarizability` façades over molpot heads |

Public imports stay stable: `from molzoo.pinet import PiNet, PiNetPotential`.
The Sonata model lives in `molpot.composition`, not here.

## Model Specifications

Each model in this package ships with **one** spec artifact in `specs/`:

- `<encoder>.md` — paper↔code↔reference traceable spec (10-section template; includes I/O, paper↔code mapping, adaptation ledger, benchmark contract with embedded run log).

**Read the spec before modifying the model** — any change to a module's math MUST be reflected in the corresponding spec.

| Model     | Spec | Paper |
|-----------|------|-------|
| Allegro   | [`specs/allegro.md`](specs/allegro.md) | Musaelian et al., Nat. Commun. 2023 ([arXiv](https://arxiv.org/abs/2204.05249)) |
| MACE      | [`specs/mace.md`](specs/mace.md) | Batatia et al., NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2206.07697)) |
| PiNet     | [`specs/pinet2.md`](specs/pinet2.md) | package under `molzoo/pinet/` |
| MACE-OMOL | [`specs/mace_omol.md`](specs/mace_omol.md) | full energy/force model (lazy import) |

### Spec workflow

One skill + one agent keep `<encoder>.md` aligned with code and paper:

| Trigger | Command | Effect |
|---------|---------|--------|
| Introducing a new encoder | `/molzoo-spec <encoder> --paper <arxiv_url>` | **create mode** (auto-detected when spec is missing): seeds `<encoder>.md` from the 10-section template |
| Filling placeholders or fixing drift | `/molzoo-spec <encoder>` | **update mode** (auto-detected when spec exists): fills §2/§3/§5 from paper + reference, reconciles drift, refreshes anchors |
| After a benchmark or training run | `/molzoo-spec <encoder> --log <k=v ...>` | appends one row to §7.4 (Run log) of `<encoder>.md`; warns + suggests `molzoo-auditor` on MAE regression or dirty tree |
| Asking a question about spec content | (no command — answer inline) | the skill's §"Lookup Behavior" rule applies: always `Read` the spec, quote verbatim, refuse on a miss |
| Verify code vs paper | `molzoo-auditor` agent | **prints** ≥ 1 verdict report to the developer; on ⚠️/🆚 patches `<encoder>.md` (§2/§3.2/§4/§5) — the spec diff is the persistent trace |

**The loop never closes silently.** Every operation either updates the spec, prints a verdict, or explicitly delegates. See `CLAUDE.md` for the full contract.

## Input Conventions

Both encoders accept keyword tensors:

- `Z`: Atomic numbers `(N,)`
- `edge_dist`: Edge distances `(E,)`
- `edge_diff`: Edge vectors `(E, 3)`
- `edge_index`: Edge indices `(E, 2)`

Output: `(N, num_layers, feature_dim)` — per-atom, per-layer features.

## Usage

```python
import torch
from molzoo import MACE, MACESpec
from molrep.embedding.node import DiscreteEmbeddingSpec
from molpot import LayerPooling, PotentialComposer, LJParameterHead, LJ126

encoder = MACE(MACESpec(
    node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64)],
    num_elements=119,
    num_features=64,
    r_max=5.0,
))

Z = torch.randint(0, 10, (20,))
features = encoder(
    Z=Z,
    edge_dist=torch.rand(80),
    edge_diff=torch.randn(80, 3),
    edge_index=torch.randint(0, 20, (80, 2)),
)

pool = LayerPooling("mean")
node_features = pool(features)  # (20, 64)
```
