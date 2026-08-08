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

MACE is a **package** too (`molzoo/mace/`), since
mace-subpackage-restructure-02-core:

| Module | Role |
|--------|------|
| `spec` | `MACESpec` base + `MACEMatpesSpec` / `MACEOMolSpec` (torch-free) |
| `geometry` | `edge_vectors` / `edge_lengths` — additive PBC shifts `S = n·h` |
| `encoder` | `MACEEncoder` — one configuration-driven foundation backbone |
| `potential` | `MACEPotential` — per-graph energy `(B,)` in eV + per-atom forces `(N, 3)` in eV/Å |
| `checkpoint` | `CheckpointRemap` + the `MATPES_REMAP` / `OMOL_REMAP` presets — official-weight key remap |
| `variants` | `MACEMatpes` / `MACEOMol` named foundation models + their `load_*_state_dict` aliases |
| `research` | the freely-configurable research `MACE` encoder (former `molzoo/mace.py`) |

`MACESpec` changed meaning with that move: it is now the **shared
foundation-variant** configuration base in `molzoo.mace.spec` — the thing
`MACEMatpesSpec` / `MACEOMolSpec` derive from and `MACEPotential` is built
out of. The research encoder is configured by keyword and keeps its own,
unrelated `MACEResearchSpec` (`molzoo.mace.research`).

The flat modules `molzoo/mace_matpes.py` and `molzoo/mace_omol.py` were
deleted in mace-subpackage-restructure-06-wire; the **top-level** names are
unchanged, so `from molzoo import MACE, MACEMatpes, MACEOMol,
load_matpes_state_dict, load_omol_state_dict` keeps working (lazily — see
`molzoo/__init__.py`), as does `from molzoo.mace import MACE`.

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

The encoders take a post-collate batch `TensorDict` and read:

- `atoms.Z`: Atomic numbers `(N,)`
- `atoms.pos`: Positions `(N, 3)`
- `edges.edge_index`: Edge indices `(E, 2)` — `[:,0]` source, `[:,1]` target
- `edges.edge_diff`: Edge vectors `(E, 3)` — `pos[target] - pos[source]`
- `edges.edge_dist`: Edge distances `(E,)`

`MACE` writes `atoms.node_features` `(N, num_layers, feature_dim)` back into
the same batch.

## Usage

Research encoder — configured by keyword, features per layer:

```python
from molzoo import MACE
from molrep.embedding.node import DiscreteEmbeddingSpec

encoder = MACE(
    node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64)],
    num_elements=119,
    num_features=64,
    r_max=5.0,
)
batch = encoder(batch)                       # writes atoms.node_features
features = batch["atoms", "node_features"]   # (N, num_layers, 64)
```

Foundation backbone — configured by a validated spec, composed by the caller:

```python
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.geometry import edge_lengths, edge_vectors
from molzoo.mace.spec import MACEMatpesSpec

spec = MACEMatpesSpec(atomic_numbers=[1, 6, 8], atomic_energies=[-13.6, -1029.0, -2041.0])
encoder = MACEEncoder(spec)

vectors = edge_vectors(pos, edge_index)          # optional shifts=... for PBC
node_attrs = encoder.node_attrs(Z, pos.dtype)
edge_feats, cutoff = encoder.radial_features(edge_lengths(vectors), Z, edge_index)
per_layer = encoder.layer_features(
    node_feats=encoder.initial_node_features(node_attrs),
    node_attrs=node_attrs,
    edge_attrs=encoder.angular_features(vectors),
    edge_feats=edge_feats,
    edge_index=edge_index,
    cutoff=cutoff,
)
```

Swap `MACEMatpesSpec` for `MACEOMolSpec` to build the charge/spin-conditioned
OMOL stack from the same backbone.
