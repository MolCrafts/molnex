# molzoo

Molecular model zoo: MACE, Allegro and PiNet2 (plus the learnable Class-I
force field Rondo). Every model package has the same shape:

| Module | Role |
|--------|------|
| `spec` | torch-free pydantic validator (`MACESpec` family, `AllegroSpec`, `PiNetSpec`) |
| `encoder` | feature body exposing primitives (`MACEEncoder`, `AllegroEncoder`, `PiNetEncoder`), each `(ElementTable, nn.Module)` |
| `layers` | the blocks only this family uses (MACE also `heads`, `coupling`) |
| `potential` | product that **subclasses** the body (flat checkpoint keys) and `molix.autograd.Potential`: `energy(frame)` writes `atoms.energy` / `graphs.energy`; `forward = Derivative(self.energy, wrt)` built in `__init__` |
| `checkpoint` + `utils` | official key tables and archive readers behind `.load(path)` (MACE, Allegro) |
| `properties` | PiNet2 dipole / polarizability products over `molpot.heads` |

Shared machinery lives one level up: `molzoo.encoder` (`ElementTable`, the
element-table precondition spent once on the first eager call) and
`molzoo.checkpoint` (`OfficialArchive`, `StubUnpickler`, `CheckpointRemap`);
the generic blocks are in `molrep` (edge geometry in
`molrep.embedding.geometry`). A block that serves one
model lives in that model's package (`mace/layers.py`, `mace/heads.py`,
`pinet/layers.py`, `allegro/layers.py`).

| Family | Products | Official weights |
|--------|----------|------------------|
| MACE (`molzoo.mace`) | `MaceMatpes`, `MaceOmol`, `MaceOff`, `MaceLes`, `MacePolar`; `MACEEncoder` for feature heads | cuEq-converted MACE checkpoints |
| Allegro (`molzoo.allegro`) | `AllegroPotential`; `AllegroEncoder` for feature heads (Sonata) | `.nequip.zip` packages (e.g. Allegro-OAM-L) |
| PiNet2 (`molzoo.pinet`) | `PiNetPotential`, `PiNetDipole`, `PiNetPolarizability`; `PiNetEncoder` | — (TensorFlow PiNN checkpoints out of scope) |

Construction takes the typed spec object and ``wrt``; every default lives on
the spec and official weights come through ``load``. Import each family from
its subpackage (``molzoo`` itself exports nothing)::

    import torch
    from molzoo.allegro import AllegroPotential, AllegroSpec
    from molzoo.mace import MACEMatpesSpec, MaceMatpes
    from molzoo.pinet import PiNetPotential, PiNetSpec

    model = MaceMatpes(MACEMatpesSpec(atomic_numbers=[1, 6, 8], atomic_energies=[-13.6, -1029.0, -2041.0]), wrt=("atoms.pos",), dtype=torch.float64)
    model = MaceMatpes.load(path, wrt=("atoms.pos",), dtype=torch.float64)
    model = AllegroPotential(AllegroSpec(atomic_numbers=[1, 6, 8], r_max=4.0, num_scalar_features=16, num_tensor_features=4), wrt=("atoms.pos",), dtype=torch.float64)
    model = AllegroPotential.load("Allegro-OAM-L-0.1.nequip.zip", wrt=("atoms.pos",), dtype=torch.float64)
    model = PiNetPotential(PiNetSpec(atomic_numbers=[1, 6, 8], rank=3), wrt=("atoms.pos",), dtype=torch.float64)

``model(frame)`` runs ``energy`` and the ``wrt`` derivatives in one forward and
one ``autograd.grad``; ``Derivative(model.energy, wrt=...)`` composed outside
is the same op.

See [MACE precision and official checkpoints](../../docs/molzoo/user-guide/mace.md)
and the [Allegro user guide](../../docs/molzoo/user-guide/allegro.md).

## Model Specifications

Each model in this package ships with **one** spec artifact in `specs/`:

- `<encoder>.md` — paper↔code↔reference traceable spec (10-section template; includes I/O, paper↔code mapping, adaptation ledger, benchmark contract with embedded run log).

**Read the spec before modifying the model** — any change to a module's math MUST be reflected in the corresponding spec.

| Model     | Spec | Paper |
|-----------|------|-------|
| Allegro   | [`specs/allegro.md`](specs/allegro.md) | Musaelian et al., Nat. Commun. 2023 ([arXiv](https://arxiv.org/abs/2204.05249)) |
| MACE-MatPES | [`specs/mace_matpes.md`](specs/mace_matpes.md) | Batatia et al., NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2206.07697)) |
| PiNet2    | [`specs/pinet2.md`](specs/pinet2.md) | Li et al., JCTC 2025 ([DOI](https://doi.org/10.1021/acs.jctc.4c01570)) |
| MACE-OMOL | [`specs/mace_omol.md`](specs/mace_omol.md) | full energy/force model |

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
- `pairs.pair_index`: Pair indices `(E, 2)` — `[:,0]` source, `[:,1]` target
- `pairs.pair_diff`: Pair vectors `(E, 3)` — `pos[target] - pos[source]`
- `pairs.pair_dist`: Pair distances `(E,)`

Encoders write their features back into the same batch: `MACEEncoder` and
`PiNetEncoder` write `atoms.node_features` `(N, num_layers, feature_dim)`,
`AllegroEncoder` writes `pairs.edge_features` `(E, F·(L+1))`.

## Usage

Encoder contract — a validated spec, one call per batch:

```python
import torch
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.spec import MACEMatpesSpec

spec = MACEMatpesSpec(atomic_numbers=[1, 6, 8], atomic_energies=[-13.6, -1029.0, -2041.0])
batch = MACEEncoder(spec, dtype=torch.float64)(batch)  # writes atoms.node_features
# (N, num_interactions, num_features): each layer's invariant 0e block
features = batch["atoms", "node_features"]
```

The same backbone, primitive by primitive, composed by the caller:

```python
import torch
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.spec import MACEMatpesSpec

spec = MACEMatpesSpec(atomic_numbers=[1, 6, 8], atomic_energies=[-13.6, -1029.0, -2041.0])
encoder = MACEEncoder(spec, dtype=torch.float64)

vectors = edge_vectors(pos, edge_index, edge_diff=pair_diff)  # imaged value, gradient via pos
elements = encoder.element_index(Z)
edge_feats, cutoff = encoder.radial_features(edge_lengths(vectors), Z, edge_index)
per_layer = encoder.layer_features(
    node_feats=encoder.initial_node_features(elements),
    elements=elements,
    edge_attrs=encoder.angular_features(vectors),
    edge_feats=edge_feats,
    edge_index=edge_index,
    cutoff=cutoff,
)
```

Swap `MACEMatpesSpec` for `MACEOMolSpec` to build the charge/spin-conditioned
OMOL stack from the same backbone.
