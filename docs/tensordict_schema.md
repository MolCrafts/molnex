# TensorDict Schema Reference

## What AtomTD is

MolNex v2.0 uses a hierarchical TensorDict schema where all molecular data flows through `AtomTD`, the protocol-level container defined in `molix.data`. The schema organizes fields into explicit namespaces that correspond to physical or semantic groupings. Each namespace acts as a first-level key in the TensorDict hierarchy, and individual fields are accessed as second-level keys under their parent namespace. For example, atomic positions are stored as `atoms.x`, neighbor distances as `pairs.dist`, and molecular energy as `target.energy`. This two-level structure allows models, datasets, and analysis tools to work with a consistent and self-documenting data layout.

## Why namespaces exist

The namespace design solves several long-standing problems in molecular machine learning workflows. Without explicit namespaces, field names must encode their semantic category through prefixes or suffixes, leading to inconsistent conventions across codebases (such as `atom_z` versus `z_atom` versus `atomic_number`). Namespaces eliminate this ambiguity by making the category explicit and separating it from the field identity. They also simplify batch processing and transfer between representations, since each namespace can be treated as a cohesive unit with its own mutability rules and gradient requirements. For instance, all topology fields under `bonds.*`, `pairs.*`, `angles.*`, and `dihedrals.*` are immutable after construction, while `atoms.x` is mutable to support molecular dynamics and gradient-based force computation. This design also ensures that predictions and targets use identical keys within the `target.*` namespace, enabling direct loss computation without manual key translation.

## AtomTD Schema

The following sections define the complete schema for `AtomTD`. Each namespace groups related fields according to their physical meaning and usage patterns. Fields marked as optional may be absent depending on the dataset or task. All shape annotations use `N` for the number of atoms, `E` for the number of bonds, `A` for angles, `D` for dihedrals, `B` for batch size, and `D` for hidden dimensionality.

### Atomic Properties (`atoms.*`)

The `atoms` namespace contains per-atom properties, including both input features (such as atomic numbers and positions) and learned representations (such as hidden states). Positions in `atoms.xyz` are mutable and support gradient computation for force prediction.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `atoms.Z` | `[N]` | int64 | Atomic numbers |
| `atoms.xyz` | `[N, 3]` | dtype | Atomic positions (mutable) |
| `atoms.batch` | `[N]` | int64 | Molecule indices for each atom |
| `atoms.v` | `[N, 3]` | dtype | Atomic velocities (optional) |
| `atoms.f` | `[N, 3]` | dtype | Atomic forces (optional, target) |
| `atoms.q` | `[N]` | dtype | Atomic charges (optional) |
| `atoms.type` | `[N]` | int64 | Atom types (optional) |
| `atoms.h` | `[N, D]` | dtype | Hidden states (invariant) |
| `atoms.h_eq` | `[N, D, 3]` | dtype | Equivariant hidden states (optional) |
| `atoms.h_sph` | `[N, D]` | dtype | Spherical harmonic features (optional) |

### Bond Topology (`bonds.*`)

The `bonds` namespace stores bond connectivity and derived geometric features. Unlike v1.0, source and target indices are stored in separate fields (`bonds.i` and `bonds.j`) rather than stacked into a single tensor. This design simplifies indexing and aligns with the naming convention used in entity-level representations.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `bonds.i` | `[E]` | int64 | Bond source atom indices |
| `bonds.j` | `[E]` | int64 | Bond target atom indices |
| `bonds.vec` | `[E, 3]` | dtype | Bond vectors (j - i) |
| `bonds.dist` | `[E]` | dtype | Bond distances |
| `bonds.type` | `[E]` | int64 | Bond types (optional) |

### Angle Topology (`angles.*`)

The `angles` namespace represents three-body interactions. Atom indices are stored separately as `i`, `j`, and `k`, where `j` is the center atom.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `angles.i` | `[A]` | int64 | First atom index |
| `angles.j` | `[A]` | int64 | Center atom index |
| `angles.k` | `[A]` | int64 | Third atom index |
| `angles.theta` | `[A]` | dtype | Angle values (radians) |
| `angles.type` | `[A]` | int64 | Angle types (optional) |

### Dihedral Topology (`dihedrals.*`)

The `dihedrals` namespace represents four-body torsional interactions. Atom indices are stored separately as `i`, `j`, `k`, and `l`, following the same convention as the `angles` namespace.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `dihedrals.i` | `[D]` | int64 | First atom index |
| `dihedrals.j` | `[D]` | int64 | Second atom index |
| `dihedrals.k` | `[D]` | int64 | Third atom index |
| `dihedrals.l` | `[D]` | int64 | Fourth atom index |
| `dihedrals.phi` | `[D]` | dtype | Dihedral values (radians) |
| `dihedrals.type` | `[D]` | int64 | Dihedral types (optional) |

### Pair Topology (`pairs.*`)

The `pairs` namespace represents neighbor lists. Atom indices are stored separately as `i` and `j`, where `j` is the neighbor atom.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `pairs.i` | `[E]` | int64 | Source atom index |
| `pairs.j` | `[E]` | int64 | Neighbor atom index |
| `pairs.diff` | `[E, 3]` | dtype | Neighbor vector (i - j) |
| `pairs.dist` | `[E]` | dtype | Neighbor distance |

### Molecular Targets (`target.*`)

The `target` namespace contains molecular-level properties used for training and evaluation. Unlike earlier conventions, target fields do not use a `y_` prefix, and model predictions write to the same keys as the ground truth. This alignment eliminates the need for key translation during loss computation and simplifies model output handling.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `target.energy` | `[B]` | dtype | Molecular energy |
| `target.dipole` | `[B, 3]` | dtype | Dipole moment (optional) |
| `target.stress` | `[B, 3, 3]` | dtype | Stress tensor (optional) |

## Naming Conventions

Namespace names are lowercase and use either plural forms (such as `atoms`, `bonds`, `angles`) or semantic identifiers (such as `graph`, `target`). This convention makes namespace keys self-documenting and consistent across the codebase. Namespaces are accessed using TensorDict's hierarchical indexing syntax, either as `td["atoms", "x"]` or equivalently as `td["atoms"]["x"]`. The two-argument form is preferred for clarity when the namespace is already known.

Field names within a namespace use lowercase with underscore separators, such as `h_sph` for spherical harmonic features or `type_logits` for type prediction outputs. Target fields in the `target` namespace do not use a `y_` prefix, so molecular energy is stored as `target.energy` rather than `target.y_energy`. Model predictions write to the same keys as the ground truth, avoiding the need for separate prediction namespaces or output key translation.

Topology indices are stored in separate fields rather than stacked tensors. For bonds and neighbor pairs, the source and target atom indices are accessed as `bonds.i`/`bonds.j` and `pairs.i`/`pairs.j` respectively, rather than indexing into a single `bonds.i[2, E]` tensor. This design simplifies code that constructs or queries topology and aligns with entity-level naming conventions used in MolPy. Index field names use semantic identifiers: `i` for the source atom, `j` for the target or center atom, `k` for the third atom in angles, and `l` for the fourth atom in dihedrals.

## Key Alignment Between Targets and Predictions

One of the most important design decisions in the schema is that model predictions write to the same keys as the ground truth targets. This eliminates a common source of bugs and boilerplate in training loops, where manually maintained mappings between target keys and prediction keys can drift out of sync. With identical keys, loss computation becomes a direct tensor operation:

```python
# Target
atomic_td["target", "energy"]  # Ground truth

# Prediction
pred_td["target", "energy"]  # Model output

# Loss computation (direct!)
loss = (pred_td["target", "energy"] - atomic_td["target", "energy"]).pow(2).mean()
```

## Immutability and Gradient Requirements

Different namespaces have different mutability semantics. The `atoms.xyz` field is mutable and typically requires gradients for force prediction, since forces are computed as the negative gradient of energy with respect to positions. All topology fields under `bonds.*`, `pairs.*`, `angles.*`, and `dihedrals.*` are immutable after construction, reflecting the fact that molecular topology does not change during a single forward pass or trajectory segment. These namespaces can be explicitly locked using `atomic_td["bonds"].lock_()` and `atomic_td["pairs"].lock_()` to enforce immutability at the TensorDict level and prevent accidental modification.

When using models with a `ForceHead` module, the `atoms.xyz` field must have gradients enabled before the forward pass:

```python
atomic_td["atoms", "xyz"].requires_grad = True
output_td = model(atomic_td)  # ForceHead computes F = -dE/dxyz
```

## Example Usage

The following example demonstrates how to create an `AtomTD` instance for a water molecule, access fields through namespaces, and query batch-level properties:

```python
from molix.data import AtomTD, Config

# Create
atomic_td = AtomTD.create(
    Z=torch.tensor([8, 1, 1]),  # O, H, H
    xyz=torch.randn(3, 3),
    batch=torch.tensor([0, 0, 0]),
    bond_i=torch.tensor([0, 1, 0]),  # Separate i
    bond_j=torch.tensor([1, 2, 2]),  # Separate j
    energy=torch.tensor([0.5]),  # target.energy
    config=Config(dtype=torch.float32),
)

# Access
Z = atomic_td["atoms", "Z"]
energy = atomic_td["target", "energy"]

# Properties
num_atoms = atomic_td.num_atoms
num_bonds = atomic_td.num_bonds
```

## Migration from v1.0

The v2.0 schema introduces several breaking changes to improve consistency and simplify downstream code. The most visible change is the shift from stacked index tensors to separate index fields. In v1.0, bond indices were stored as a single tensor with shape `[2, E]`, requiring users to slice along the first dimension to extract source and target indices:
```python
bonds_i = td["bonds", "i"]  # Shape: [2, E]
src = bonds_i[0]
dst = bonds_i[1]
```

**After** (separate):
```python
src = td["bonds", "i"]  # Shape: [E]
dst = td["bonds", "j"]  # Shape: [E]
```

In v2.0, source and target indices are stored in dedicated fields, eliminating the need for slicing and making the code more readable.

The second major change is the introduction of the `target` namespace, which replaces the earlier `mol` namespace and removes the `y_` prefix from target fields. In v1.0, molecular energy was accessed as:
```python
energy = td["mol", "y_energy"]
```

**After**:
```python
energy = td["target", "energy"]
```

This change makes target field names consistent with prediction field names, since both now use the same keys within the `target` namespace.

Finally, v2.0 requires an explicit `Config` object to be passed during `AtomTD` creation, ensuring that dtype and device settings are consistently applied across all fields. In v1.0, the config was inferred or defaulted:
```python
atomic_td = AtomTD.create(Z=..., xyz=..., batch=...)
```

**After**:
```python
config = Config(dtype=torch.float32)
atomic_td = AtomTD.create(Z=..., xyz=..., batch=..., config=config)
```
