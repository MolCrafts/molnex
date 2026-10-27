# Equivariant operators

`molrep.equivariance` holds every equivariant contraction molnex's models
evaluate — on molnex's own kernels. cuEquivariance survives as the group theory
and the descriptors (`molrep.eq`, the ported Apache-2.0 frontend); no
cuEquivariance kernel runs anywhere in molnex.

| Module | What | Used by |
|---|---|---|
| `ChannelWiseTensorProduct` | the indexed channelwise product of MACE's interaction blocks | MACE |
| `SymmetricContraction` | the symmetric contraction of MACE's product block | MACE |
| `SphericalHarmonics` | real spherical harmonics, degrees 0..3 | MACE, Allegro (via `molrep.embedding.angular`) |
| `Contracter` | Allegro's path-channel-coupled `u,iu,ju,ku+ijk` product | Allegro |
| `WeightedChannels` | Allegro's per-(channel, irrep) weighting with the environment sum | Allegro |
| `Linear` | the equivariant linear map, one GEMM per irrep block | every model |

Each module reads its term table — one scalar monomial per non-zero
coefficient — off cuEquivariance's descriptor at construction, and evaluates
it in one of two ways, chosen by the device of the tensor it is given:

- **CUDA**: `libmolnex_equivariance`, molnex's operator library
  (`csrc/equivariance/`, namespace `molnex_equivariance`). Every reduction
  runs in a fixed order — no atomics — so results are bitwise reproducible
  and PyTorch's deterministic mode is honoured. Each operator registers its
  first adjoint as one fused pass and that adjoint's adjoint as the same
  kernel with the cotangents as further operands, so force training's second
  derivatives are exact. Third derivatives are not implemented. Meta kernels
  make the library usable from an AOTInductor package in a process without
  Python (`interface/runtime` loads it by the name `molnex_equivariance`).
- **CPU**: `molrep.equivariance.reference`, the same term table in plain
  PyTorch — the oracle the kernels are tested against. Autograd is PyTorch's
  own. It is correct, not fast: a CPU evaluation serves tests and
  correctness checks.

There is no `method=` switch and no cuEquivariance fallback.

## Precision and layout

Everything is `ir_mul`. The precision is fixed at construction (`dtype=`):
build at the wanted dtype rather than casting afterwards; an input of another
dtype is refused.

## The channelwise product

```
out[target, (l3 m3), c] += C(l1 m1, l2 m2 | l3 m3) · a[source, (l1 m1), c] · b[e, (l2 m2)] · w[e, path, c]
```

summed over every path `(l1, l2, l3)` of cuEquivariance's
`channelwise_tensor_product` descriptor for MACE's natural-parity family:
`irreps_in1 = U×0e + U×1o + … + U×L1`, `irreps_in2 = 0e + 1o + … + L2` and the
output filter `0e + … + L2`. Weights are external and per edge. The term table
is built for any uniform `U` and any `(L1, L2)`; the CUDA kernels are
instantiated for `L1 ≤ 3` with `L2 ∈ {2, 3}` and for `L1 ≤ 1` with `L2 = 1`
(the tiny `l_max = 1` test models), at widths `U ∈ {8, 16, 32, 64, 128}` or
any multiple of 128 (walked in tiles of 128 by one CTA: OMol's first layer is
1024 wide).

- **Term tables at compile time.** The positions of all non-zero
  Clebsch–Gordan coefficients are enumerated at compile time
  (`csrc/equivariance/terms.hpp`) from the selection rule of the real
  spherical-harmonic basis; the coefficient *values* are read off the
  descriptor at construction and passed to the kernels as parameters, so the
  contraction is a fully unrolled sequence of FMAs on register-resident
  operands. The descriptor check refuses any basis convention the rule does
  not cover.
- **Edge schedule instead of atomics.** `ChannelWiseTensorProduct.schedule`
  builds the CSR views of the edge list by target and by source (a stable
  narrow-key radix sort). The product sums a node's incoming edges in
  registers and writes each row once; the node-operand adjoint sums outgoing
  edges the same way. The MACE encoder builds the schedule once per batch
  and shares it across its layers, forward and adjoints.

```python
import molrep.eq as meq
import torch
from molrep.equivariance import ChannelWiseTensorProduct

tp = ChannelWiseTensorProduct(
    meq.Irreps("O3", "128x0e+128x1o"),
    meq.Irreps("O3", "0e+1o+2e+3o"),
    meq.Irreps("O3", "128x0e+128x1o+128x2e+128x3o"),
    dtype=torch.float32, device="cuda",
)
schedule = tp.schedule(source, target, num_nodes, num_nodes)   # once per batch
out = tp(node_feats, edge_sh, weights, indices_1=source, indices_out=target,
         size_out=num_nodes, schedule=schedule)
```

## The symmetric contraction

After the original-MACE projection `einsum("zau,ab->zbu", weight, projection)`
— the module owns both, in MACE's layout, so an official checkpoint loads
unchanged — the kernel evaluates the standard polynomial. For each node `n`
and channel `c`:

```
y[n, o, c] += C · W[index[n], w, c] · x[n, i, c] · x[n, j, c]? · x[n, k, c]?
```

Degrees 1..3, one multiplicity shared by the input and the output, at most 64
input or output components and 1023 weight segments.

- **Merged monomials.** A monomial is symmetric in its slots, so the
  descriptor's paths that differ only by a permutation of `(i, j, k)` are one
  term: slots sorted, coefficients summed, exact zeros dropped. At MACE's
  shapes this is 4–4.6× fewer terms than cuEquivariance evaluates
  (5520 → 1203 for the OMol-shaped product, 2039 → 480 for MatPES, 370 → 94
  for the scalar layer), exactly.
- **Compiled tables for MACE's family.** For natural-parity irreps with
  `l_out ≤ l_in ≤ 3` and degree 2 or 3 the merged table is compiled in
  (`csrc/equivariance/contraction_tables.hpp`, written by
  `generate_contraction_tables.py`; 18 tables, 480 terms for MatPES's first
  layer, 1203 for the OMol-shaped product). The kernels hold the node row, the
  element's weight row and every accumulator in registers, unroll the table
  with the coefficient values as a kernel parameter, and reuse the product of
  consecutive terms sharing `(i, j)`. The module compares the table it reads
  off its own descriptor with the compiled one before selecting it.
- **The generic table otherwise.** Any other shape runs a shared-memory
  kernel over the packed table (one `int32` per term beside its coefficient),
  sorted by output segment, with a sentinel segment whose value is 1 (and
  whose cotangent is 0) standing in for a missing slot.
- **Adjoints.** The feature adjoint is local to the node. The weight adjoint
  is written per node and summed into the elements: nodes grouped by element
  with a stable sort, one CTA per (element, chunk) adding its nodes in order,
  then the chunks from first to last. Second derivatives are the product
  rule on the same table.

## The spherical harmonics

The monomial table of cuEquivariance's `SO3` convention, degrees 0..3
(`l = 1` is `sqrt(3)` times the Cartesian components). Normalisation of the
direction, when wanted, is `molrep.embedding.angular.SphericalHarmonics`'s
job; the polynomial evaluates the vectors it is given.

## The Allegro contracter and weighted channels

`Contracter` evaluates the `u,iu,ju,ku+ijk` product of one Allegro layer,

```
y[e, (q3 m3), c] += W[p, c] · Σ C(l1 m1, l2 m2 | l3 m3) · x1[e, (q1 m1), c] · x2[index[e], (q2 m2), c]
```

with one shared weight row `(paths, channels)`.

- **Compiled tables for Allegro's layers.** The 21 `(irreps_in1, irreps_in2,
  irreps_out)` triples an Allegro model builds (`l_max ≤ 3`, both parities, up
  to five layers; `generate_contracter_tables.py` →
  `contracter_tables.hpp`, 353 terms for an Allegro-OAM-L layer) are compiled
  in: the kernels hold the weight row across a chunk of edges and the pair
  row, the gathered atom row and the cotangents per edge in registers, and
  unroll the table with the coefficient values as a parameter. The module
  compares its own flattened table with the compiled ones before selecting
  one.
- **Clebsch–Gordan classes otherwise.** Any other triple runs one path at a
  time: a path is three irreps and a coupling class `(l1, l2, l3)` whose
  coefficient positions are compiled (`csrc/equivariance/cg.hpp` — every
  coupling with `l ≤ 3` the triangle rule allows, 34 classes, from the
  real-basis selection rule including the antisymmetric odd-sum couplings;
  verified against cuEquivariance's tables) and whose values arrive as a
  kernel parameter, dispatched by a switch over the classes on blocks of at
  most seven registers, with the rows staged in shared memory per edge.
- **Adjoints.** The pair adjoint is local to the edge; the atom adjoint is
  written per edge and summed into its atom in a fixed order; the weight
  adjoint accumulates per CTA and the CTAs are added in order. The second
  order runs the class-blocked kernel even for a compiled table (its
  register footprint is beyond what the compiled body keeps in registers).
  The flattened term table (one scalar per non-zero coefficient) remains the
  CPU reference and the test oracle.

`WeightedChannels` evaluates `y[e, d, c] += basis[e, d] · W[e, r(d), c]`,
one straight loop over the components, and, with `index` / `num_rows`, sums
the rows into the atoms so the `(E, D · u)` message is never built. The
Allegro layers (`molzoo.allegro.layers`) own the parameters in upstream's
layout and hand these modules the transposed views.

## Building the library

```
cmake -S csrc -B build/csrc-$(uname -m) -DMOLNEX_OP_ENABLE_CUDA=ON \
      -DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch
cmake --build build/csrc-$(uname -m) -j     # -> build/csrc-<arch>/equivariance/libmolnex_equivariance.<arch>.so
```

The loader searches `MOLNEX_EQUIVARIANCE_LIBRARY` (an explicit file), the
directories of `MOLNEX_OP_LIBRARY_PATH` (`scripts/env/activate.sh` sets it)
and `molrep/lib/`, where a wheel installs it. Without a CUDA device nothing
is loaded; with one and no library, construction warns once and CUDA tensors
cannot be evaluated until it is built. A deployed package that calls these
operators records `molnex_equivariance` in its metadata (`molix.deploy`), and
the C++ runtime loads the library before the package.

Measured standing against cuEquivariance's kernels is recorded in
`.claude/notes/notes.md` (topics `native-channelwise-csr` and
`native-equivariance-default`).
