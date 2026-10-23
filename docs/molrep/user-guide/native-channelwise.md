# Native channelwise tensor product

`molrep.equivariance.ChannelWiseTensorProduct` is molnex's own CUDA
implementation of the indexed channelwise tensor product that MACE's
interaction blocks evaluate — the same signature as
`molrep.eq.torch.ChannelWiseTensorProduct` restricted to the subset MACE
uses, and an explicit opt-in: MACE constructs cuEquivariance's operation by
default. The frontend selects it with `method="native"`
(`molrep.eq.torch.ChannelWiseTensorProduct` or `molrep.cueq.ChannelWiseTensorProduct`).
`uniform_1d`, `fused_tp` and `indexed_linear` stay on `torch.ops.cuequivariance`.

## What it computes

For edges `e = (source, target)`, node features `a` (`ir_mul`, uniform
multiplicity `U`), edge harmonics `b` and per-edge weights `w`:

```
out[target, (l3 m3), c] += C(l1 m1, l2 m2 | l3 m3) · a[source, (l1 m1), c] · b[e, (l2 m2)] · w[e, path, c]
```

summed over every path `(l1, l2, l3)` of cuEquivariance's
`channelwise_tensor_product` descriptor. Supported: O3, `ir_mul`, external
unshared weights, `irreps_in1 = U×0e + U×1o + … + U×L1` (natural parity,
`L1 ≤ 3`, `U ∈ {32, 64, 128}`), `irreps_in2 = 0e + 1o + … + L2`
(`L2 ∈ {2, 3}`) and the output filter `0e + … + L2`. Anything else raises at
construction; there is no cuEquivariance fallback.

## How it is built

- **Term tables at compile time.** The positions of all non-zero
  Clebsch–Gordan coefficients are enumerated at compile time
  (`csrc/equivariance/terms.hpp`) from the selection rule of the real
  spherical-harmonic basis. The coefficient *values* are read off the
  cuEquivariance descriptor when the module is constructed and passed to the
  kernels as parameters, so the contraction is a fully unrolled sequence of
  FMAs on register-resident operands. The descriptor check refuses any basis
  convention the rule does not cover.
- **Edge schedule instead of atomics.** `ChannelWiseTensorProduct.schedule`
  builds the CSR views of the edge list by target and by source (a stable
  narrow-key radix sort). The product sums a node's incoming edges in
  registers and writes each row once; the node-operand adjoint sums outgoing
  edges the same way. No atomics, so results are bitwise reproducible and
  PyTorch's deterministic mode is honoured. The schedule is built once per
  batch; the MACE encoder shares it across its layers, forward and adjoints.
- **Fused adjoints.** The backward of the product is one pass by target that
  returns the gradients of all three operands; its backward is the same kernel
  with the cotangents as further operands, so force training's second
  derivatives are exact and one launch. Third derivatives are not implemented.

## Using it

```python
import molrep.eq as meq
import torch
from molrep.equivariance import ChannelWiseTensorProduct

tp = ChannelWiseTensorProduct(
    meq.Irreps("O3", "128x0e+128x1o"),
    meq.Irreps("O3", "0e+1o+2e+3o"),
    meq.Irreps("O3", "128x0e+128x1o+128x2e+128x3o"),
    layout=meq.ir_mul, shared_weights=False, internal_weights=False,
    dtype=torch.float32, device="cuda",
)
schedule = tp.schedule(source, target, num_nodes, num_nodes)   # once per batch
out = tp(node_feats, edge_sh, weights, indices_1=source, indices_out=target,
         size_out=num_nodes, schedule=schedule)
```

Build the library with `-DMOLNEX_EQUIVARIANCE_ENABLE_CUDA=ON` (see
`csrc/CMakeLists.txt`) and point `MOLNEX_EQUIVARIANCE_LIBRARY` at
`libmolnex_equivariance.so`, or install it as `molrep/lib/`. The op library is
LibTorch-only, with Meta kernels, so an AOTInductor package that captured the
product and its adjoints runs in a process without Python.

Measured standing against cuEquivariance 0.12 on a GH200 (MACE-MatPES,
same-job pairs) is recorded in `.claude/notes/notes.md`, topic
`native-channelwise-csr`.
