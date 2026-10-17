# NVIDIA/cuEquivariance issue #309 — filed 2026-10-16

Status: FILED 2026-10-16 as https://github.com/NVIDIA/cuEquivariance/issues/309 (text below, unchanged).
Operator approved the text; request only, no reproduction.

Follow-ups stay requests: no internal symbol names, nothing about `csrc/cueq`
(internal only; see csrc/cueq/INTERNAL.md).

Checked against cuequivariance / cuequivariance-ops-torch 0.13.0 (2026-10-15):
every item below is still open there.

---

**Title:** Feature request: native C++ registration of the `cuequivariance` torch ops for Python-free AOTInductor deployment

**Request**

Please ship native (C++ `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL(..., CUDA, ...)`)
registrations of the `cuequivariance::*` torch operators, under the same names
and schemas as the current Python `torch.library.custom_op` definitions, in a
shared library that a C++ process can load (`dlopen` /
`torch::...load_library`) — so that a model exported with `torch.export` and
packaged with AOTInductor runs through `AOTIModelPackageLoader` without a Python
interpreter.

The operators an interatomic potential needs, forward and backward (forces are
taken by autograd inside the exported graph):

- `uniform_1d`
- `fused_tensor_product`, `fused_tensor_product_bwd`, `fused_tensor_product_bwd_bwd`
- `indexed_linear_B`, `indexed_linear_C`

If shipping the registrations is not possible, a stable, versioned public C API
(headers) for these kernels would let downstream projects register them
themselves.

Two smaller requests on the same path:

- Register fake (meta) implementations for `indexed_linear_B` and
  `indexed_linear_C`, so `torch.compile` / `torch.export` can trace them.
- Make `cuet.Linear(..., method="indexed_linear")` traceable: replace the
  host-side sortedness `assert` and the data-dependent `torch.bincount` with a
  fixed-size count and `torch._check`.

Context: AOTInductor is PyTorch's path for running models from C++, and the
only one left now that TorchScript is unsupported here (#30, #137); the ML-IAP
route embeds Python in the MD engine. Related: #34 (C-level API), #137.
