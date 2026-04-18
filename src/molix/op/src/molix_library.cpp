// Central schema registration for the `molix` torch custom op library.
//
// Every op exposed via `torch.ops.molix.*` declares its schema here so that
// new contributors have a single place to look when adding or renaming an op.
// Device-specific implementations live alongside their kernels and register
// via TORCH_LIBRARY_IMPL(molix, ...).
//
// Contract for new ops:
//   1. Add the schema in this file.
//   2. Provide AutogradCPU + AutogradCUDA + Meta impls in src/<domain>/.
//   3. Add a thin Python wrapper in src/molix/F/<domain>.py that handles
//      dtype / shape assertions before calling torch.ops.molix.<name>.

#include <torch/extension.h>

TORCH_LIBRARY(molix, m) {
    // ── Geometry ───────────────────────────────────────────────────────
    // O(N^2) pair enumeration with optional triclinic PBC. Returns a fixed-
    // width (2, max_num_pairs) view padded with -1 / NaN beyond the actual
    // count so shapes stay static for downstream ops. `num_pairs` reports
    // the actual count within cutoff (self-interactions excluded).
    m.def(
        "get_neighbor_pairs("
        "Tensor positions, Scalar cutoff, Scalar max_num_pairs, Tensor box_vectors"
        ") -> (Tensor neighbors, Tensor deltas, Tensor distances, Tensor num_pairs)"
    );

    // ── Electrostatics: PME ────────────────────────────────────────────
    m.def(
        "pme_direct("
        "Tensor positions, Tensor charges, Tensor neighbors, Tensor deltas, "
        "Tensor distances, Tensor exclusions, Scalar alpha, Scalar coulomb"
        ") -> Tensor"
    );
    m.def(
        "pme_reciprocal("
        "Tensor positions, Tensor charges, Tensor box_vectors, "
        "Scalar gridx, Scalar gridy, Scalar gridz, Scalar order, "
        "Scalar alpha, Scalar coulomb, "
        "Tensor xmoduli, Tensor ymoduli, Tensor zmoduli"
        ") -> Tensor"
    );
}
