// Central schema registration for the `molix` torch custom op library.
//
// Every op exposed via `torch.ops.molix.*` declares its schema here so that
// new contributors have a single place to look when adding or renaming an op.
// Device-specific implementations live alongside their kernels and register
// via TORCH_LIBRARY_IMPL(molix, ...).
//
// Contract for new ops:
//   1. Add the schema in this file.
//   2. Provide CPU + CUDA + Meta impls in src/<domain>/.
//   3. Register autograd in the Python wrapper (``torch.library.register_autograd``)
//      so dynamo does not trace into CUDA ``data_ptr()``.
//   4. Add a thin Python wrapper in src/molix/F/<domain>.py that handles
//      dtype / shape assertions before calling torch.ops.molix.<name>.

#include <torch/all.h>
#include <torch/library.h>

// What a deployed package records for this library (molix.deploy metadata) and
// what molnex's C++ runtime checks before loading it: the version of the
// operator schemas below. Bump it whenever a schema changes.
extern "C" const char* molnex_op_library_requirement() { return "molix_ops==1"; }

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

    // ── Scatter ────────────────────────────────────────────────────────
    // Streaming scatter-add along dim 0: ``out[index[e]] += src[e]``.
    // Workspace is the output ``(dim_size, *src.shape[1:])`` only — no
    // ``(N, tile)`` scratch. Same primitive as pytorch_scatter ``scatter_add``
    // (unsorted COO); backward is gather ``grad_out[index]``. ``dim_size`` is
    // a SymInt so an exported graph keeps the output row count symbolic.
    m.def("scatter_sum(Tensor src, Tensor index, SymInt dim_size) -> Tensor");

}
