// Meta-tensor implementations for `molix` ops.
//
// Meta kernels return empty tensors with correct shape/dtype but no data.
// They are required for torch.compile / FakeTensor tracing — without them
// the compiler cannot trace through `torch.ops.molix.*` calls and will
// either fall back to eager execution (losing fusion opportunities) or
// raise "No fake impl registered" errors in strict modes.
//
// Keep this file in sync with the schema in `molix_library.cpp`.

#include <torch/extension.h>
#include <tuple>

using torch::Scalar;
using torch::Tensor;

namespace {

std::tuple<Tensor, Tensor, Tensor, Tensor> get_neighbor_pairs_meta(
    const Tensor& positions,
    const Scalar& /*cutoff*/,
    const Scalar& max_num_pairs,
    const Tensor& /*box_vectors*/
) {
    const int64_t n = positions.sym_size(0).guard_int(__FILE__, __LINE__);
    const int64_t mnp = max_num_pairs.toLong();
    const int64_t width = (mnp == -1) ? n * (n - 1) / 2 : mnp;

    auto pos_opts = positions.options();
    auto int_opts = pos_opts.dtype(torch::kInt32);
    return {
        torch::empty({2, width}, int_opts),
        torch::empty({width, 3}, pos_opts),
        torch::empty({width}, pos_opts),
        torch::empty({1}, int_opts),
    };
}

Tensor pme_direct_meta(
    const Tensor& positions,
    const Tensor& /*charges*/,
    const Tensor& /*neighbors*/,
    const Tensor& /*deltas*/,
    const Tensor& /*distances*/,
    const Tensor& /*exclusions*/,
    const Scalar& /*alpha*/,
    const Scalar& /*coulomb*/
) {
    return torch::empty({}, positions.options());
}

Tensor pme_reciprocal_meta(
    const Tensor& positions,
    const Tensor& /*charges*/,
    const Tensor& /*box_vectors*/,
    const Scalar& /*gridx*/, const Scalar& /*gridy*/, const Scalar& /*gridz*/,
    const Scalar& /*order*/,
    const Scalar& /*alpha*/, const Scalar& /*coulomb*/,
    const Tensor& /*xmoduli*/, const Tensor& /*ymoduli*/, const Tensor& /*zmoduli*/
) {
    return torch::empty({}, positions.options());
}

}  // namespace

TORCH_LIBRARY_IMPL(molix, Meta, m) {
    m.impl("get_neighbor_pairs", &get_neighbor_pairs_meta);
    m.impl("pme_direct", &pme_direct_meta);
    m.impl("pme_reciprocal", &pme_reciprocal_meta);
}
