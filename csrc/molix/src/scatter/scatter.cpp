// CPU streaming scatter-add along dim 0.
//
// ``out[index[e]] += src[e]`` for ``e`` in ``[0, E)``. Output is
// ``(dim_size, *src.shape[1:])``; there is no ``(N, tile)`` workspace.
// Same primitive as pytorch_scatter ``scatter_add`` on unsorted COO.
// Backward is gather: ``grad_src = grad_out.index_select(0, index)``.

#include <torch/all.h>
#include <torch/library.h>

using torch::Tensor;

namespace {

Tensor scatter_sum_forward(const Tensor& src, const Tensor& index, int64_t dim_size) {
    TORCH_CHECK(src.dim() >= 1, "scatter_sum: src must be at least 1-D");
    TORCH_CHECK(index.dim() == 1, "scatter_sum: index must be 1-D");
    TORCH_CHECK(
        index.size(0) == src.size(0),
        "scatter_sum: index.length must equal src.size(0), got ",
        index.size(0),
        " vs ",
        src.size(0)
    );
    TORCH_CHECK(dim_size >= 0, "scatter_sum: dim_size must be >= 0");
    auto sizes = src.sizes().vec();
    sizes[0] = dim_size;
    Tensor out = torch::zeros(sizes, src.options());
    if (src.size(0) == 0) {
        return out;
    }
    return out.index_add_(0, index, src);
}

}  // namespace

TORCH_LIBRARY_IMPL(molix, CPU, m) {
    m.impl("scatter_sum", &scatter_sum_forward);
}
