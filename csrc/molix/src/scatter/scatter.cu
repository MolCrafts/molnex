// CUDA streaming scatter-add along dim 0.
//
// Grid-stride over source rows; each row atomically adds into
// ``out[index[e]]``. Workspace is the output tensor only (pytorch_scatter
// ``scatter_add`` on unsorted COO). 64-bit addressing: ``E * D`` can exceed
// 2^31 on large molecular graphs.

#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using torch::Tensor;

namespace {

template <typename scalar_t>
__global__ void scatter_sum_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ out,
    int64_t n_src,
    int64_t feat
) {
    for (int64_t e = static_cast<int64_t>(blockIdx.x); e < n_src;
         e += static_cast<int64_t>(gridDim.x)) {
        const int64_t dst = index[e];
        const scalar_t* row = src + e * feat;
        scalar_t* out_row = out + dst * feat;
        for (int64_t d = static_cast<int64_t>(threadIdx.x); d < feat;
             d += static_cast<int64_t>(blockDim.x)) {
            gpuAtomicAdd(out_row + d, row[d]);
        }
    }
}

Tensor scatter_sum_cuda_forward(const Tensor& src, const Tensor& index, int64_t dim_size) {
    TORCH_CHECK(src.is_cuda(), "scatter_sum: src must be CUDA");
    TORCH_CHECK(src.dim() >= 1, "scatter_sum: src must be at least 1-D");
    TORCH_CHECK(index.dim() == 1, "scatter_sum: index must be 1-D");
    TORCH_CHECK(
        index.size(0) == src.size(0),
        "scatter_sum: index.length must equal src.size(0)"
    );
    TORCH_CHECK(dim_size >= 0, "scatter_sum: dim_size must be >= 0");

    const auto stream = getCurrentCUDAStream(src.get_device());
    const CUDAStreamGuard guard(stream);

    Tensor src_c = src.contiguous();
    Tensor index_c = index.contiguous().to(torch::kLong);
    auto sizes = src_c.sizes().vec();
    sizes[0] = dim_size;
    Tensor out = torch::zeros(sizes, src_c.options());
    const int64_t n_src = src_c.size(0);
    if (n_src == 0) {
        return out;
    }
    int64_t feat = 1;
    for (int64_t d = 1; d < src_c.dim(); ++d) {
        feat *= src_c.size(d);
    }
    const int block = 128;
    const int grid = static_cast<int>(std::min(n_src, static_cast<int64_t>(4096)));

    AT_DISPATCH_V2(
        src_c.scalar_type(),
        "molix::scatter_sum",
        AT_WRAP([&] {
            scatter_sum_kernel<scalar_t><<<grid, block, 0, stream>>>(
                src_c.data_ptr<scalar_t>(),
                index_c.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(),
                n_src,
                feat
            );
        }),
        AT_EXPAND(AT_FLOATING_TYPES)
    );
    return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(molix, CUDA, m) {
    m.impl("scatter_sum", &scatter_sum_cuda_forward);
}
