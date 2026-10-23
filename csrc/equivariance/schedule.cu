// Edge schedule: the CSR views of an edge list by target and by source.
//
// A stable narrow-key radix sort (NVIDIA CUB, ``ceil(log2(rows))`` bits, one or
// two digit passes) orders the edges; a binary search over the sorted keys
// gives the row offsets. Integer-only, deterministic, no host synchronisation,
// CUDA-graph capturable. Built once per batch and shared by every contraction
// of every layer, forward and adjoints.
#include <ATen/ATen.h>

#include <algorithm>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_radix_sort.cuh>

namespace molnex::equivariance {
namespace {

__global__ void row_offsets(const int32_t* __restrict__ keys, int64_t edges, int64_t rows,
                            int32_t* __restrict__ rowptr) {
  const int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n > rows) return;
  // first position with key >= n
  int64_t lo = 0, hi = edges;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (keys[mid] < n) lo = mid + 1; else hi = mid;
  }
  rowptr[n] = int32_t(lo);
}

void csr(const at::Tensor& index, int64_t rows, at::Tensor& order, at::Tensor& rowptr) {
  const int64_t edges = index.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto keys = index.to(at::kInt);
  auto sorted = at::empty_like(keys);
  if (edges > 0) {
    auto iota = at::arange(edges, keys.options());
    int bits = 1;
    while ((int64_t(1) << bits) < rows) ++bits;
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys.const_data_ptr<int32_t>(), sorted.mutable_data_ptr<int32_t>(),
        iota.const_data_ptr<int32_t>(), order.mutable_data_ptr<int32_t>(), int(edges), 0, bits,
        stream));
    auto temp = at::empty({int64_t(bytes)}, keys.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.mutable_data_ptr(), bytes, keys.const_data_ptr<int32_t>(),
        sorted.mutable_data_ptr<int32_t>(), iota.const_data_ptr<int32_t>(),
        order.mutable_data_ptr<int32_t>(), int(edges), 0, bits, stream));
  }
  const int64_t threads = 256, blocks = (rows + 1 + threads - 1) / threads;
  row_offsets<<<unsigned(blocks), unsigned(threads), 0, stream>>>(
      sorted.const_data_ptr<int32_t>(), edges, rows, rowptr.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule_cuda(
    const at::Tensor& source, const at::Tensor& target, int64_t rows_a, int64_t rows_d) {
  const c10::cuda::CUDAGuard guard(source.device());
  TORCH_CHECK(source.is_cuda() && source.scalar_type() == at::kLong && source.dim() == 1 &&
                  source.is_contiguous(),
              "source must be a contiguous CUDA int64 vector");
  TORCH_CHECK(target.device() == source.device() && target.scalar_type() == at::kLong &&
                  target.dim() == 1 && target.is_contiguous() && target.numel() == source.numel(),
              "target must match source");
  TORCH_CHECK(rows_a >= 0 && rows_d >= 0, "row counts must be non-negative");
  TORCH_CHECK(source.numel() < std::numeric_limits<int32_t>::max() &&
                  std::max(rows_a, rows_d) < std::numeric_limits<int32_t>::max(),
              "edge schedule is 32-bit");
  const auto options = source.options().dtype(at::kInt);
  auto order_t = at::empty({source.numel()}, options);
  auto order_s = at::empty({source.numel()}, options);
  auto rowptr_t = at::empty({rows_d + 1}, options);
  auto rowptr_s = at::empty({rows_a + 1}, options);
  csr(target, rows_d, order_t, rowptr_t);
  csr(source, rows_a, order_s, rowptr_s);
  return {order_t, rowptr_t, order_s, rowptr_s};
}

}  // namespace molnex::equivariance
