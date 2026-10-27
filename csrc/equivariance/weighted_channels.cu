// Allegro weighted channels. One scalar path per basis component:
//
//   y[n, d, c] += C * basis[e, d] * W[e, r, c]
//
// n is the edge when num_rows < 0, and index[e] otherwise. W is irrep-major,
// one row per edge. The formulas match the product rule checked in
// tests/test_molrep/test_equivariance/test_weighted_channels_terms.py.
//
//   ga[e, d] = sum_c C * W[e, r, c] * g[n, d, c]
//   gw[e, r, c] = sum_{d in r} C * basis[e, d] * g[n, d, c]
//
// Second adjoint, with dga the cotangent of ga and dgw the cotangent of gw.
// da does not read dga, and dw does not read dgw.
//   da[e, d] = sum_c C * g[n, d, c] * dgw[e, r, c]
//   dw[e, r, c] = sum_{d in r} C * dga[e, d] * g[n, d, c]
//   dgy[n, d, c] += C * (W[e, r, c] * dga[e, d] + basis[e, d] * dgw[e, r, c])
//
// Indexed y and dgy are deterministic two-pass sums. No atomics. Edges are
// grouped by atom with a stable radix sort (original order inside an atom)
// and chunked. A second kernel adds the chunks from first to last.
//
// An index outside [0, num_rows) contributes nothing and is not read.
// Third derivatives are refused in Python. Accumulation stays in the compute
// dtype.
//
// Threads run along channels so a weight row is coalesced; a component loop
// inside walks the table in order. ga/da is a CTA sum over channels; that sum
// is the only cross-thread reduction.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace molnex::equivariance {
namespace {

constexpr int kMaxThreads = 128;
constexpr int kMaxSegments = 64;
constexpr int kMaxChunks = 256;
constexpr int64_t kPartialBudget = 32LL << 20;

struct Term {
  int d, r;
};

struct Problem {
  int edges, atoms, components, irreps, channels, nterms;
  bool indexed;
};

__device__ Term load_term(const int16_t* terms, int t) {
  const int16_t* p = terms + static_cast<long long>(t) * 2;
  return {p[0], p[1]};
}

// The term table is one path per component, sorted by component: row ``d`` of
// ``terms`` is ``(d, r(d))``, so the kernels walk ``d`` in order, never index a
// register array at run time, and a per-irrep sum closes when ``r`` changes.

__device__ __forceinline__ int irrep_of(const int16_t* __restrict__ terms, int d) {
  return terms[2 * d + 1];
}

template <typename T>
__global__ void scale_kernel(const T* __restrict__ basis, const T* __restrict__ weight,
                             const int16_t* __restrict__ terms, const T* __restrict__ coeff, T* __restrict__ y,
                             int edges, int components, int irreps, int channels) {
  const int edge = static_cast<int>(blockIdx.x);
  if (edge >= edges) return;
  const T* a = basis + static_cast<long long>(edge) * components;
  const T* w = weight + static_cast<long long>(edge) * irreps * channels;
  T* dst = y + static_cast<long long>(edge) * components * channels;
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    for (int d = 0; d < components; ++d) {
      const int r = irrep_of(terms, d);
      dst[static_cast<long long>(d) * channels + channel] =
          coeff[d] * a[d] * w[static_cast<long long>(r) * channels + channel];
    }
  }
}

// Sum of one value per thread over the CTA, in a fixed order: warp shuffles,
// then the warps. Every thread receives the sum.
template <typename T>
__device__ __forceinline__ T block_sum(T v, T* red) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffffu, v, off);
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  T total = T(0);
  for (int q = 0; q < warps; ++q) total += red[q];
  return total;
}

// kSecond == 0 writes ga and gw. kSecond == 1 writes da, dw, and, when the
// output is not indexed, dgy. ga/da is the CTA sum over channels.
template <typename T, int kSecond, int kIndexed>
__global__ void adjoint_kernel(const T* __restrict__ basis, const T* __restrict__ weight,
                               const int32_t* __restrict__ index, const T* __restrict__ g,
                               const T* __restrict__ dga, const T* __restrict__ dgw,
                               const int16_t* __restrict__ terms, const T* __restrict__ coeff,
                               T* __restrict__ grad_basis, T* __restrict__ grad_weight, T* __restrict__ dgy,
                               int edges, int rows, int components, int irreps, int channels) {
  __shared__ T red[kMaxThreads / 32];
  const int edge = static_cast<int>(blockIdx.x);
  if (edge >= edges) return;
  int atom = edge;
  if constexpr (kIndexed) atom = index[edge];
  const bool valid = atom >= 0 && atom < rows;
  const T* a = basis + static_cast<long long>(edge) * components;
  const T* w = weight + static_cast<long long>(edge) * irreps * channels;
  const T* gv = valid ? g + static_cast<long long>(atom) * components * channels : nullptr;
  const T* dga_row = nullptr;
  const T* dgw_row = nullptr;
  if constexpr (kSecond) {
    dga_row = dga + static_cast<long long>(edge) * components;
    dgw_row = dgw + static_cast<long long>(edge) * irreps * channels;
  }
  T* ga = grad_basis + static_cast<long long>(edge) * components;
  T* gw = grad_weight + static_cast<long long>(edge) * irreps * channels;
  // The channel loop is outermost for the per-channel outputs; the component
  // loop inside closes a weight irrep's sum when ``r`` changes.
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    T wacc = T(0);
    int prev = -1;
    for (int d = 0; d < components; ++d) {
      const int r = irrep_of(terms, d);
      if (r != prev) {
        if (prev >= 0) gw[static_cast<long long>(prev) * channels + channel] = wacc;
        wacc = T(0);
        prev = r;
      }
      if (!valid) continue;
      const T c = coeff[d];
      const T gc = gv[static_cast<long long>(d) * channels + channel];
      if constexpr (kSecond == 0) {
        wacc += c * a[d] * gc;
      } else {
        wacc += c * dga_row[d] * gc;
        if constexpr (!kIndexed) {
          const T dgc = dgw_row[static_cast<long long>(r) * channels + channel];
          dgy[(static_cast<long long>(edge) * components + d) * channels + channel] =
              c * (w[static_cast<long long>(r) * channels + channel] * dga_row[d] + a[d] * dgc);
        }
      }
    }
    if (prev >= 0) gw[static_cast<long long>(prev) * channels + channel] = wacc;
  }
  // ga[d] / da[d]: a sum over the channels, one CTA reduction per component.
  for (int d = 0; d < components; ++d) {
    const int r = irrep_of(terms, d);
    T part = T(0);
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
      if (!valid) break;
      const T gc = gv[static_cast<long long>(d) * channels + channel];
      if constexpr (kSecond == 0) {
        part += coeff[d] * w[static_cast<long long>(r) * channels + channel] * gc;
      } else {
        part += coeff[d] * gc * dgw_row[static_cast<long long>(r) * channels + channel];
      }
    }
    const T total = block_sum(part, red);
    if (threadIdx.x == 0) ga[d] = total;
  }
}

// kSecond == 0 accumulates y. kSecond == 1 accumulates dgy. One CTA owns one
// chunk of one atom and writes that partial in full, including zeros.
template <typename T, int kSecond>
__global__ void scatter_partial_kernel(const T* __restrict__ basis, const T* __restrict__ weight,
                                        const int32_t* __restrict__ index, const int32_t* __restrict__ order,
                                        const int32_t* __restrict__ rowptr, const T* __restrict__ dga,
                                        const T* __restrict__ dgw, const int16_t* __restrict__ terms,
                                        const T* __restrict__ coeff, T* __restrict__ partial, int atoms,
                                        int components, int irreps, int channels, int chunk) {
  const int job = static_cast<int>(blockIdx.x);
  const int atom = job % atoms;
  const int part = job / atoms;
  const int begin = rowptr[atom];
  const int count = rowptr[atom + 1] - begin;
  const int local0 = min(count, part * chunk);
  const int local1 = min(count, local0 + chunk);
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    for (int d = 0; d < components; ++d) {
      const int r = irrep_of(terms, d);
      const T c = coeff[d];
      T acc = T(0);
      for (int local = local0; local < local1; ++local) {
        const int edge = order[begin + local];
        if (index[edge] != atom) continue;
        const T av = basis[static_cast<long long>(edge) * components + d];
        const T wv = weight[(static_cast<long long>(edge) * irreps + r) * channels + channel];
        if constexpr (kSecond == 0) {
          acc += c * av * wv;
        } else {
          const T dgc = dgw[(static_cast<long long>(edge) * irreps + r) * channels + channel];
          acc += c * (wv * dga[static_cast<long long>(edge) * components + d] + av * dgc);
        }
      }
      partial[((static_cast<long long>(part) * atoms + atom) * components + d) * channels + channel] = acc;
    }
  }
}

template <typename T>
__global__ void reduce_kernel(const T* __restrict__ partial, T* __restrict__ out, int parts,
                              long long stride) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= stride) return;
  T acc = T(0);
  for (int part = 0; part < parts; ++part) acc += partial[static_cast<long long>(part) * stride + idx];
  out[idx] = acc;
}

__global__ void atom_row_offsets(const int32_t* __restrict__ keys, int64_t edges, int64_t atoms,
                                  int32_t* __restrict__ rowptr) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row > atoms) return;
  int64_t lo = 0;
  int64_t hi = edges;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (keys[mid] < row) lo = mid + 1;
    else hi = mid;
  }
  rowptr[row] = static_cast<int32_t>(lo);
}

void atom_csr(const at::Tensor& index, int64_t atoms, at::Tensor& order, at::Tensor& rowptr) {
  const int64_t edges = index.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto sorted = at::empty_like(index);
  if (edges > 0) {
    auto iota = at::arange(edges, index.options());
    // Full key width. A shorter radix interleaves out-of-range ids into the
    // valid buckets and the row search then drops live edges.
    const int bits = 32;
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, index.const_data_ptr<int32_t>(), sorted.mutable_data_ptr<int32_t>(),
        iota.const_data_ptr<int32_t>(), order.mutable_data_ptr<int32_t>(), static_cast<int>(edges), 0,
        bits, stream));
    auto temp = at::empty({static_cast<int64_t>(bytes)}, index.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.mutable_data_ptr(), bytes, index.const_data_ptr<int32_t>(),
        sorted.mutable_data_ptr<int32_t>(), iota.const_data_ptr<int32_t>(),
        order.mutable_data_ptr<int32_t>(), static_cast<int>(edges), 0, bits, stream));
  }
  const int64_t threads = 256;
  const int64_t blocks = (atoms + 1 + threads - 1) / threads;
  atom_row_offsets<<<static_cast<unsigned>(blocks), static_cast<unsigned>(threads), 0, stream>>>(
      sorted.const_data_ptr<int32_t>(), edges, atoms, rowptr.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int threads_for(int channels) {
  if (channels <= 32) return 32;
  if (channels <= 64) return 64;
  return kMaxThreads;
}

int64_t choose_parts(int64_t row_bytes, int64_t n) {
  int64_t parts = 1;
  if (row_bytes > 0 && n > 0) {
    const int64_t fit = kPartialBudget / row_bytes;
    parts = n < fit ? n : fit;
    if (parts < 1) parts = 1;
    if (parts > kMaxChunks) parts = kMaxChunks;
  }
  return parts;
}

void launch_reduce(const at::Tensor& partial, at::Tensor& out, int parts, long long stride) {
  const int threads = 256;
  const long long blocks = (stride + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "weighted-channels reduction grid is too large");
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "weighted_channels_reduce", [&] {
    reduce_kernel<scalar_t><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        partial.const_data_ptr<scalar_t>(), out.mutable_data_ptr<scalar_t>(), parts, stride);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor as_int_index(const at::Tensor& indices, const at::Tensor& like, int64_t edges) {
  TORCH_CHECK(indices.device() == like.device() && indices.dim() == 1 && indices.numel() == edges,
              "index must be a vector of one atom id per edge");
  TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
              "index must be int32 or int64");
  at::Tensor index = indices.scalar_type() == at::kInt ? indices : indices.to(at::kInt);
  TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
  return index;
}

void check_terms(const at::Tensor& terms, const at::Tensor& coeff, const at::Tensor& like, int* nterms) {
  TORCH_CHECK(terms.device() == like.device() && coeff.device() == like.device(),
              "term tables must be on the feature device");
  TORCH_CHECK(terms.scalar_type() == at::kShort && terms.is_contiguous() && terms.dim() == 2 &&
                  terms.size(1) == 2,
              "terms must be a contiguous int16 (T, 2) table");
  TORCH_CHECK(coeff.scalar_type() == like.scalar_type() && coeff.is_contiguous() && coeff.dim() == 1 &&
                  coeff.numel() == terms.size(0),
              "coefficients must match the term count and the feature dtype");
  TORCH_CHECK(terms.size(0) <= std::numeric_limits<int>::max(), "too many terms");
  *nterms = static_cast<int>(terms.size(0));
}

// The kernels read row ``d`` of the table as the irrep of component ``d``.
void check_one_term_per_component(const at::Tensor& terms, int components) {
  TORCH_CHECK(terms.size(0) == components,
              "weighted channels expects one term per component, in component order");
}

Problem decode(const at::Tensor& basis, const at::Tensor& weight, int64_t channels, int64_t num_irreps,
               int64_t num_rows) {
  TORCH_CHECK(basis.is_cuda() && basis.is_contiguous() && basis.dim() == 2,
              "basis must be a contiguous CUDA matrix");
  TORCH_CHECK(basis.scalar_type() == at::kFloat || basis.scalar_type() == at::kDouble,
              "weighted channels supports float32 and float64");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == basis.scalar_type() &&
                  weight.device() == basis.device(),
              "weight must be a contiguous CUDA tensor of the basis dtype");
  TORCH_CHECK(channels > 0 && channels <= std::numeric_limits<int>::max(), "channels must be positive");
  TORCH_CHECK(num_irreps > 0 && num_irreps <= kMaxSegments, "irreps exceed ", kMaxSegments);
  TORCH_CHECK(basis.size(0) <= std::numeric_limits<int>::max(), "edge count exceeds int32");
  TORCH_CHECK(basis.size(1) > 0 && basis.size(1) <= kMaxSegments, "components exceed ", kMaxSegments);
  const int64_t width = num_irreps * channels;
  if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == basis.size(0) && weight.size(1) == num_irreps &&
                    weight.size(2) == channels,
                "weight must be (edges, irreps, channels)");
  } else if (weight.dim() == 2) {
    TORCH_CHECK(weight.size(0) == basis.size(0) && weight.size(1) == width,
                "weight must be (edges, irreps * channels)");
  } else {
    TORCH_CHECK(false, "weight must be (edges, irreps, channels) or the flat irrep-major row");
  }
  Problem problem;
  problem.edges = static_cast<int>(basis.size(0));
  problem.components = static_cast<int>(basis.size(1));
  problem.irreps = static_cast<int>(num_irreps);
  problem.channels = static_cast<int>(channels);
  problem.nterms = 0;
  problem.indexed = num_rows >= 0;
  if (problem.indexed) {
    TORCH_CHECK(num_rows <= std::numeric_limits<int>::max(), "num_rows exceeds int32");
    problem.atoms = static_cast<int>(num_rows);
  } else {
    TORCH_CHECK(num_rows == -1, "num_rows must be -1 or a non-negative row count");
    problem.atoms = 0;
  }
  return problem;
}

int64_t output_rows(const Problem& problem) {
  return problem.indexed ? problem.atoms : problem.edges;
}

template <typename T, int kSecond>
void launch_scatter(const at::Tensor& basis, const at::Tensor& weight, const at::Tensor& index,
                    const at::Tensor& dga, const at::Tensor& dgw, const at::Tensor& terms,
                    const at::Tensor& coeff, at::Tensor& out, const Problem& problem) {
  if (problem.atoms == 0 || problem.edges == 0 || problem.nterms == 0) return;
  auto order = at::empty({problem.edges}, index.options());
  auto rowptr = at::empty({problem.atoms + 1}, index.options());
  atom_csr(index, problem.atoms, order, rowptr);
  const int64_t row_bytes = static_cast<int64_t>(problem.atoms) * problem.components * problem.channels *
                            basis.element_size();
  const int parts = static_cast<int>(choose_parts(row_bytes, problem.edges));
  const int chunk = static_cast<int>((static_cast<long long>(problem.edges) + parts - 1) / parts);
  auto partial = at::empty({parts, problem.atoms, problem.components * problem.channels}, basis.options());
  const long long jobs = static_cast<long long>(parts) * problem.atoms;
  TORCH_CHECK(jobs <= std::numeric_limits<int>::max(), "weighted-channels scatter grid is too large");
  const int threads = threads_for(problem.channels);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const T* dga_ptr = nullptr;
  const T* dgw_ptr = nullptr;
  if constexpr (kSecond) {
    dga_ptr = dga.const_data_ptr<T>();
    dgw_ptr = dgw.const_data_ptr<T>();
  }
  scatter_partial_kernel<T, kSecond><<<static_cast<unsigned>(jobs), threads, 0, stream>>>(
      basis.const_data_ptr<T>(), weight.const_data_ptr<T>(), index.const_data_ptr<int32_t>(),
      order.const_data_ptr<int32_t>(), rowptr.const_data_ptr<int32_t>(), dga_ptr, dgw_ptr,
      terms.const_data_ptr<int16_t>(), coeff.const_data_ptr<T>(), partial.mutable_data_ptr<T>(),
      problem.atoms, problem.components, problem.irreps, problem.channels, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_reduce(partial, out, parts, static_cast<long long>(problem.atoms) * problem.components *
                                         problem.channels);
}

template <typename T, int kSecond, int kIndexed>
void launch_adjoint(const at::Tensor& basis, const at::Tensor& weight, const at::Tensor& index,
                    const at::Tensor& g, const at::Tensor& dga, const at::Tensor& dgw,
                    const at::Tensor& terms, const at::Tensor& coeff, at::Tensor& grad_basis,
                    at::Tensor& grad_weight, at::Tensor& dgy, const Problem& problem) {
  if (problem.edges == 0 || problem.nterms == 0) return;
  const int threads = threads_for(problem.channels);
  TORCH_CHECK(threads <= kMaxThreads, "weighted-channels block is larger than the channel reduction");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = kIndexed ? problem.atoms : problem.edges;
  const int32_t* ids = nullptr;
  const T* dga_ptr = nullptr;
  const T* dgw_ptr = nullptr;
  if constexpr (kIndexed) ids = index.const_data_ptr<int32_t>();
  if constexpr (kSecond) {
    dga_ptr = dga.const_data_ptr<T>();
    dgw_ptr = dgw.const_data_ptr<T>();
  }
  adjoint_kernel<T, kSecond, kIndexed><<<problem.edges, threads, 0, stream>>>(
      basis.const_data_ptr<T>(), weight.const_data_ptr<T>(), ids, g.const_data_ptr<T>(), dga_ptr, dgw_ptr,
      terms.const_data_ptr<int16_t>(), coeff.const_data_ptr<T>(), grad_basis.mutable_data_ptr<T>(),
      grad_weight.mutable_data_ptr<T>(), dgy.mutable_data_ptr<T>(), problem.edges, rows, problem.components,
      problem.irreps, problem.channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor weighted_channels_cuda(const at::Tensor& basis, const at::Tensor& weight, const at::Tensor& index,
                                  const at::Tensor& terms, const at::Tensor& coefficients, int64_t channels,
                                  int64_t num_irreps, int64_t num_rows) {
  const c10::cuda::CUDAGuard guard(basis.device());
  Problem problem = decode(basis, weight, channels, num_irreps, num_rows);
  check_terms(terms, coefficients, basis, &problem.nterms);
  check_one_term_per_component(terms, problem.components);
  const int64_t width = static_cast<int64_t>(problem.components) * problem.channels;
  auto y = at::zeros({output_rows(problem), width}, basis.options());
  if (problem.edges == 0 || problem.nterms == 0) return y;
  const int threads = threads_for(problem.channels);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (!problem.indexed) {
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "weighted_channels_forward", [&] {
      scale_kernel<scalar_t><<<problem.edges, threads, 0, stream>>>(
          basis.const_data_ptr<scalar_t>(), weight.const_data_ptr<scalar_t>(),
          terms.const_data_ptr<int16_t>(), coefficients.const_data_ptr<scalar_t>(),
          y.mutable_data_ptr<scalar_t>(), problem.edges, problem.components, problem.irreps,
          problem.channels);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  auto ids = as_int_index(index, basis, problem.edges);
  AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "weighted_channels_scatter", [&] {
    launch_scatter<scalar_t, 0>(basis, weight, ids, basis, basis, terms, coefficients, y, problem);
  });
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> weighted_channels_adjoints_cuda(
    const at::Tensor& basis, const at::Tensor& weight, const at::Tensor& index, const at::Tensor& g,
    const at::Tensor& dga, const at::Tensor& dgw, const at::Tensor& dgy, const at::Tensor& terms,
    const at::Tensor& coefficients, bool second, int64_t channels, int64_t num_irreps, int64_t num_rows) {
  const c10::cuda::CUDAGuard guard(basis.device());
  Problem problem = decode(basis, weight, channels, num_irreps, num_rows);
  check_terms(terms, coefficients, basis, &problem.nterms);
  check_one_term_per_component(terms, problem.components);
  const int64_t width = static_cast<int64_t>(problem.components) * problem.channels;
  const int64_t rows = output_rows(problem);
  TORCH_CHECK(g.device() == basis.device() && g.scalar_type() == basis.scalar_type() && g.is_contiguous() &&
                  g.dim() == 2 && g.size(0) == rows && g.size(1) == width,
              "cotangent g must be a contiguous (rows, components * channels) matrix");
  TORCH_CHECK(dgy.is_cuda() && dgy.scalar_type() == basis.scalar_type(),
              "dgy must be a CUDA tensor of the compute dtype");
  auto grad_basis = at::zeros({problem.edges, problem.components}, basis.options());
  auto grad_weight = at::zeros_like(weight);
  auto grad_g = at::zeros({second ? rows : int64_t(0), width}, basis.options());
  if (second) {
    TORCH_CHECK(dga.sizes() == basis.sizes() && dgw.sizes() == weight.sizes() && dga.is_contiguous() &&
                    dgw.is_contiguous() && dga.scalar_type() == basis.scalar_type() &&
                    dgw.scalar_type() == basis.scalar_type() && dga.device() == basis.device() &&
                    dgw.device() == basis.device(),
                "second-derivative cotangents must match the basis and the weight");
  }
  if (problem.edges == 0 || problem.nterms == 0) return {grad_basis, grad_weight, grad_g};
  at::Tensor ids = problem.indexed ? as_int_index(index, basis, problem.edges) : index;
  AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "weighted_channels_adjoints", [&] {
    if (!second && !problem.indexed) {
      launch_adjoint<scalar_t, 0, 0>(basis, weight, ids, g, dga, dgw, terms, coefficients, grad_basis,
                                      grad_weight, grad_g, problem);
    } else if (!second && problem.indexed) {
      launch_adjoint<scalar_t, 0, 1>(basis, weight, ids, g, dga, dgw, terms, coefficients, grad_basis,
                                      grad_weight, grad_g, problem);
    } else if (second && !problem.indexed) {
      launch_adjoint<scalar_t, 1, 0>(basis, weight, ids, g, dga, dgw, terms, coefficients, grad_basis,
                                      grad_weight, grad_g, problem);
    } else {
      launch_adjoint<scalar_t, 1, 1>(basis, weight, ids, g, dga, dgw, terms, coefficients, grad_basis,
                                      grad_weight, grad_g, problem);
      launch_scatter<scalar_t, 1>(basis, weight, ids, dga, dgw, terms, coefficients, grad_g, problem);
    }
  });
  return {grad_basis, grad_weight, grad_g};
}

}  // namespace molnex::equivariance
