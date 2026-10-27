// Symmetric contraction of MACE's product block, one CTA per node and channel
// tile, every operand staged in shared memory, the term table merged and
// sorted by output segment.
//
// A term (C, w, i, j, k, o) contributes y[o] += C · W[index[n], w] · x[i] · x[j] · x[k].
// The table is the merged one of SymmetricContractionTerms: slots ascending, a
// missing slot pointing at the sentinel segment S whose staged value is 1 (and
// whose cotangent is 0), so every term is a degree-3 monomial and the loops
// carry no degree branch. Terms are packed into one int32
// (w << 21 | i << 14 | j << 7 | k) beside their coefficient; both are staged
// into shared memory once per CTA. Sorted by (o, i, j, k, w), consecutive
// terms share (i, j): the product x[i]·x[j] and the partial sums the adjoints
// need per (i, j) run stay in registers, so a term costs the reads of x[k],
// W[w] and its own table entry.
//
// Threads run along a channel tile (32 / 64 / 128 channels); wider
// multiplicities loop over tiles with the operands restaged. A node whose
// element id is out of range contributes zeros and reads nothing.
//
// First adjoint (cotangent g of y), per term with base = C · W[w] · g[o]:
//   gx[i] += base · x[j] · x[k],  gx[j] += base · x[i] · x[k],  gx[k] += base · x[i] · x[j]
//   gw[w] += C · g[o] · x[i] · x[j] · x[k]
// gx accumulates in shared memory (one column per thread, no atomics); the
// weight adjoint is written per node (G[n, w, c]) and summed into the elements
// in a deterministic two-pass scheme: nodes in a stable element order, chunks
// of the sorted list summed by column-parallel CTAs, then the chunks first to
// last (sum_by_element). MACE's family (contraction_tables.hpp) runs the
// compiled kernels of contraction_compiled.cuh instead — every operand in
// registers, the term loop unrolled — selected by ``config``.
//
// Second adjoint (cotangents dgx of gx and dgw of gw), with p = C · g[o],
// wv = W[w], dwv = dgw[w], d = dgx:
//   dx[i] += p · (wv · (d[j] x[k] + d[k] x[j]) + dwv · x[j] x[k])   and the two cyclic slots
//   dg[o] += C · (wv · (d[i] x[j] x[k] + d[j] x[i] x[k] + d[k] x[i] x[j]) + dwv · x[i] x[j] x[k])
//   dw[w] += C · g[o] · (d[i] x[j] x[k] + d[j] x[i] x[k] + d[k] x[i] x[j])
// dw does not read dgw (gw does not depend on W). Third derivatives are refused
// in Python. Accumulation is in the compute dtype.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "sc_dispatch.cuh"

namespace molnex::equivariance {
namespace {

constexpr int kMaxSegments = 64;      // input / output components per operand
constexpr int kMaxWeight = 1023;      // 10-bit weight segment in the packed term

struct Problem {
  int n, z, s, b, o, u, nterms, tile;
};

__device__ __forceinline__ int tw(int packed) { return packed >> 21; }
__device__ __forceinline__ int ti(int packed) { return (packed >> 14) & 127; }
__device__ __forceinline__ int tj(int packed) { return (packed >> 7) & 127; }
__device__ __forceinline__ int tk(int packed) { return packed & 127; }

// Shared-memory layout of one CTA: the staged term table and the operand rows
// of one channel tile. The coefficients lead (T-aligned at offset 0), the
// int32 words follow, and the rows start at the next 16-byte boundary.
template <typename T>
struct Stage {
  T* coef;          // nterms
  int32_t* packed;  // nterms
  T* x;             // (S + 1) * tile, x[S] = 1
  T* w;             // B * tile
  T* g;             // O * tile
  T* acc;           // (S + 1) * tile: gx / dx accumulators
  T* d;             // (S + 1) * tile: dgx, d[S] = 0
  T* dw;            // B * tile: dgw[element]
  T* gw;            // B * tile: weight partials
};

__host__ __device__ inline size_t stage_bytes(int nterms, int s, int b, int o, int tile, int elem,
                                              bool g, bool acc, bool d, bool dw, bool gw) {
  size_t bytes = size_t(nterms) * (sizeof(int32_t) + elem);
  bytes = (bytes + 15) & ~size_t(15);
  bytes += size_t(s + 1) * tile * elem;
  bytes += size_t(b) * tile * elem;
  if (g) bytes += size_t(o) * tile * elem;
  if (acc) bytes += size_t(s + 1) * tile * elem;
  if (d) bytes += size_t(s + 1) * tile * elem;
  if (dw) bytes += size_t(b) * tile * elem;
  if (gw) bytes += size_t(b) * tile * elem;
  return bytes;
}

template <typename T>
__device__ Stage<T> carve(char* smem, int nterms, int s, int b, int o, int tile, bool g, bool acc,
                          bool d, bool dw, bool gw) {
  Stage<T> st;
  st.coef = reinterpret_cast<T*>(smem);
  st.packed = reinterpret_cast<int32_t*>(smem + size_t(nterms) * sizeof(T));
  size_t off = (size_t(nterms) * (sizeof(int32_t) + sizeof(T)) + 15) & ~size_t(15);
  T* base = reinterpret_cast<T*>(smem + off);
  st.x = base;
  base += size_t(s + 1) * tile;
  st.w = base;
  base += size_t(b) * tile;
  st.g = g ? base : nullptr;
  if (g) base += size_t(o) * tile;
  st.acc = acc ? base : nullptr;
  if (acc) base += size_t(s + 1) * tile;
  st.d = d ? base : nullptr;
  if (d) base += size_t(s + 1) * tile;
  st.dw = dw ? base : nullptr;
  if (dw) base += size_t(b) * tile;
  st.gw = gw ? base : nullptr;
  return st;
}

template <typename T>
__device__ void stage_table(const Stage<T>& st, const int32_t* __restrict__ packed,
                            const T* __restrict__ coef, int nterms) {
  for (int t = threadIdx.x; t < nterms; t += blockDim.x) {
    st.packed[t] = packed[t];
    st.coef[t] = coef[t];
  }
}

// One row of a (rows, segments * u) matrix, channels c0 .. c0 + tile, into
// smem[seg * tile + c]; the sentinel segment gets `sentinel`.
template <typename T>
__device__ void stage_row(T* __restrict__ dst, const T* __restrict__ src, int segments, int u,
                          int c0, int tile, T sentinel) {
  const int c = threadIdx.x;
  const bool live = c0 + c < u;
  for (int seg = 0; seg < segments; ++seg)
    dst[seg * tile + c] = live ? src[size_t(seg) * u + c0 + c] : T(0);
  dst[segments * tile + c] = sentinel;
}

template <typename T>
__device__ void stage_row_plain(T* __restrict__ dst, const T* __restrict__ src, int segments, int u,
                                int c0, int tile) {
  const int c = threadIdx.x;
  const bool live = c0 + c < u;
  for (int seg = 0; seg < segments; ++seg)
    dst[seg * tile + c] = live ? src[size_t(seg) * u + c0 + c] : T(0);
}

template <typename T>
__device__ void zero_rows(T* __restrict__ dst, int segments, int tile) {
  for (int seg = 0; seg < segments; ++seg) dst[seg * tile + threadIdx.x] = T(0);
}

// ── forward ────────────────────────────────────────────────────────────────

template <typename T>
__global__ void forward_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               const int32_t* __restrict__ index, const int32_t* __restrict__ packed,
                               const T* __restrict__ coef, const int32_t* __restrict__ out_offsets,
                               T* __restrict__ y, Problem p) {
  extern __shared__ __align__(16) char smem[];
  const int node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < p.z;
  const int tile = p.tile, c = threadIdx.x;
  Stage<T> st = carve<T>(smem, p.nterms, p.s, p.b, p.o, tile, false, false, false, false, false);
  stage_table(st, packed, coef, p.nterms);
  const T* xn = x + size_t(node) * p.s * p.u;
  const T* wz = valid ? w + size_t(element) * p.b * p.u : nullptr;
  for (int c0 = 0; c0 < p.u; c0 += tile) {
    __syncthreads();
    if (valid) {
      stage_row(st.x, xn, p.s, p.u, c0, tile, T(1));
      stage_row_plain(st.w, wz, p.b, p.u, c0, tile);
    }
    __syncthreads();
    const bool live = c0 + c < p.u;
    T* yn = y + size_t(node) * p.o * p.u + c0 + c;
    for (int o = 0; o < p.o; ++o) {
      T acc = T(0);
      if (valid) {
        const int t0 = out_offsets[o], t1 = out_offsets[o + 1];
        int prev = -1;
        T xij = T(0);
        for (int t = t0; t < t1; ++t) {
          const int term = st.packed[t];
          const int ij = term >> 7;  // (w, i, j) — w is constant within few runs, harmless
          if ((ij & 0x3fff) != prev) {
            prev = ij & 0x3fff;
            xij = st.x[ti(term) * tile + c] * st.x[tj(term) * tile + c];
          }
          acc += (st.coef[t] * xij * st.x[tk(term) * tile + c]) * st.w[tw(term) * tile + c];
        }
      }
      if (live) yn[size_t(o) * p.u] = acc;
    }
  }
}

// ── first adjoint: node pass (gx) ──────────────────────────────────────────

// gx, and the node's weight partial G[node, w, c] = Σ C · g[o] · x[i] x[j] x[k]
// (summed into the elements afterwards by ``sum_by_element``).
template <typename T>
__global__ void feature_adjoint_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                       const int32_t* __restrict__ index,
                                       const int32_t* __restrict__ packed, const T* __restrict__ coef,
                                       const int32_t* __restrict__ out_offsets, const T* __restrict__ g,
                                       T* __restrict__ gx, T* __restrict__ gw_node, Problem p) {
  extern __shared__ __align__(16) char smem[];
  const int node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < p.z;
  const int tile = p.tile, c = threadIdx.x;
  Stage<T> st = carve<T>(smem, p.nterms, p.s, p.b, p.o, tile, true, true, false, false, true);
  stage_table(st, packed, coef, p.nterms);
  const T* xn = x + size_t(node) * p.s * p.u;
  const T* gn = g + size_t(node) * p.o * p.u;
  const T* wz = valid ? w + size_t(element) * p.b * p.u : nullptr;
  for (int c0 = 0; c0 < p.u; c0 += tile) {
    __syncthreads();
    zero_rows(st.acc, p.s + 1, tile);
    zero_rows(st.gw, p.b, tile);
    if (valid) {
      stage_row(st.x, xn, p.s, p.u, c0, tile, T(1));
      stage_row_plain(st.w, wz, p.b, p.u, c0, tile);
      stage_row_plain(st.g, gn, p.o, p.u, c0, tile);
    }
    __syncthreads();
    if (valid) {
      for (int o = 0; o < p.o; ++o) {
        const T go = st.g[o * tile + c];
        const int t0 = out_offsets[o], t1 = out_offsets[o + 1];
        int prev = -1, pi = 0, pj = 0;
        T xi = T(0), xj = T(0), ri = T(0), rj = T(0);
        for (int t = t0; t < t1; ++t) {
          const int term = st.packed[t];
          const int ij = (term >> 7) & 0x3fff;
          if (ij != prev) {
            if (prev >= 0) {
              st.acc[pi * tile + c] += xj * ri;
              st.acc[pj * tile + c] += xi * rj;
            }
            prev = ij;
            pi = ti(term);
            pj = tj(term);
            xi = st.x[pi * tile + c];
            xj = st.x[pj * tile + c];
            ri = rj = T(0);
          }
          const int k = tk(term), wseg = tw(term);
          const T xk = st.x[k * tile + c];
          const T cg = st.coef[t] * go;
          const T base = cg * st.w[wseg * tile + c];
          ri += base * xk;
          rj += base * xk;
          st.acc[k * tile + c] += base * xi * xj;
          st.gw[wseg * tile + c] += cg * (xi * xj * xk);
        }
        if (prev >= 0) {
          st.acc[pi * tile + c] += xj * ri;
          st.acc[pj * tile + c] += xi * rj;
        }
      }
    }
    __syncthreads();
    if (c0 + c < p.u) {
      T* out = gx + size_t(node) * p.s * p.u + c0 + c;
      for (int seg = 0; seg < p.s; ++seg) out[size_t(seg) * p.u] = st.acc[seg * tile + c];
      T* gwn = gw_node + size_t(node) * p.b * p.u + c0 + c;
      for (int wseg = 0; wseg < p.b; ++wseg) gwn[size_t(wseg) * p.u] = st.gw[wseg * tile + c];
    }
  }
}

// ── weight adjoint: the nodes' partials summed into their elements ─────────
//
// Nodes in a stable element order (radix sort); the sorted list is cut into
// chunks of a few nodes regardless of element, and one CTA per (chunk, column
// block) walks its chunk in order, flushing a partial row whenever the
// element changes and recording which element each row holds. The reduction
// then adds, per element and column, the rows of that element over the
// chunks its node range spans (from the CSR offsets), first to last. Many
// CTAs even when one element owns every node (MACE's shared product).

template <typename T>
__global__ void chunk_sum_kernel(const T* __restrict__ values, const int32_t* __restrict__ order,
                                 const int32_t* __restrict__ index, T* __restrict__ partial,
                                 int32_t* __restrict__ slot_element, int64_t n, int64_t width,
                                 int chunk, int slots) {
  const int part = blockIdx.x;
  const int64_t col = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  const int64_t first = min(n, int64_t(part) * chunk), last = min(n, first + chunk);
  const bool keeper = blockIdx.y == 0 && threadIdx.x == 0;
  if (keeper)
    for (int s = 0; s < slots; ++s) slot_element[size_t(part) * slots + s] = -1;
  if (first >= last) return;
  int current = index[order[first]], slot = 0;
  T acc = T(0);
  for (int64_t i = first; i < last; ++i) {
    const int node = order[i];
    const int element = index[node];
    if (element != current) {
      if (col < width) partial[(size_t(part) * slots + slot) * width + col] = acc;
      if (keeper) slot_element[size_t(part) * slots + slot] = current;
      ++slot;
      current = element;
      acc = T(0);
    }
    if (col < width) acc += values[size_t(node) * width + col];
  }
  if (col < width) partial[(size_t(part) * slots + slot) * width + col] = acc;
  if (keeper) slot_element[size_t(part) * slots + slot] = current;
}

template <typename T>
__global__ void element_reduce_kernel(const T* __restrict__ partial,
                                      const int32_t* __restrict__ slot_element,
                                      const int32_t* __restrict__ rowptr, T* __restrict__ out,
                                      int chunk, int slots, int64_t z, int64_t width) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= z * width) return;
  const int64_t element = idx / width, col = idx % width;
  const int begin = rowptr[element], end = rowptr[element + 1];
  T acc = T(0);
  if (begin < end) {
    // The chunks the element's sorted node range spans, in order.
    for (int part = begin / chunk; part <= (end - 1) / chunk; ++part) {
      for (int s = 0; s < slots; ++s) {
        const int held = slot_element[size_t(part) * slots + s];
        if (held == element) {
          acc += partial[(size_t(part) * slots + s) * width + col];
          break;
        }
        if (held < 0 || held > element) break;
      }
    }
  }
  out[idx] = acc;
}

// ── second adjoint: node pass (dx, dg) ─────────────────────────────────────

template <typename T>
__global__ void second_node_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                   const int32_t* __restrict__ index,
                                   const int32_t* __restrict__ packed, const T* __restrict__ coef,
                                   const int32_t* __restrict__ out_offsets, const T* __restrict__ g,
                                   const T* __restrict__ dgx, const T* __restrict__ dgw,
                                   T* __restrict__ dx, T* __restrict__ dg, T* __restrict__ dw_node,
                                   Problem p) {
  extern __shared__ __align__(16) char smem[];
  const int node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < p.z;
  const int tile = p.tile, c = threadIdx.x;
  Stage<T> st = carve<T>(smem, p.nterms, p.s, p.b, p.o, tile, true, true, true, true, true);
  stage_table(st, packed, coef, p.nterms);
  const T* xn = x + size_t(node) * p.s * p.u;
  const T* dn = dgx + size_t(node) * p.s * p.u;
  const T* gn = g + size_t(node) * p.o * p.u;
  const T* wz = valid ? w + size_t(element) * p.b * p.u : nullptr;
  const T* dwz = valid ? dgw + size_t(element) * p.b * p.u : nullptr;
  for (int c0 = 0; c0 < p.u; c0 += tile) {
    __syncthreads();
    zero_rows(st.acc, p.s + 1, tile);
    zero_rows(st.gw, p.b, tile);
    if (valid) {
      stage_row(st.x, xn, p.s, p.u, c0, tile, T(1));
      stage_row(st.d, dn, p.s, p.u, c0, tile, T(0));
      stage_row_plain(st.w, wz, p.b, p.u, c0, tile);
      stage_row_plain(st.dw, dwz, p.b, p.u, c0, tile);
      stage_row_plain(st.g, gn, p.o, p.u, c0, tile);
    }
    __syncthreads();
    const bool live = c0 + c < p.u;
    T* dgn = dg + size_t(node) * p.o * p.u + c0 + c;
    for (int o = 0; o < p.o; ++o) {
      T dgo = T(0);
      if (valid) {
        const T go = st.g[o * tile + c];
        const int t0 = out_offsets[o], t1 = out_offsets[o + 1];
        for (int t = t0; t < t1; ++t) {
          const int term = st.packed[t];
          const int i = ti(term), j = tj(term), k = tk(term), wseg = tw(term);
          const T xi = st.x[i * tile + c], xj = st.x[j * tile + c], xk = st.x[k * tile + c];
          const T di = st.d[i * tile + c], dj = st.d[j * tile + c], dk = st.d[k * tile + c];
          const T wv = st.w[wseg * tile + c], dwv = st.dw[wseg * tile + c];
          const T cc = st.coef[t], pp = cc * go;
          const T sym = di * xj * xk + dj * xi * xk + dk * xi * xj;
          dgo += cc * (wv * sym + dwv * xi * xj * xk);
          st.acc[i * tile + c] += pp * (wv * (dj * xk + dk * xj) + dwv * xj * xk);
          st.acc[j * tile + c] += pp * (wv * (di * xk + dk * xi) + dwv * xi * xk);
          st.acc[k * tile + c] += pp * (wv * (di * xj + dj * xi) + dwv * xi * xj);
          st.gw[wseg * tile + c] += pp * sym;
        }
      }
      if (live) dgn[size_t(o) * p.u] = dgo;
    }
    __syncthreads();
    if (live) {
      T* out = dx + size_t(node) * p.s * p.u + c0 + c;
      for (int seg = 0; seg < p.s; ++seg) out[size_t(seg) * p.u] = st.acc[seg * tile + c];
      T* dwn = dw_node + size_t(node) * p.b * p.u + c0 + c;
      for (int wseg = 0; wseg < p.b; ++wseg) dwn[size_t(wseg) * p.u] = st.gw[wseg * tile + c];
    }
  }
}

// ── host side ──────────────────────────────────────────────────────────────

__global__ void element_row_offsets(const int32_t* __restrict__ keys, int64_t nodes, int64_t elements,
                                    int32_t* __restrict__ rowptr) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row > elements) return;
  int64_t lo = 0, hi = nodes;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (keys[mid] < row) lo = mid + 1; else hi = mid;
  }
  rowptr[row] = int32_t(lo);
}

void element_csr(const at::Tensor& index, int64_t elements, at::Tensor& order, at::Tensor& rowptr) {
  const int64_t nodes = index.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto sorted = at::empty_like(index);
  if (nodes > 0) {
    auto iota = at::arange(nodes, index.options());
    int bits = 1;
    while ((int64_t(1) << bits) < elements) ++bits;
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, index.const_data_ptr<int32_t>(), sorted.mutable_data_ptr<int32_t>(),
        iota.const_data_ptr<int32_t>(), order.mutable_data_ptr<int32_t>(), int(nodes), 0, bits,
        stream));
    auto temp = at::empty({int64_t(bytes)}, index.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.mutable_data_ptr(), bytes, index.const_data_ptr<int32_t>(),
        sorted.mutable_data_ptr<int32_t>(), iota.const_data_ptr<int32_t>(),
        order.mutable_data_ptr<int32_t>(), int(nodes), 0, bits, stream));
  }
  const int64_t threads = 256, blocks = (elements + 1 + threads - 1) / threads;
  element_row_offsets<<<unsigned(blocks), unsigned(threads), 0, stream>>>(
      sorted.const_data_ptr<int32_t>(), nodes, elements, rowptr.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int tile_for(int channels) {
  if (channels <= 32) return 32;
  if (channels <= 64) return 64;
  return 128;
}

template <typename T, typename Kernel>
void allow_smem(Kernel kernel, size_t bytes) {
  const size_t limit = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  TORCH_CHECK(bytes <= limit, "symmetric contraction needs ", bytes,
              " bytes of shared memory per CTA; this device allows ", limit);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}

void check_cuda_pair(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "symmetric contraction operands must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "symmetric contraction supports float32 and float64");
  TORCH_CHECK(w.scalar_type() == x.scalar_type() && w.device() == x.device(),
              "weight must share the dtype and device of the features");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.is_contiguous() && w.is_contiguous(),
              "features and weight must be contiguous matrices");
}

at::Tensor as_int_index(const at::Tensor& indices, const at::Tensor& like, int64_t nodes) {
  TORCH_CHECK(indices.device() == like.device() && indices.dim() == 1 && indices.numel() == nodes,
              "indices must be a vector of one element id per node");
  TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
              "indices must be int32 or int64");
  at::Tensor index = indices.scalar_type() == at::kInt ? indices : indices.to(at::kInt);
  TORCH_CHECK(index.is_contiguous(), "indices must be contiguous");
  return index;
}

void check_table(const at::Tensor& packed, const at::Tensor& coef, const at::Tensor& offsets,
                 const at::Tensor& like, int64_t groups, const char* name) {
  TORCH_CHECK(packed.device() == like.device() && coef.device() == like.device() &&
                  offsets.device() == like.device(),
              name, " tables must be on the feature device");
  TORCH_CHECK(packed.scalar_type() == at::kInt && packed.is_contiguous() && packed.dim() == 1,
              name, " packed terms must be a contiguous int32 vector");
  TORCH_CHECK(coef.scalar_type() == like.scalar_type() && coef.is_contiguous() && coef.dim() == 1 &&
                  coef.numel() == packed.numel(),
              name, " coefficients must match the term count and the feature dtype");
  TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.is_contiguous() && offsets.dim() == 1 &&
                  offsets.numel() == groups + 1,
              name, " offsets must be int32 of length groups + 1");
  TORCH_CHECK(packed.numel() <= std::numeric_limits<int>::max(), "too many terms");
}

Problem decode(const at::Tensor& x, const at::Tensor& w, int64_t channels, int64_t segments_out,
               int64_t nterms) {
  check_cuda_pair(x, w);
  TORCH_CHECK(channels > 0 && channels <= std::numeric_limits<int>::max(), "channels must be positive");
  TORCH_CHECK(segments_out >= 0 && segments_out <= kMaxSegments, "output components exceed ", kMaxSegments);
  TORCH_CHECK(x.size(1) % channels == 0 && w.size(1) % channels == 0,
              "feature and weight widths must be multiples of the channel count");
  const int64_t s = x.size(1) / channels, b = w.size(1) / channels;
  TORCH_CHECK(s > 0 && s <= kMaxSegments, "input components exceed ", kMaxSegments);
  TORCH_CHECK(b > 0 && b <= kMaxWeight, "weight segments exceed ", kMaxWeight);
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() && w.size(0) <= std::numeric_limits<int>::max(),
              "node or element count exceeds int32");
  Problem p;
  p.n = int(x.size(0));
  p.z = int(w.size(0));
  p.s = int(s);
  p.b = int(b);
  p.o = int(segments_out);
  p.u = int(channels);
  p.nterms = int(nterms);
  p.tile = tile_for(p.u);
  return p;
}

// ``out[z] = Σ_{nodes of z} values[n]`` in a fixed order (see chunk_sum_kernel).
template <typename T>
void sum_by_element(const at::Tensor& values, const at::Tensor& index, at::Tensor& out, int64_t z) {
  const int64_t n = values.size(0), width = values.size(1);
  if (n == 0 || z == 0 || width == 0) return;
  auto order = at::empty({n}, index.options());
  auto rowptr = at::empty({z + 1}, index.options());
  element_csr(index, z, order, rowptr);
  const int parts = int(std::clamp<int64_t>((n + 15) / 16, 1, 4096));
  const int chunk = int((n + parts - 1) / parts);
  const int slots = int(std::min<int64_t>(z, chunk));
  auto partial = at::empty({int64_t(parts) * slots, width}, values.options());
  auto slot_element = at::empty({int64_t(parts) * slots}, index.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (width + 255) / 256;
  TORCH_CHECK(blocks <= 65535, "weight rows too wide for one grid");
  chunk_sum_kernel<T><<<dim3(unsigned(parts), unsigned(blocks)), 256, 0, stream>>>(
      values.const_data_ptr<T>(), order.const_data_ptr<int32_t>(), index.const_data_ptr<int32_t>(),
      partial.mutable_data_ptr<T>(), slot_element.mutable_data_ptr<int32_t>(), n, width, chunk, slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t total = z * width;
  element_reduce_kernel<T><<<unsigned((total + 255) / 256), 256, 0, stream>>>(
      partial.const_data_ptr<T>(), slot_element.const_data_ptr<int32_t>(),
      rowptr.const_data_ptr<int32_t>(), out.mutable_data_ptr<T>(), chunk, slots, z, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// A compiled configuration, or -1: config = l_in·100 + l_out·10 + degree.
bool compiled(int64_t config, const Problem& p) {
  if (config < 0) return false;
#define MOLNEX_SC_MATCH(A, B, C)                                                             \
  if (config == sc::config_id(A, B, C)) {                                                    \
    using Tb = sc::Table<A, B, C>;                                                           \
    TORCH_CHECK(p.s == Tb::NS && p.b == Tb::NB && p.o == Tb::NO && p.nterms == Tb::NT,      \
                "compiled configuration ", config, " does not match the operands");          \
    return true;                                                                             \
  }
  MOLNEX_SC_CONFIGS(MOLNEX_SC_MATCH)
#undef MOLNEX_SC_MATCH
  TORCH_CHECK(false, "no compiled symmetric contraction for configuration ", config);
}

// The generic kernels need their tables in shared memory; a wide table at a
// wide tile may not fit, so the tile halves until it does.
void fit_tile(Problem& p, int elem, bool g, bool acc, bool d, bool dw, bool gw) {
  const size_t limit = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  while (p.tile > 32 && stage_bytes(p.nterms, p.s, p.b, p.o, p.tile, elem, g, acc, d, dw, gw) > limit)
    p.tile /= 2;
}

}  // namespace

at::Tensor contraction_table(int64_t l_in, int64_t l_out, int64_t degree) {
  const int64_t config = sc::config_id(int(l_in), int(l_out), int(degree));
#define MOLNEX_SC_TABLE(A, B, C)                                                          \
  if (config == sc::config_id(A, B, C)) {                                                 \
    using Tb = sc::Table<A, B, C>;                                                        \
    auto table = at::empty({Tb::NT, 5}, at::TensorOptions().dtype(at::kShort));          \
    auto* out = table.mutable_data_ptr<int16_t>();                                        \
    for (int t = 0; t < Tb::NT; ++t) {                                                    \
      const auto m = Tb::terms[t];                                                        \
      out[5 * t] = m.w; out[5 * t + 1] = m.i; out[5 * t + 2] = m.j; out[5 * t + 3] = m.k; \
      out[5 * t + 4] = m.o;                                                               \
    }                                                                                     \
    return table;                                                                         \
  }
  MOLNEX_SC_CONFIGS(MOLNEX_SC_TABLE)
#undef MOLNEX_SC_TABLE
  return at::empty({0, 5}, at::TensorOptions().dtype(at::kShort));
}

at::Tensor contraction_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& indices,
                            const at::Tensor& packed, const at::Tensor& coefficients,
                            const at::Tensor& out_offsets, at::ArrayRef<double> values, int64_t config,
                            int64_t channels, int64_t segments_out) {
  const c10::cuda::CUDAGuard guard(x.device());
  Problem p = decode(x, w, channels, segments_out, packed.numel());
  check_table(packed, coefficients, out_offsets, x, p.o, "forward");
  auto y = at::empty({x.size(0), int64_t(p.o) * p.u}, x.options());
  if (p.n == 0 || p.o == 0) return y;
  if (p.nterms == 0) return y.zero_();
  auto index = as_int_index(indices, x, p.n);
  if (compiled(config, p)) {
    sc::Args a{x, w, index, {}, {}, {}, y, {}, {}, {}, values, p.u, p.tile};
#define MOLNEX_SC_RUN(A, B, C) if (config == sc::config_id(A, B, C)) sc::run_forward<A, B, C>(a);
    MOLNEX_SC_CONFIGS(MOLNEX_SC_RUN)
#undef MOLNEX_SC_RUN
    return y;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "symmetric_contraction_forward", [&] {
    fit_tile(p, int(x.element_size()), false, false, false, false, false);
    const size_t smem = stage_bytes(p.nterms, p.s, p.b, p.o, p.tile, int(x.element_size()), false,
                                    false, false, false, false);
    allow_smem<scalar_t>(forward_kernel<scalar_t>, smem);
    forward_kernel<scalar_t><<<p.n, p.tile, smem, stream>>>(
        x.const_data_ptr<scalar_t>(), w.const_data_ptr<scalar_t>(), index.const_data_ptr<int32_t>(),
        packed.const_data_ptr<int32_t>(), coefficients.const_data_ptr<scalar_t>(),
        out_offsets.const_data_ptr<int32_t>(), y.mutable_data_ptr<scalar_t>(), p);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> contraction_adjoints_cuda(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& indices, const at::Tensor& g,
    const at::Tensor& dgx, const at::Tensor& dgw, const at::Tensor& dgy, const at::Tensor& packed,
    const at::Tensor& coefficients, const at::Tensor& out_offsets, at::ArrayRef<double> values,
    int64_t config, bool second, int64_t channels, int64_t segments_out) {
  const c10::cuda::CUDAGuard guard(x.device());
  Problem p = decode(x, w, channels, segments_out, packed.numel());
  check_table(packed, coefficients, out_offsets, x, p.o, "forward");
  const int64_t width = int64_t(p.o) * p.u;
  TORCH_CHECK(g.device() == x.device() && g.scalar_type() == x.scalar_type() && g.is_contiguous() &&
                  g.dim() == 2 && g.size(0) == p.n && g.size(1) == width,
              "cotangent g must be a contiguous (nodes, output_dim) matrix");
  TORCH_CHECK(dgy.is_cuda() && dgy.scalar_type() == x.scalar_type(),
              "dgy must be a CUDA tensor of the compute dtype");
  auto gx = at::empty({p.n, x.size(1)}, x.options());
  auto gw = at::zeros_like(w);
  auto grad_g = at::empty({second ? p.n : int64_t(0), width}, x.options());
  if (p.n == 0 || p.nterms == 0 || p.z == 0) return {gx.zero_(), gw, grad_g.zero_()};
  if (second) {
    TORCH_CHECK(dgx.sizes() == x.sizes() && dgw.sizes() == w.sizes() && dgx.is_contiguous() &&
                    dgw.is_contiguous() && dgx.scalar_type() == x.scalar_type() &&
                    dgw.scalar_type() == x.scalar_type() && dgx.device() == x.device() &&
                    dgw.device() == x.device(),
                "second-derivative cotangents must match the features and the weight");
  }
  auto index = as_int_index(indices, x, p.n);
  // The weight adjoint per node, summed into the elements afterwards.
  auto gw_node = at::empty({p.n, w.size(1)}, x.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (compiled(config, p)) {
    sc::Args a{x, w, index, g, dgx, dgw, {}, gx, gw_node, grad_g, values, p.u, p.tile};
#define MOLNEX_SC_RUN(A, B, C)                                   \
  if (config == sc::config_id(A, B, C)) {                        \
    if (second) sc::run_second<A, B, C>(a);                      \
    else sc::run_first<A, B, C>(a);                              \
  }
    MOLNEX_SC_CONFIGS(MOLNEX_SC_RUN)
#undef MOLNEX_SC_RUN
  } else {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "symmetric_contraction_adjoints", [&] {
      const int elem = int(x.element_size());
      if (!second) {
        fit_tile(p, elem, true, true, false, false, true);
        const size_t smem = stage_bytes(p.nterms, p.s, p.b, p.o, p.tile, elem, true, true, false, false, true);
        allow_smem<scalar_t>(feature_adjoint_kernel<scalar_t>, smem);
        feature_adjoint_kernel<scalar_t><<<p.n, p.tile, smem, stream>>>(
            x.const_data_ptr<scalar_t>(), w.const_data_ptr<scalar_t>(), index.const_data_ptr<int32_t>(),
            packed.const_data_ptr<int32_t>(), coefficients.const_data_ptr<scalar_t>(),
            out_offsets.const_data_ptr<int32_t>(), g.const_data_ptr<scalar_t>(),
            gx.mutable_data_ptr<scalar_t>(), gw_node.mutable_data_ptr<scalar_t>(), p);
      } else {
        fit_tile(p, elem, true, true, true, true, true);
        const size_t smem = stage_bytes(p.nterms, p.s, p.b, p.o, p.tile, elem, true, true, true, true, true);
        allow_smem<scalar_t>(second_node_kernel<scalar_t>, smem);
        second_node_kernel<scalar_t><<<p.n, p.tile, smem, stream>>>(
            x.const_data_ptr<scalar_t>(), w.const_data_ptr<scalar_t>(), index.const_data_ptr<int32_t>(),
            packed.const_data_ptr<int32_t>(), coefficients.const_data_ptr<scalar_t>(),
            out_offsets.const_data_ptr<int32_t>(), g.const_data_ptr<scalar_t>(),
            dgx.const_data_ptr<scalar_t>(), dgw.const_data_ptr<scalar_t>(),
            gx.mutable_data_ptr<scalar_t>(), grad_g.mutable_data_ptr<scalar_t>(),
            gw_node.mutable_data_ptr<scalar_t>(), p);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "symmetric_contraction_weight_sum", [&] {
    sum_by_element<scalar_t>(gw_node, index, gw, p.z);
  });
  return {gx, gw, grad_g};
}

}  // namespace molnex::equivariance
