// Allegro contracter, ``u,iu,ju,ku+ijk``: one Clebsch–Gordan block per path on
// register-resident operand blocks.
//
//   y[e, (q3 m3), c] += W[p, c] · Σ_{m1 m2} C(l1 m1, l2 m2 | l3 m3) · x1[e, (q1 m1), c] · x2[index[e], (q2 m2), c]
//
// over the paths p = (q3, q1, q2) of one layer. A path is three irreps and a
// coupling class (l1, l2, l3); the class's coefficient positions are compiled
// (``cg.hpp``) and its values arrive as a kernel parameter, so the contraction
// of one path is an unrolled run of FMAs on blocks of at most seven registers,
// dispatched by a switch over the 34 classes. Rows (x1, the gathered x2, the
// cotangents) are staged in shared memory per edge; the shared weight row and
// the path table once per CTA. One CTA of a channel tile walks a chunk of
// edges; wider multiplicities loop over tiles.
//
// First adjoint (cotangent g of y), per path with gm = C · g[m3]:
//   gx1[m1] += W · gm · x2[m2]        (local to the edge)
//   gx2[m2] += W · gm · x1[m1]        (per edge, then summed into its atom in a fixed order)
//   gw[p]   += gm · x1[m1] · x2[m2]   (per CTA in shared memory, then the CTAs in order)
// Second adjoint (cotangents dx1 = dgx1, dx2 = dgx2 gathered, dW = dgw):
//   dg[m3]  += C · (W · (dx1[m1] x2[m2] + x1[m1] dx2[m2]) + dW · x1[m1] x2[m2])
//   dx1[m1] += gm · (W · dx2[m2] + dW · x2[m2]);  dx2[m2] += gm · (W · dx1[m1] + dW · x1[m1])
//   dw[p]   += gm · (dx1[m1] x2[m2] + x1[m1] dx2[m2])
// No atomics anywhere. An index outside [0, atoms) contributes nothing and
// is not read. Third derivatives are refused in Python.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "cg.hpp"
#include "channelwise.cuh"
#include "ct_dispatch.cuh"

namespace molnex::equivariance {
namespace {

constexpr int kMaxSegments = 64;
constexpr int kMaxPaths = 4096;
constexpr int kMaxParts = 16384;
constexpr int64_t kPartialBudget = 64LL << 20;

struct Problem {
  int edges, atoms, s1, s2, b, o, u, paths, tile;
};

template <typename T>
struct Stage {
  int4* path;        // paths: (class, off1, off2, off3)
  T* w;              // B × tile
  T* dw;             // B × tile (second order)
  T* x1;             // S1 × tile
  T* x2;             // S2 × tile
  T* g;              // O × tile
  T* d1;             // S1 × tile
  T* d2;             // S2 × tile
  T* acc1;           // S1 × tile: gx1 / dx1
  T* acc2;           // S2 × tile: gx2 / dx2 of this edge
  T* acc3;           // O × tile: y / dg
  T* gw;             // B × tile: gw / dw of this CTA
};

enum Kind { kForward = 0, kFirst = 1, kSecond = 2 };

__host__ __device__ inline size_t stage_bytes(const Problem& p, int elem, int kind) {
  size_t bytes = (size_t(p.paths) * sizeof(int4) + 15) & ~size_t(15);
  const size_t t = size_t(p.tile) * elem;
  bytes += size_t(p.b) * t;               // w
  bytes += size_t(p.s1) * t + size_t(p.s2) * t;  // x1, x2
  bytes += size_t(p.o) * t;               // acc3 (y / dg)
  if (kind != kForward) {
    bytes += size_t(p.o) * t;             // g
    bytes += size_t(p.s1) * t + size_t(p.s2) * t;  // acc1, acc2
    bytes += size_t(p.b) * t;             // gw
  }
  if (kind == kSecond) bytes += size_t(p.b) * t + size_t(p.s1) * t + size_t(p.s2) * t;  // dw, d1, d2
  return bytes;
}

template <typename T>
__device__ Stage<T> carve(char* smem, const Problem& p, int kind) {
  Stage<T> st{};
  st.path = reinterpret_cast<int4*>(smem);
  T* base = reinterpret_cast<T*>(smem + ((size_t(p.paths) * sizeof(int4) + 15) & ~size_t(15)));
  const size_t t = p.tile;
  st.w = base; base += size_t(p.b) * t;
  st.x1 = base; base += size_t(p.s1) * t;
  st.x2 = base; base += size_t(p.s2) * t;
  st.acc3 = base; base += size_t(p.o) * t;
  if (kind != kForward) {
    st.g = base; base += size_t(p.o) * t;
    st.acc1 = base; base += size_t(p.s1) * t;
    st.acc2 = base; base += size_t(p.s2) * t;
    st.gw = base; base += size_t(p.b) * t;
  }
  if (kind == kSecond) {
    st.dw = base; base += size_t(p.b) * t;
    st.d1 = base; base += size_t(p.s1) * t;
    st.d2 = base; base += size_t(p.s2) * t;
  }
  return st;
}

// One row of a (rows, segments * u) matrix, channels c0 .. c0 + tile, into smem[seg * tile + c].
template <typename T>
__device__ __forceinline__ void stage_row(T* __restrict__ dst, const T* __restrict__ src, int segments,
                                          int u, int c0, int tile) {
  const int c = threadIdx.x;
  const bool live = c0 + c < u;
  for (int seg = 0; seg < segments; ++seg)
    dst[seg * tile + c] = live && src != nullptr ? src[size_t(seg) * u + c0 + c] : T(0);
}

template <typename T>
__device__ __forceinline__ void zero_rows(T* __restrict__ dst, int segments, int tile) {
  for (int seg = 0; seg < segments; ++seg) dst[seg * tile + threadIdx.x] = T(0);
}

template <typename T>
__device__ __forceinline__ void store_rows(T* __restrict__ dst, const T* __restrict__ src, int segments,
                                           int u, int c0, int tile) {
  const int c = threadIdx.x;
  if (c0 + c >= u) return;
  for (int seg = 0; seg < segments; ++seg) dst[size_t(seg) * u + c0 + c] = src[seg * tile + c];
}

// A block of up to seven components of a staged row into registers; the
// components beyond the class's own stay zero and are never read.
template <typename T>
__device__ __forceinline__ void load_block(const T* __restrict__ row, int off, int n, int tile, int c,
                                           T (&blk)[cg::kBlock]) {
#pragma unroll
  for (int m = 0; m < cg::kBlock; ++m) blk[m] = m < n ? row[(off + m) * tile + c] : T(0);
}

template <typename T>
struct Blocks {
  T a[cg::kBlock], b[cg::kBlock], g[cg::kBlock], da[cg::kBlock], db[cg::kBlock];
  T wp, dwp;
  int off1, off2, off3;
};

template <typename T>
struct ForwardOp {
  template <int L1, int L2, int L3>
  __device__ static void run(const Blocks<T>& k, const Stage<T>& st, int tile, int c,
                             const cg::Values<T>& cgv) {
    constexpr int N3 = 2 * L3 + 1;
    T acc[N3];
#pragma unroll
    for (int m = 0; m < N3; ++m) acc[m] = T(0);
    cg::each<L1, L2, L3>(cgv, [&]<int m1, int m2, int m3>(T cf) { acc[m3] += cf * (k.a[m1] * k.b[m2]); });
#pragma unroll
    for (int m = 0; m < N3; ++m) st.acc3[(k.off3 + m) * tile + c] += k.wp * acc[m];
  }
};

template <typename T>
struct FirstOp {
  template <int L1, int L2, int L3>
  __device__ static void run(const Blocks<T>& k, const Stage<T>& st, int tile, int c,
                             const cg::Values<T>& cgv, int p) {
    constexpr int N1 = 2 * L1 + 1, N2 = 2 * L2 + 1;
    T ga1[N1], ga2[N2], gwp = T(0);
#pragma unroll
    for (int m = 0; m < N1; ++m) ga1[m] = T(0);
#pragma unroll
    for (int m = 0; m < N2; ++m) ga2[m] = T(0);
    cg::each<L1, L2, L3>(cgv, [&]<int m1, int m2, int m3>(T cf) {
      const T gm = cf * k.g[m3];
      ga1[m1] += gm * k.b[m2];
      ga2[m2] += gm * k.a[m1];
      gwp += gm * (k.a[m1] * k.b[m2]);
    });
#pragma unroll
    for (int m = 0; m < N1; ++m) st.acc1[(k.off1 + m) * tile + c] += k.wp * ga1[m];
#pragma unroll
    for (int m = 0; m < N2; ++m) st.acc2[(k.off2 + m) * tile + c] += k.wp * ga2[m];
    st.gw[p * tile + c] += gwp;
  }
};

template <typename T>
struct SecondOp {
  template <int L1, int L2, int L3>
  __device__ static void run(const Blocks<T>& k, const Stage<T>& st, int tile, int c,
                             const cg::Values<T>& cgv, int p) {
    constexpr int N1 = 2 * L1 + 1, N2 = 2 * L2 + 1, N3 = 2 * L3 + 1;
    T dg[N3], dx1[N1], dx2[N2], dwp = T(0);
#pragma unroll
    for (int m = 0; m < N3; ++m) dg[m] = T(0);
#pragma unroll
    for (int m = 0; m < N1; ++m) dx1[m] = T(0);
#pragma unroll
    for (int m = 0; m < N2; ++m) dx2[m] = T(0);
    cg::each<L1, L2, L3>(cgv, [&]<int m1, int m2, int m3>(T cf) {
      const T ab = k.a[m1] * k.b[m2];
      const T cross = k.da[m1] * k.b[m2] + k.a[m1] * k.db[m2];
      dg[m3] += cf * (k.wp * cross + k.dwp * ab);
      const T gm = cf * k.g[m3];
      dx1[m1] += gm * (k.wp * k.db[m2] + k.dwp * k.b[m2]);
      dx2[m2] += gm * (k.wp * k.da[m1] + k.dwp * k.a[m1]);
      dwp += gm * cross;
    });
#pragma unroll
    for (int m = 0; m < N3; ++m) st.acc3[(k.off3 + m) * tile + c] += dg[m];
#pragma unroll
    for (int m = 0; m < N1; ++m) st.acc1[(k.off1 + m) * tile + c] += dx1[m];
#pragma unroll
    for (int m = 0; m < N2; ++m) st.acc2[(k.off2 + m) * tile + c] += dx2[m];
    st.gw[p * tile + c] += dwp;
  }
};

#define MOLNEX_CG_FORWARD(L1, L2, L3) \
  case cg::id(L1, L2, L3): ForwardOp<T>::template run<L1, L2, L3>(k, st, tile, c, cgv); break;
#define MOLNEX_CG_FIRST(L1, L2, L3) \
  case cg::id(L1, L2, L3): FirstOp<T>::template run<L1, L2, L3>(k, st, tile, c, cgv, p); break;
#define MOLNEX_CG_SECOND(L1, L2, L3) \
  case cg::id(L1, L2, L3): SecondOp<T>::template run<L1, L2, L3>(k, st, tile, c, cgv, p); break;

__device__ __forceinline__ int l1_of(int cls) { return cls >> 4; }
__device__ __forceinline__ int l2_of(int cls) { return (cls >> 2) & 3; }
__device__ __forceinline__ int l3_of(int cls) { return cls & 3; }

// The edge chunk of this CTA: [first, last).
__device__ __forceinline__ void chunk_range(int edges, int chunk, int& first, int& last) {
  first = min(edges, int(blockIdx.x) * chunk);
  last = min(edges, first + chunk);
}

template <typename T, int kKind>
__global__ void __launch_bounds__(128) contracter_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ w,
    const int32_t* __restrict__ index, const int4* __restrict__ paths, const T* __restrict__ g,
    const T* __restrict__ dgx1, const T* __restrict__ dgx2, const T* __restrict__ dgw,
    T* __restrict__ out3, T* __restrict__ out1, T* __restrict__ out2, T* __restrict__ partial,
    Problem pr, int chunk, const __grid_constant__ cg::Values<T> cgv) {
  extern __shared__ __align__(16) char smem[];
  const int tile = pr.tile, c = threadIdx.x;
  Stage<T> st = carve<T>(smem, pr, kKind);
  for (int q = c; q < pr.paths; q += blockDim.x) st.path[q] = paths[q];
  int first, last;
  chunk_range(pr.edges, chunk, first, last);
  for (int c0 = 0; c0 < pr.u; c0 += tile) {
    __syncthreads();
    stage_row(st.w, w, pr.b, pr.u, c0, tile);
    if constexpr (kKind == kSecond) stage_row(st.dw, dgw, pr.b, pr.u, c0, tile);
    if constexpr (kKind != kForward) zero_rows(st.gw, pr.b, tile);
    for (int e = first; e < last; ++e) {
      const int atom = index[e];
      const bool valid = atom >= 0 && atom < pr.atoms;
      __syncthreads();
      stage_row(st.x1, x1 + size_t(e) * pr.s1 * pr.u, pr.s1, pr.u, c0, tile);
      stage_row(st.x2, valid ? x2 + size_t(atom) * pr.s2 * pr.u : nullptr, pr.s2, pr.u, c0, tile);
      zero_rows(st.acc3, pr.o, tile);
      if constexpr (kKind != kForward) {
        stage_row(st.g, g + size_t(e) * pr.o * pr.u, pr.o, pr.u, c0, tile);
        zero_rows(st.acc1, pr.s1, tile);
        zero_rows(st.acc2, pr.s2, tile);
      }
      if constexpr (kKind == kSecond) {
        stage_row(st.d1, dgx1 + size_t(e) * pr.s1 * pr.u, pr.s1, pr.u, c0, tile);
        stage_row(st.d2, valid ? dgx2 + size_t(atom) * pr.s2 * pr.u : nullptr, pr.s2, pr.u, c0, tile);
      }
      __syncthreads();
      for (int pth = 0; pth < pr.paths; ++pth) {
        const int4 path = st.path[pth];
        const int cls = path.x;
        Blocks<T> k;
        k.off1 = path.y; k.off2 = path.z; k.off3 = path.w;
        load_block(st.x1, k.off1, 2 * l1_of(cls) + 1, tile, c, k.a);
        load_block(st.x2, k.off2, 2 * l2_of(cls) + 1, tile, c, k.b);
        k.wp = st.w[pth * tile + c];
        if constexpr (kKind != kForward) load_block(st.g, k.off3, 2 * l3_of(cls) + 1, tile, c, k.g);
        if constexpr (kKind == kSecond) {
          load_block(st.d1, k.off1, 2 * l1_of(cls) + 1, tile, c, k.da);
          load_block(st.d2, k.off2, 2 * l2_of(cls) + 1, tile, c, k.db);
          k.dwp = st.dw[pth * tile + c];
        }
        const int pidx = pth;
        if constexpr (kKind == kForward) {
          switch (cls) { MOLNEX_CG_CLASSES(MOLNEX_CG_FORWARD) default: break; }
        } else if constexpr (kKind == kFirst) {
          const int p = pidx;
          switch (cls) { MOLNEX_CG_CLASSES(MOLNEX_CG_FIRST) default: break; }
        } else {
          const int p = pidx;
          switch (cls) { MOLNEX_CG_CLASSES(MOLNEX_CG_SECOND) default: break; }
        }
      }
      __syncthreads();
      if constexpr (kKind == kForward) {
        store_rows(out3 + size_t(e) * pr.o * pr.u, st.acc3, pr.o, pr.u, c0, tile);
      } else {
        store_rows(out1 + size_t(e) * pr.s1 * pr.u, st.acc1, pr.s1, pr.u, c0, tile);
        if (!valid) zero_rows(st.acc2, pr.s2, tile);
        __syncthreads();
        store_rows(out2 + size_t(e) * pr.s2 * pr.u, st.acc2, pr.s2, pr.u, c0, tile);
        if constexpr (kKind == kSecond) store_rows(out3 + size_t(e) * pr.o * pr.u, st.acc3, pr.o, pr.u, c0, tile);
      }
    }
    if constexpr (kKind != kForward) {
      __syncthreads();
      store_rows(partial + size_t(blockIdx.x) * pr.b * pr.u, st.gw, pr.b, pr.u, c0, tile);
    }
  }
}

template <typename T>
__global__ void reduce_parts_kernel(const T* __restrict__ partial, T* __restrict__ out, int parts,
                                    size_t stride) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= stride) return;
  T acc = T(0);
  for (int part = 0; part < parts; ++part) acc += partial[size_t(part) * stride + idx];
  out[idx] = acc;
}

__global__ void atom_row_offsets(const int32_t* __restrict__ keys, int64_t edges, int64_t atoms,
                                 int32_t* __restrict__ rowptr) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row > atoms) return;
  int64_t lo = 0, hi = edges;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (keys[mid] < row) lo = mid + 1; else hi = mid;
  }
  rowptr[row] = int32_t(lo);
}

// Stable CSR of the edges by atom; an out-of-range id lands beyond ``atoms``
// (full-width keys) and is dropped by the row search.
void atom_csr(const at::Tensor& index, int64_t atoms, at::Tensor& order, at::Tensor& rowptr) {
  const int64_t edges = index.numel();
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto sorted = at::empty_like(index);
  if (edges > 0) {
    auto iota = at::arange(edges, index.options());
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, index.const_data_ptr<int32_t>(), sorted.mutable_data_ptr<int32_t>(),
        iota.const_data_ptr<int32_t>(), order.mutable_data_ptr<int32_t>(), int(edges), 0, 32, stream));
    auto temp = at::empty({int64_t(bytes)}, index.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.mutable_data_ptr(), bytes, index.const_data_ptr<int32_t>(),
        sorted.mutable_data_ptr<int32_t>(), iota.const_data_ptr<int32_t>(),
        order.mutable_data_ptr<int32_t>(), int(edges), 0, 32, stream));
  }
  const int64_t threads = 256, blocks = (atoms + 1 + threads - 1) / threads;
  atom_row_offsets<<<unsigned(blocks), unsigned(threads), 0, stream>>>(
      sorted.const_data_ptr<int32_t>(), edges, atoms, rowptr.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ``out[atom] = Σ_{edges of atom} values[e]`` in the schedule's order; an edge
// whose id is out of range is in no row.
template <typename T>
at::Tensor sum_by_atom(const at::Tensor& values, const at::Tensor& index, int64_t atoms) {
  auto out = at::zeros({atoms, values.size(1)}, values.options());
  if (atoms == 0 || values.size(0) == 0) return out;
  auto order = at::empty({values.size(0)}, index.options());
  auto rowptr = at::empty({atoms + 1}, index.options());
  atom_csr(index, atoms, order, rowptr);
  TORCH_CHECK(atoms <= std::numeric_limits<int>::max(), "too many atoms for one grid");
  segment_sum_kernel<T><<<unsigned(atoms), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      values.const_data_ptr<T>(), order.const_data_ptr<int32_t>(), rowptr.const_data_ptr<int32_t>(),
      out.mutable_data_ptr<T>(), values.size(1), values.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int tile_for(int channels) {
  if (channels <= 32) return 32;
  if (channels <= 64) return 64;
  return 128;
}

template <typename T, typename Kernel>
void allow_smem(Kernel kernel, size_t bytes) {
  const size_t limit = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  TORCH_CHECK(bytes <= limit, "contracter needs ", bytes, " bytes of shared memory per CTA; this device allows ", limit);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}

template <typename T>
cg::Values<T> pack_cg(at::ArrayRef<double> values) {
  TORCH_CHECK(values.size() == size_t(cg::kTotal), "the Clebsch-Gordan table has ", cg::kTotal,
              " entries, got ", values.size());
  cg::Values<T> out;
  for (int t = 0; t < cg::kTotal; ++t) out.v[t] = T(values[t]);
  return out;
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

int weight_segments(const at::Tensor& w, int channels) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous(), "weight must be a contiguous CUDA tensor");
  int64_t segments = 0;
  if (w.dim() == 2) {
    TORCH_CHECK(w.size(1) == channels, "weight width must equal the channel count");
    segments = w.size(0);
  } else if (w.dim() == 1) {
    TORCH_CHECK(channels > 0 && w.numel() % channels == 0, "weight numel must be a multiple of channels");
    segments = w.numel() / channels;
  } else {
    TORCH_CHECK(false, "weight must be (paths, channels) or a flat vector of that size");
  }
  TORCH_CHECK(segments > 0 && segments <= kMaxPaths, "weight segments must be 1..", kMaxPaths);
  return int(segments);
}

Problem decode(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& w, const at::Tensor& paths,
               int64_t channels, int64_t segments_out) {
  TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "contracter operands must be CUDA tensors");
  TORCH_CHECK(x1.scalar_type() == at::kFloat || x1.scalar_type() == at::kDouble,
              "contracter supports float32 and float64");
  TORCH_CHECK(x2.scalar_type() == x1.scalar_type() && w.scalar_type() == x1.scalar_type() &&
                  x2.device() == x1.device() && w.device() == x1.device(),
              "operands must share dtype and device");
  TORCH_CHECK(x1.dim() == 2 && x2.dim() == 2 && x1.is_contiguous() && x2.is_contiguous(),
              "pair and atom features must be contiguous matrices");
  TORCH_CHECK(channels > 0 && channels <= std::numeric_limits<int>::max(), "channels must be positive");
  TORCH_CHECK(segments_out >= 0 && segments_out <= kMaxSegments, "output components exceed ", kMaxSegments);
  TORCH_CHECK(x1.size(1) % channels == 0 && x2.size(1) % channels == 0,
              "feature widths must be multiples of the channel count");
  const int64_t s1 = x1.size(1) / channels, s2 = x2.size(1) / channels;
  TORCH_CHECK(s1 > 0 && s1 <= kMaxSegments && s2 > 0 && s2 <= kMaxSegments,
              "operand components exceed ", kMaxSegments);
  TORCH_CHECK(x1.size(0) <= std::numeric_limits<int>::max() && x2.size(0) <= std::numeric_limits<int>::max(),
              "edge or atom count exceeds int32");
  TORCH_CHECK(paths.device() == x1.device() && paths.scalar_type() == at::kInt && paths.dim() == 2 &&
                  paths.size(1) == 4 && paths.is_contiguous(),
              "paths must be a contiguous int32 (P, 4) table on the feature device");
  Problem p;
  p.edges = int(x1.size(0));
  p.atoms = int(x2.size(0));
  p.s1 = int(s1);
  p.s2 = int(s2);
  p.b = weight_segments(w, int(channels));
  p.o = int(segments_out);
  p.u = int(channels);
  p.paths = int(paths.size(0));
  p.tile = tile_for(p.u);
  TORCH_CHECK(p.paths == p.b, "one weight segment per path: ", p.b, " weights for ", p.paths, " paths");
  return p;
}

// A compiled table id, or -1: the generic class-blocked kernels.
bool compiled(int64_t config, const Problem& p, int64_t nterms) {
  if (config < 0) return false;
#define MOLNEX_CT_MATCH(ID)                                                                   \
  if (config == ID) {                                                                         \
    using Tb = ct::Table<ID>;                                                                 \
    TORCH_CHECK(p.s1 == Tb::NS1 && p.s2 == Tb::NS2 && p.b == Tb::NB && p.o == Tb::NO &&       \
                    nterms == Tb::NT,                                                         \
                "compiled contracter table ", config, " does not match the operands");        \
    return true;                                                                              \
  }
  MOLNEX_CT_CONFIGS(MOLNEX_CT_MATCH)
#undef MOLNEX_CT_MATCH
  TORCH_CHECK(false, "no compiled contracter table ", config);
}

int parts_for(const Problem& p, int elem) {
  const int64_t row = int64_t(p.b) * p.u * elem;
  int64_t parts = std::max<int64_t>(1, (int64_t(p.edges) + 31) / 32);
  parts = std::min<int64_t>({parts, kMaxParts, std::max<int64_t>(1, kPartialBudget / std::max<int64_t>(row, 1))});
  return int(parts);
}

}  // namespace

at::Tensor contracter_table(int64_t id) {
#define MOLNEX_CT_TABLE(ID)                                                              \
  if (id == ID) {                                                                        \
    using Tb = ct::Table<ID>;                                                            \
    auto table = at::empty({Tb::NT, 4}, at::TensorOptions().dtype(at::kShort));         \
    auto* out = table.mutable_data_ptr<int16_t>();                                       \
    for (int t = 0; t < Tb::NT; ++t) {                                                   \
      const auto m = Tb::terms[t];                                                       \
      out[4 * t] = m.w; out[4 * t + 1] = m.i; out[4 * t + 2] = m.j; out[4 * t + 3] = m.o; \
    }                                                                                    \
    return table;                                                                        \
  }
  MOLNEX_CT_CONFIGS(MOLNEX_CT_TABLE)
#undef MOLNEX_CT_TABLE
  return at::empty({0, 4}, at::TensorOptions().dtype(at::kShort));
}

at::Tensor contracter_cuda(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& w,
                           const at::Tensor& index, const at::Tensor& paths, at::ArrayRef<double> cg_values,
                           at::ArrayRef<double> values, int64_t config, int64_t channels,
                           int64_t segments_out) {
  const c10::cuda::CUDAGuard guard(x1.device());
  Problem p = decode(x1, x2, w, paths, channels, segments_out);
  auto y = at::empty({p.edges, int64_t(p.o) * p.u}, x1.options());
  if (p.edges == 0 || p.o == 0) return y;
  auto ids = as_int_index(index, x1, p.edges);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (compiled(config, p, int64_t(values.size()))) {
    const int parts = parts_for(p, int(x1.element_size()));
    ct::Args a{x1, x2, w, ids, {}, {}, {}, {}, y, {}, {}, {}, {}, values, p.u, p.tile, parts,
               (p.edges + parts - 1) / parts};
#define MOLNEX_CT_RUN(ID) if (config == ID) ct::run_forward<ID>(a);
    MOLNEX_CT_CONFIGS(MOLNEX_CT_RUN)
#undef MOLNEX_CT_RUN
    return y;
  }
  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "contracter_forward", [&] {
    const auto cgv = pack_cg<scalar_t>(cg_values);
    const int parts = parts_for(p, int(x1.element_size()));
    const int chunk = (p.edges + parts - 1) / parts;
    const size_t smem = stage_bytes(p, int(x1.element_size()), kForward);
    allow_smem<scalar_t>(contracter_kernel<scalar_t, kForward>, smem);
    contracter_kernel<scalar_t, kForward><<<unsigned(parts), p.tile, smem, stream>>>(
        x1.const_data_ptr<scalar_t>(), x2.const_data_ptr<scalar_t>(), w.const_data_ptr<scalar_t>(),
        ids.const_data_ptr<int32_t>(), reinterpret_cast<const int4*>(paths.const_data_ptr<int32_t>()),
        nullptr, nullptr, nullptr, nullptr, y.mutable_data_ptr<scalar_t>(), nullptr, nullptr, nullptr, p,
        chunk, cgv);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> contracter_adjoints_cuda(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& w, const at::Tensor& index,
    const at::Tensor& g, const at::Tensor& dgx1, const at::Tensor& dgx2, const at::Tensor& dgw,
    const at::Tensor& dgy, const at::Tensor& paths, at::ArrayRef<double> cg_values,
    at::ArrayRef<double> values, int64_t config, bool second, int64_t channels,
    int64_t segments_out) {
  const c10::cuda::CUDAGuard guard(x1.device());
  Problem p = decode(x1, x2, w, paths, channels, segments_out);
  const int64_t width = int64_t(p.o) * p.u;
  TORCH_CHECK(g.device() == x1.device() && g.scalar_type() == x1.scalar_type() && g.is_contiguous() &&
                  g.dim() == 2 && g.size(0) == p.edges && g.size(1) == width,
              "cotangent g must be a contiguous (edges, output_dim) matrix");
  TORCH_CHECK(dgy.is_cuda() && dgy.scalar_type() == x1.scalar_type(),
              "dgy must be a CUDA tensor of the compute dtype");
  auto gx1 = at::empty({p.edges, x1.size(1)}, x1.options());
  auto gx2_edge = at::empty({p.edges, x2.size(1)}, x1.options());
  auto gw = at::zeros_like(w);
  auto grad_g = at::empty({second ? p.edges : int64_t(0), width}, x1.options());
  if (p.edges == 0) return {gx1, at::zeros({p.atoms, x2.size(1)}, x1.options()), gw, grad_g};
  if (second) {
    TORCH_CHECK(dgx1.sizes() == x1.sizes() && dgx2.sizes() == x2.sizes() && dgw.sizes() == w.sizes() &&
                    dgx1.is_contiguous() && dgx2.is_contiguous() && dgw.is_contiguous() &&
                    dgx1.scalar_type() == x1.scalar_type() && dgx2.scalar_type() == x1.scalar_type() &&
                    dgw.scalar_type() == x1.scalar_type() && dgx1.device() == x1.device() &&
                    dgx2.device() == x1.device() && dgw.device() == x1.device(),
                "second-derivative cotangents must match the pair, the atom, and the weight");
  }
  auto ids = as_int_index(index, x1, p.edges);
  const auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor gx2;
  // The compiled second-order kernel holds more rows than the register file
  // takes (nvcc demotes its accumulators to local memory: 125 ms against the
  // class-blocked kernel's 35 ms on an Allegro-OAM-L layer, GH200 2026-10-27),
  // so the second order runs the class-blocked kernel.
  const bool use_compiled = !second && compiled(config, p, int64_t(values.size()));
  // Defined outside the dispatch macro's argument: a directive inside one is undefined.
#define MOLNEX_CT_RUN(ID)                        \
  if (config == ID) {                            \
    if (second) ct::run_second<ID>(a);           \
    else ct::run_first<ID>(a);                   \
  }
  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "contracter_adjoints", [&] {
    const int elem = int(x1.element_size());
    const int parts = parts_for(p, elem);
    const int chunk = (p.edges + parts - 1) / parts;
    auto partial = at::empty({parts, int64_t(p.b) * p.u}, x1.options());
    if (use_compiled) {
      ct::Args a{x1, x2, w, ids, g, dgx1, dgx2, dgw, {}, gx1, gx2_edge, grad_g, partial, values,
                 p.u, p.tile, parts, chunk};
      MOLNEX_CT_CONFIGS(MOLNEX_CT_RUN)
      const size_t stride = size_t(p.b) * p.u;
      reduce_parts_kernel<scalar_t><<<unsigned((stride + 255) / 256), 256, 0, stream>>>(
          partial.const_data_ptr<scalar_t>(), gw.mutable_data_ptr<scalar_t>(), parts, stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      gx2 = sum_by_atom<scalar_t>(gx2_edge, ids, p.atoms);
      return;
    }
    const auto cgv = pack_cg<scalar_t>(cg_values);
    const int kind = second ? kSecond : kFirst;
    const size_t smem = stage_bytes(p, elem, kind);
    auto launch = [&](auto kernel) {
      allow_smem<scalar_t>(kernel, smem);
      kernel<<<unsigned(parts), p.tile, smem, stream>>>(
          x1.const_data_ptr<scalar_t>(), x2.const_data_ptr<scalar_t>(), w.const_data_ptr<scalar_t>(),
          ids.const_data_ptr<int32_t>(), reinterpret_cast<const int4*>(paths.const_data_ptr<int32_t>()),
          g.const_data_ptr<scalar_t>(), second ? dgx1.const_data_ptr<scalar_t>() : nullptr,
          second ? dgx2.const_data_ptr<scalar_t>() : nullptr, second ? dgw.const_data_ptr<scalar_t>() : nullptr,
          second ? grad_g.mutable_data_ptr<scalar_t>() : nullptr, gx1.mutable_data_ptr<scalar_t>(),
          gx2_edge.mutable_data_ptr<scalar_t>(), partial.mutable_data_ptr<scalar_t>(), p, chunk, cgv);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (second) launch(contracter_kernel<scalar_t, kSecond>);
    else launch(contracter_kernel<scalar_t, kFirst>);
    const size_t stride = size_t(p.b) * p.u;
    reduce_parts_kernel<scalar_t><<<unsigned((stride + 255) / 256), 256, 0, stream>>>(
        partial.const_data_ptr<scalar_t>(), gw.mutable_data_ptr<scalar_t>(), parts, stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gx2 = sum_by_atom<scalar_t>(gx2_edge, ids, p.atoms);
  });
#undef MOLNEX_CT_RUN
  return {gx1, gx2, gw, grad_g};
}

}  // namespace molnex::equivariance
