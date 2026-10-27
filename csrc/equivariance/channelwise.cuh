// Indexed channelwise tensor product, MACE's natural-parity O3 family, as
// compile-time term tables.
//
// The product is ``out[k, c] += C[i, j, k] · a[i, c] · b[j] · w[p, c]`` over the
// paths ``(l1, l2, l3)`` of cuEquivariance's ``channelwise_tensor_product`` for
// ``irreps_in1 = U×0e + U×1o + … + U×L1(natural parity)``,
// ``irreps_in2 = 0e + 1o + … + L2`` and the output filter ``0e + … + L2``. The
// Clebsch–Gordan *values* come from the cuEquivariance descriptor at
// construction (kernel parameters, so they are constant-bank operands of the
// FMAs); only their *positions* are enumerated here, by the selection rule of
// the real spherical-harmonic basis: a product of components ``m1`` and ``m2``
// has non-zero coefficients only at ``m3 ∈ {±(|m1|+|m2|), ±(||m1|−|m2||)}``,
// cosine-type (``m3 ≥ 0``) when both factors are of the same type and
// sine-type (``m3 < 0``) otherwise, and ``m3 = m2`` / ``m3 = m1`` when the other
// is zonal. The Python descriptor checks every non-zero cuEquivariance
// coefficient against this candidate set, so a basis convention this rule does
// not cover is refused at construction rather than computed wrongly.
//
// Every term index is a compile-time constant after the fold over the table,
// so operands and accumulators stay in registers — no dynamic register
// indexing, no shared-memory staging of operands.
#pragma once

#include <c10/macros/Macros.h>

#include <utility>

#include "terms.hpp"

namespace molnex::equivariance {

// The coefficient table travels as a kernel parameter: constant bank, read as
// an FMA operand, no per-call device copy, captured by value in a CUDA graph.
// Kernels take it ``__grid_constant__``: the unrolled term loop reaches it
// through a reference, and without the annotation the compiler copies the
// whole table into local memory (a multi-kilobyte stack frame per thread).
template <typename T, int NT>
struct Coefficients {
  T c[NT];
};

template <int NT, class F>
__device__ __forceinline__ void for_each_term(F&& f) {
  [&]<int... Ts>(std::integer_sequence<int, Ts...>) {
    (f.template operator()<Ts>(), ...);
  }(std::make_integer_sequence<int, NT>{});
}


// One segment of the schedule, consumed in chunks of U edges whose ids,
// source rows and angular rows (``b``, and ``db`` when given) are staged in
// shared memory by the whole CTA: the per-edge critical path is then the
// channel-operand loads alone, not the dependent chain order → source → row,
// and the per-edge scalars common to every channel cost no registers.
// ``f(e, s, q)`` runs for every edge of the segment, in schedule order, by all
// threads; ``q`` indexes the staged rows of the chunk.
template <int U, int Unroll, int NB, typename T, class F>
__device__ __forceinline__ void for_each_edge(const int64_t* __restrict__ source,
                                              const int32_t* __restrict__ order,
                                              const T* __restrict__ b, const T* __restrict__ db,
                                              int32_t begin, int32_t end, int64_t edges,
                                              int64_t rows_a, int32_t* s_edge, int32_t* s_source,
                                              T* s_b, T* s_db, F&& f) {
  const int c = threadIdx.x;
  for (int32_t chunk = begin; chunk < end; chunk += U) {
    const int count = min(U, end - chunk);
    __syncthreads();  // the previous chunk's staged rows are consumed
    if (c < count) {
      const int64_t e = order[chunk + c];
      CUDA_KERNEL_ASSERT(e >= 0 && e < edges);
      const int64_t s = source[e];
      CUDA_KERNEL_ASSERT(s >= 0 && s < rows_a);
      s_edge[c] = int32_t(e);
      s_source[c] = int32_t(s);
#pragma unroll
      for (int j = 0; j < NB; ++j) s_b[c * NB + j] = b[e * NB + j];
      if (db != nullptr) {
#pragma unroll
        for (int j = 0; j < NB; ++j) s_db[c * NB + j] = db[e * NB + j];
      }
    }
    __syncthreads();
    // Unrolled so the next edge's operand loads are in flight during this
    // edge's contraction: with one CTA per row the warps alone cannot hide
    // the latency.
#pragma unroll Unroll
    for (int q = 0; q < count; ++q) f(int64_t(s_edge[q]), int64_t(s_source[q]), q);
  }
}

// The CTA's share of row ``n``: rows split over ``gridDim.y`` CTAs when the
// system is too small to fill the device; partial sums are reduced afterwards.
__device__ __forceinline__ void row_range(const int32_t* __restrict__ rowptr, int64_t n,
                                          int64_t edges, int32_t& begin, int32_t& end) {
  begin = rowptr[n];
  end = rowptr[n + 1];
  CUDA_KERNEL_ASSERT(begin >= 0 && begin <= end && end <= edges);
  const int32_t per = (end - begin + gridDim.y - 1) / gridDim.y;
  begin = begin + int32_t(blockIdx.y) * per;
  end = min(end, begin + per);
}

// ── product: one CTA per target row, one thread per channel ─────────────────
//
// The sum over a row's incoming edges runs in registers in the schedule's
// (stable) order and is written once — no atomics, bitwise-reproducible.
// Resident CTAs per SM the register allocation must allow — the measured
// optimum for each kernel on a GH200 (job 4314633 vs 4314372): the float
// product at 4 (≤ 128 registers, no spill), the float first-order adjoint at 3
// (165 registers; 4 forces 128 and spills), everything else at 2 (255).
template <typename T, bool Heavy>
constexpr int kMinBlocks = sizeof(T) == 4 ? (Heavy ? 2 : 4) : 2;

// ``channels`` is the row stride of the channel operands; a CTA of ``U`` threads
// walks it in tiles of ``U`` (one tile at MACE's 128, eight at OMol's 1024
// first-layer width), re-walking the row's edges per tile.
template <typename T, int L1, int L2, int U>
__global__ void __launch_bounds__(U, kMinBlocks<T, false>) product_kernel(
    const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ w,
    const int64_t* __restrict__ source, const int32_t* __restrict__ order,
    const int32_t* __restrict__ rowptr, T* __restrict__ out, int64_t rows_a,
    int64_t edges, int64_t channels, const __grid_constant__ Coefficients<T, Config<L1, L2>::NT> coef) {
  using Cfg = Config<L1, L2>;
  constexpr int NA = Cfg::NA, NB = Cfg::NB, NP = Cfg::NP, NK = Cfg::NK;
  const int64_t n = blockIdx.x, rows = gridDim.x;
  int32_t begin, end;
  row_range(rowptr, n, edges, begin, end);
  __shared__ int32_t s_edge[U], s_source[U];
  __shared__ T s_b[U * NB];
  for (int64_t c0 = 0; c0 < channels; c0 += U) {
    const int64_t c = c0 + threadIdx.x;
    T acc[NK];
#pragma unroll
    for (int k = 0; k < NK; ++k) acc[k] = T(0);
    for_each_edge<U, 2, NB, T>(source, order, b, nullptr, begin, end, edges, rows_a, s_edge,
                               s_source, s_b, nullptr, [&](int64_t e, int64_t s, int q) {
      const T* bv = s_b + q * NB;
      T av[NA], wv[NP];
#pragma unroll
      for (int i = 0; i < NA; ++i) av[i] = a[(s * NA + i) * channels + c];
#pragma unroll
      for (int p = 0; p < NP; ++p) wv[p] = w[(e * NP + p) * channels + c];
      for_each_term<Cfg::NT>([&]<int t>() {
        constexpr Term m = kTable<L1, L2>[t];
        acc[m.k] += coef.c[t] * (av[m.i] * bv[m.j]) * wv[m.p];
      });
    });
#pragma unroll
    for (int k = 0; k < NK; ++k) out[((blockIdx.y * rows + n) * NK + k) * channels + c] = acc[k];
  }
}

// ``out[n] = Σ_k partial[k, n]`` over the gridDim.y partials of a split kernel.
template <typename T>
__global__ void reduce_partials_kernel(const T* __restrict__ partial, T* __restrict__ out,
                                       int64_t splits, int64_t numel) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  T acc = T(0);
  for (int64_t k = 0; k < splits; ++k) acc += partial[k * numel + i];
  out[i] = acc;
}

// Sum of the per-channel partials of one edge over the CTA's channels, in a
// fixed order: warp shuffles, then the warps. ``values`` holds the partials,
// ``red`` is the CTA's scratch; lanes ``c < N`` receive the sums.
template <typename T, int N, int U>
__device__ __forceinline__ void reduce_channels(T (&values)[N], T (*red)[N], int c) {
  // A CTA narrower than a warp (U = 8, 16: tiny models) is one partial warp.
  constexpr int kWarps = U < 32 ? 1 : U / 32;
  constexpr int kLanes = U < 32 ? U : 32;
  constexpr unsigned kMask = U < 32 ? (1u << U) - 1u : 0xffffffffu;
  const int lane = c % 32, warp = c / 32;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    T v = values[j];
#pragma unroll
    for (int off = kLanes / 2; off > 0; off /= 2) v += __shfl_xor_sync(kMask, v, off);
    if (lane == 0) red[warp][j] = v;
  }
  __syncthreads();
  if (c < N) {
    T v = T(0);
#pragma unroll
    for (int q = 0; q < kWarps; ++q) v += red[q][c];
    values[0] = v;  // the lane's own sum
  }
  __syncthreads();
}

// ── adjoints: one CTA per target row ────────────────────────────────────────
//
// The cotangent row ``g[t]`` is read once per node, not once per edge: the
// 5120-wide row at MACE widths is four times an edge's weights. Per edge the
// two edge-local adjoints are written directly; the node-operand adjoint is
// written per edge to ``ga_edge`` and summed by source afterwards
// (``segment_sum``), so no atomics anywhere.
//
// Second: the adjoint of the adjoints. With cotangents ``da, db, dw`` of
// ``(ga, gb, gw)``, every output is a sum of the same contractions with one
// operand substituted, and ``grad_g[t]`` accumulates per target in registers.
// Part splits the double-precision second order, whose operands and
// accumulators exceed the register file in one pass: 1 computes grad_g only,
// 2 the edge adjoints and ga only; 0 is everything (first order, float).
template <typename T, int L1, int L2, int U, bool Second, int Part>
__global__ void __launch_bounds__(U, Second || sizeof(T) == 8 ? 2 : 3) adjoint_kernel(
    const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ w,
    const T* __restrict__ g, const T* __restrict__ da, const T* __restrict__ db,
    const T* __restrict__ dw, const int64_t* __restrict__ source,
    const int32_t* __restrict__ order, const int32_t* __restrict__ rowptr,
    T* __restrict__ ga_edge, T* __restrict__ gb, T* __restrict__ gw,
    T* __restrict__ grad_g, int64_t rows_a, int64_t edges, int64_t channels,
    const __grid_constant__ Coefficients<T, Config<L1, L2>::NT> coef) {
  using Cfg = Config<L1, L2>;
  constexpr int NA = Cfg::NA, NB = Cfg::NB, NP = Cfg::NP, NK = Cfg::NK;
  constexpr int kWarps = U < 32 ? 1 : U / 32;
  __shared__ T red[kWarps][NB];
  __shared__ int32_t s_edge[U], s_source[U];
  __shared__ T s_b[U * NB], s_db[Second ? U * NB : 1];
  const int64_t n = blockIdx.x, rows = gridDim.x;
  int32_t begin, end;
  row_range(rowptr, n, edges, begin, end);
  constexpr bool kEdge = Part != 1, kGradG = Second && Part != 2;
  for (int64_t c0 = 0; c0 < channels; c0 += U) {
  const int64_t c = c0 + threadIdx.x;
  // The angular adjoint sums over every channel: the tiles add up in gb.
  const bool first_tile = c0 == 0;
  T gv[kEdge ? NK : 1];
  if constexpr (kEdge) {
#pragma unroll
    for (int k = 0; k < NK; ++k) gv[k] = g[(n * NK + k) * channels + c];
  }
  T acc_g[kGradG ? NK : 1];
#pragma unroll
  for (int k = 0; k < (kGradG ? NK : 1); ++k) acc_g[k] = T(0);
  for_each_edge<U, Second ? 1 : 2, NB, T>(source, order, b, Second ? db : nullptr, begin, end,
                                          edges, rows_a, s_edge, s_source, s_b, s_db,
                                          [&](int64_t e, int64_t s, int q) {
    const T* bv = s_b + q * NB;
    T av[NA], wv[NP], ga[NA], gbv[NB], gwv[NP];
#pragma unroll
    for (int i = 0; i < NA; ++i) av[i] = a[(s * NA + i) * channels + c];
#pragma unroll
    for (int p = 0; p < NP; ++p) wv[p] = w[(e * NP + p) * channels + c];
#pragma unroll
    for (int i = 0; i < NA; ++i) ga[i] = T(0);
#pragma unroll
    for (int j = 0; j < NB; ++j) gbv[j] = T(0);
#pragma unroll
    for (int p = 0; p < NP; ++p) gwv[p] = T(0);
    if constexpr (!Second) {
      for_each_term<Cfg::NT>([&]<int t>() {
        constexpr Term m = kTable<L1, L2>[t];
        const T cg = coef.c[t] * gv[m.k];
        ga[m.i] += cg * (bv[m.j] * wv[m.p]);
        gbv[m.j] += cg * (av[m.i] * wv[m.p]);
        gwv[m.p] += cg * (av[m.i] * bv[m.j]);
      });
    } else {
      const T* dbv = s_db + q * NB;
      T dav[NA], dwv[NP];
#pragma unroll
      for (int i = 0; i < NA; ++i) dav[i] = da[(s * NA + i) * channels + c];
#pragma unroll
      for (int p = 0; p < NP; ++p) dwv[p] = dw[(e * NP + p) * channels + c];
      for_each_term<Cfg::NT>([&]<int t>() {
        constexpr Term m = kTable<L1, L2>[t];
        if constexpr (kGradG) {
          // grad_g: the product with each operand replaced by its cotangent
          acc_g[m.k] += coef.c[t] * ((dav[m.i] * bv[m.j]) * wv[m.p] +
                                     (av[m.i] * dbv[m.j]) * wv[m.p] +
                                     (av[m.i] * bv[m.j]) * dwv[m.p]);
        }
        if constexpr (kEdge) {
          const T cg = coef.c[t] * gv[m.k];
          // grad_a: through gb (db) and gw (dw); ga itself does not depend on a
          ga[m.i] += cg * (dbv[m.j] * wv[m.p] + bv[m.j] * dwv[m.p]);
          // grad_b: through ga (da) and gw (dw)
          gbv[m.j] += cg * (dav[m.i] * wv[m.p] + av[m.i] * dwv[m.p]);
          // grad_w: through ga (da) and gb (db)
          gwv[m.p] += cg * (dav[m.i] * bv[m.j] + av[m.i] * dbv[m.j]);
        }
      });
    }
    if constexpr (kEdge) {
#pragma unroll
      for (int i = 0; i < NA; ++i) ga_edge[(e * NA + i) * channels + c] = ga[i];
#pragma unroll
      for (int p = 0; p < NP; ++p) gw[(e * NP + p) * channels + c] = gwv[p];
      reduce_channels<T, NB, U>(gbv, red, int(threadIdx.x));
      if (threadIdx.x < NB) {
        if (first_tile) gb[e * NB + threadIdx.x] = gbv[0];
        else gb[e * NB + threadIdx.x] += gbv[0];
      }
    }
  });
  if constexpr (kGradG) {
#pragma unroll
    for (int k = 0; k < NK; ++k)
      grad_g[((blockIdx.y * rows + n) * NK + k) * channels + c] = acc_g[k];
  }
  }  // channel tiles
}

// ``out[r, :] = Σ_{q in segment r} values[order[q], :]`` in the schedule's order.
template <typename T>
__global__ void __launch_bounds__(256) segment_sum_kernel(
    const T* __restrict__ values, const int32_t* __restrict__ order,
    const int32_t* __restrict__ rowptr, T* __restrict__ out, int64_t width, int64_t edges) {
  __shared__ int32_t s_edge[256];
  const int64_t n = blockIdx.x;
  const int32_t begin = rowptr[n], end = rowptr[n + 1];
  CUDA_KERNEL_ASSERT(begin >= 0 && begin <= end && end <= edges);
  for (int64_t col = threadIdx.x; col < width; col += 256) out[n * width + col] = T(0);
  for (int32_t chunk = begin; chunk < end; chunk += 256) {
    const int count = min(256, end - chunk);
    __syncthreads();
    if (threadIdx.x < count) {
      const int64_t e = order[chunk + threadIdx.x];
      CUDA_KERNEL_ASSERT(e >= 0 && e < edges);
      s_edge[threadIdx.x] = int32_t(e);
    }
    __syncthreads();
    for (int64_t col = threadIdx.x; col < width; col += 256) {
      T acc = T(0);
#pragma unroll 4
      for (int q = 0; q < count; ++q) acc += values[int64_t(s_edge[q]) * width + col];
      out[n * width + col] += acc;
    }
  }
}

}  // namespace molnex::equivariance
