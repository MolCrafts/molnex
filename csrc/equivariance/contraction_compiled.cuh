// Symmetric contraction with a compile-time term table: MACE's family
// (``contraction_tables.hpp``, generated), every operand and accumulator in
// registers, the coefficient values a kernel parameter.
//
// One CTA per node and channel tile (``tile`` threads, one channel each; wider
// rows loop over tiles). A thread holds its node's row ``x[NS + 1]`` (the
// sentinel ``x[NS] = 1`` stands in for a missing slot, its cotangent is 0),
// its element's weight row ``W[NB]`` and the accumulators. The table is sorted
// by (o, i, j, k, w): the terms of one monomial ``x[i] x[j] x[k]`` are a run,
// and what every term of the run shares — the monomial, the per-slot partial
// derivatives — is computed once at the run's start and applied once at its
// end, so a term costs two to six FMAs on registers and leaves the compiler
// no common subexpression to hoist across the unrolled table (which is what
// saturated the register file before). The weight adjoint is written per
// node (``G[n, w, c]``) and summed into the elements afterwards in a fixed
// order (``sum_by_element``); no atomics anywhere.
//
// Second order is two kernels: ``second_node`` (dx, dg) and ``second_weight``
// (the per-node dw partial).
#pragma once

#include <c10/macros/Macros.h>

#include <utility>

#include "channelwise.cuh"  // Coefficients, for_each_term
#include "contraction_tables.hpp"

namespace molnex::equivariance::sc {

template <typename T, int NS, int NB>
__device__ __forceinline__ void load_rows(const T* __restrict__ x, const T* __restrict__ w,
                                          int64_t node, int64_t element, int64_t u, int64_t c,
                                          T (&xr)[NS + 1], T (&wr)[NB]) {
#pragma unroll
  for (int s = 0; s < NS; ++s) xr[s] = x[(node * NS + s) * u + c];
  xr[NS] = T(1);
#pragma unroll
  for (int b = 0; b < NB; ++b) wr[b] = w[(element * NB + b) * u + c];
}

template <typename T, int N>
__device__ __forceinline__ void load_row(const T* __restrict__ src, int64_t row, int64_t u, int64_t c,
                                         T (&dst)[N]) {
#pragma unroll
  for (int s = 0; s < N; ++s) dst[s] = src[(row * N + s) * u + c];
}

template <typename T, int N>
__device__ __forceinline__ void store_row(T* __restrict__ dst, int64_t row, int64_t u, int64_t c,
                                          const T (&src)[N]) {
#pragma unroll
  for (int s = 0; s < N; ++s) dst[(row * N + s) * u + c] = src[s];
}

template <typename T, int N>
__device__ __forceinline__ void zero(T (&a)[N]) {
#pragma unroll
  for (int s = 0; s < N; ++s) a[s] = T(0);
}

// Term t starts a run (its monomial differs from term t-1's) / ends one.
template <int LIN, int LOUT, int DEG, int t>
constexpr bool starts_run() {
  using Tb = Table<LIN, LOUT, DEG>;
  if constexpr (t == 0) return true;
  else
    return Tb::terms[t - 1].i != Tb::terms[t].i || Tb::terms[t - 1].j != Tb::terms[t].j ||
           Tb::terms[t - 1].k != Tb::terms[t].k;
}
template <int LIN, int LOUT, int DEG, int t>
constexpr bool ends_run() {
  using Tb = Table<LIN, LOUT, DEG>;
  if constexpr (t + 1 == Tb::NT) return true;
  else
    return Tb::terms[t + 1].i != Tb::terms[t].i || Tb::terms[t + 1].j != Tb::terms[t].j ||
           Tb::terms[t + 1].k != Tb::terms[t].k;
}

template <typename T, int LIN, int LOUT, int DEG>
__global__ void __launch_bounds__(128, 2) forward_kernel(
    const T* __restrict__ x, const T* __restrict__ w, const int32_t* __restrict__ index,
    T* __restrict__ y, int64_t z, int64_t u,
    const __grid_constant__ Coefficients<T, Table<LIN, LOUT, DEG>::NT> coef) {
  using Tb = Table<LIN, LOUT, DEG>;
  constexpr int NS = Tb::NS, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  const int64_t node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < z;
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T acc[NO];
    zero(acc);
    if (valid) {
      T xr[NS + 1], wr[NB];
      load_rows<T, NS, NB>(x, w, node, element, u, c, xr, wr);
      T mono = T(0);
      for_each_term<NT>([&]<int t>() {
        constexpr Term m = Tb::terms[t];
        if constexpr (starts_run<LIN, LOUT, DEG, t>()) mono = xr[m.i] * xr[m.j] * xr[m.k];
        acc[m.o] += (coef.c[t] * wr[m.w]) * mono;
      });
    }
    store_row<T, NO>(y, node, u, c, acc);
  }
}

// gx and the per-node weight partial G. Per run: r = Σ C · g[o] · W[w]; then
// gx[i] += x[j] x[k] r, gx[j] += x[i] x[k] r, gx[k] += x[i] x[j] r; per term
// G[w] += C · g[o] · x[i] x[j] x[k].
template <typename T, int LIN, int LOUT, int DEG>
__global__ void __launch_bounds__(128, 2) first_kernel(
    const T* __restrict__ x, const T* __restrict__ w, const int32_t* __restrict__ index,
    const T* __restrict__ g, T* __restrict__ gx, T* __restrict__ gw_node, int64_t z, int64_t u,
    const __grid_constant__ Coefficients<T, Table<LIN, LOUT, DEG>::NT> coef) {
  using Tb = Table<LIN, LOUT, DEG>;
  constexpr int NS = Tb::NS, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  const int64_t node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < z;
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T ax[NS + 1], aw[NB];
    zero(ax);
    zero(aw);
    if (valid) {
      T xr[NS + 1], wr[NB], gr[NO];
      load_rows<T, NS, NB>(x, w, node, element, u, c, xr, wr);
      load_row<T, NO>(g, node, u, c, gr);
      T mono = T(0), r = T(0);
      for_each_term<NT>([&]<int t>() {
        constexpr Term m = Tb::terms[t];
        if constexpr (starts_run<LIN, LOUT, DEG, t>()) {
          mono = xr[m.i] * xr[m.j] * xr[m.k];
          r = T(0);
        }
        const T cg = coef.c[t] * gr[m.o];
        r += cg * wr[m.w];
        aw[m.w] += cg * mono;
        if constexpr (ends_run<LIN, LOUT, DEG, t>()) {
          ax[m.i] += (xr[m.j] * xr[m.k]) * r;
          ax[m.j] += (xr[m.i] * xr[m.k]) * r;
          ax[m.k] += (xr[m.i] * xr[m.j]) * r;
        }
      });
    }
    T out[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) out[s] = ax[s];
    store_row<T, NS>(gx, node, u, c, out);
    store_row<T, NB>(gw_node, node, u, c, aw);
  }
}

// dx and dg from the cotangents dgx (d) and dgw (dw). Per run with
// sym = d[i] x[j] x[k] + d[j] x[i] x[k] + d[k] x[i] x[j]: s = Σ C g[o] W[w],
// sd = Σ C g[o] dW[w]; then dx[i] += s (d[j] x[k] + d[k] x[j]) + sd x[j] x[k]
// and the two cyclic slots; per term dg[o] += C (W[w] sym + dW[w] mono).
template <typename T, int LIN, int LOUT, int DEG>
__global__ void __launch_bounds__(128, 2) second_node_kernel(
    const T* __restrict__ x, const T* __restrict__ w, const int32_t* __restrict__ index,
    const T* __restrict__ g, const T* __restrict__ dgx, const T* __restrict__ dgw,
    T* __restrict__ dx, T* __restrict__ dg, int64_t z, int64_t u,
    const __grid_constant__ Coefficients<T, Table<LIN, LOUT, DEG>::NT> coef) {
  using Tb = Table<LIN, LOUT, DEG>;
  constexpr int NS = Tb::NS, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  const int64_t node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < z;
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T adx[NS + 1], adg[NO];
    zero(adx);
    zero(adg);
    if (valid) {
      T xr[NS + 1], wr[NB], gr[NO], dr[NS + 1], dwr[NB];
      load_rows<T, NS, NB>(x, w, node, element, u, c, xr, wr);
      load_rows<T, NS, NB>(dgx, dgw, node, element, u, c, dr, dwr);
      dr[NS] = T(0);
      load_row<T, NO>(g, node, u, c, gr);
      T mono = T(0), sym = T(0), s = T(0), sd = T(0);
      for_each_term<NT>([&]<int t>() {
        constexpr Term m = Tb::terms[t];
        if constexpr (starts_run<LIN, LOUT, DEG, t>()) {
          mono = xr[m.i] * xr[m.j] * xr[m.k];
          sym = dr[m.i] * xr[m.j] * xr[m.k] + dr[m.j] * xr[m.i] * xr[m.k] + dr[m.k] * xr[m.i] * xr[m.j];
          s = sd = T(0);
        }
        const T cc = coef.c[t], pp = cc * gr[m.o];
        adg[m.o] += cc * (wr[m.w] * sym + dwr[m.w] * mono);
        s += pp * wr[m.w];
        sd += pp * dwr[m.w];
        if constexpr (ends_run<LIN, LOUT, DEG, t>()) {
          adx[m.i] += s * (dr[m.j] * xr[m.k] + dr[m.k] * xr[m.j]) + sd * (xr[m.j] * xr[m.k]);
          adx[m.j] += s * (dr[m.i] * xr[m.k] + dr[m.k] * xr[m.i]) + sd * (xr[m.i] * xr[m.k]);
          adx[m.k] += s * (dr[m.i] * xr[m.j] + dr[m.j] * xr[m.i]) + sd * (xr[m.i] * xr[m.j]);
        }
      });
    }
    T out[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) out[s] = adx[s];
    store_row<T, NS>(dx, node, u, c, out);
    store_row<T, NO>(dg, node, u, c, adg);
  }
}

// The per-node dw partial: dw[w] += C g[o] sym.
template <typename T, int LIN, int LOUT, int DEG>
__global__ void __launch_bounds__(128, 2) second_weight_kernel(
    const T* __restrict__ x, const int32_t* __restrict__ index, const T* __restrict__ g,
    const T* __restrict__ dgx, T* __restrict__ dw_node, int64_t z, int64_t u,
    const __grid_constant__ Coefficients<T, Table<LIN, LOUT, DEG>::NT> coef) {
  using Tb = Table<LIN, LOUT, DEG>;
  constexpr int NS = Tb::NS, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  const int64_t node = blockIdx.x;
  const int element = index[node];
  const bool valid = element >= 0 && element < z;
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T aw[NB];
    zero(aw);
    if (valid) {
      T xr[NS + 1], dr[NS + 1], gr[NO];
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        xr[s] = x[(node * NS + s) * u + c];
        dr[s] = dgx[(node * NS + s) * u + c];
      }
      xr[NS] = T(1);
      dr[NS] = T(0);
      load_row<T, NO>(g, node, u, c, gr);
      T sym = T(0);
      for_each_term<NT>([&]<int t>() {
        constexpr Term m = Tb::terms[t];
        if constexpr (starts_run<LIN, LOUT, DEG, t>())
          sym = dr[m.i] * xr[m.j] * xr[m.k] + dr[m.j] * xr[m.i] * xr[m.k] + dr[m.k] * xr[m.i] * xr[m.j];
        aw[m.w] += (coef.c[t] * gr[m.o]) * sym;
      });
    }
    store_row<T, NB>(dw_node, node, u, c, aw);
  }
}

}  // namespace molnex::equivariance::sc
