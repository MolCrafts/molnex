// Allegro contracter with a compile-time term table (``contracter_tables.hpp``):
// every operand and accumulator in registers, the coefficient values a kernel
// parameter, the term loop unrolled.
//
// One CTA of a channel tile walks a chunk of edges. A thread holds the shared
// weight row ``W[NB]`` (and, in the second order, its cotangent) across the
// chunk, and per edge the pair row ``x1[NS1]``, the gathered atom row
// ``x2[NS2]`` and the cotangents it needs. The table is sorted by
// (i, j, w, o): the terms of one product ``x1[i] x2[j]`` are a run, and the
// run's product and sums are computed once, so a term costs two to five FMAs
// on registers with no common subexpression for the compiler to hoist. The
// weight adjoint accumulates in registers across the chunk and is written
// once per CTA, then the CTAs are added in order; the atom adjoint is written
// per edge and summed into its atom in a fixed order. An edge whose atom
// index is out of range contributes nothing. Kernels: forward; first order in
// two passes (gx1 + gx2 per edge, then gw per CTA); second order in two
// passes (dx1, dx2, dg; then dw).
#pragma once

#include <c10/macros/Macros.h>

#include "channelwise.cuh"  // Coefficients, for_each_term
#include "contracter_tables.hpp"

namespace molnex::equivariance::ct {

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

__device__ __forceinline__ void chunk_of(int64_t edges, int64_t chunk, int64_t& first, int64_t& last) {
  first = min(edges, int64_t(blockIdx.x) * chunk);
  last = min(edges, first + chunk);
}

template <int ID, int t>
constexpr bool starts_run() {
  using Tb = Table<ID>;
  if constexpr (t == 0) return true;
  else return Tb::terms[t - 1].i != Tb::terms[t].i || Tb::terms[t - 1].j != Tb::terms[t].j;
}
template <int ID, int t>
constexpr bool ends_run() {
  using Tb = Table<ID>;
  if constexpr (t + 1 == Tb::NT) return true;
  else return Tb::terms[t + 1].i != Tb::terms[t].i || Tb::terms[t + 1].j != Tb::terms[t].j;
}

template <typename T, int ID>
__global__ void __launch_bounds__(128, 2) forward_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ w,
    const int32_t* __restrict__ index, T* __restrict__ y, int64_t edges, int64_t atoms, int64_t u,
    int64_t chunk, const __grid_constant__ Coefficients<T, Table<ID>::NT> coef) {
  using Tb = Table<ID>;
  constexpr int NS1 = Tb::NS1, NS2 = Tb::NS2, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  int64_t first, last;
  chunk_of(edges, chunk, first, last);
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T wr[NB];
    load_row<T, NB>(w, 0, u, c, wr);
    for (int64_t e = first; e < last; ++e) {
      const int atom = index[e];
      T acc[NO];
      zero(acc);
      if (atom >= 0 && atom < atoms) {
        T a[NS1], b[NS2];
        load_row<T, NS1>(x1, e, u, c, a);
        load_row<T, NS2>(x2, atom, u, c, b);
        T ab = T(0);
        for_each_term<NT>([&]<int t>() {
          constexpr Term m = Tb::terms[t];
          if constexpr (starts_run<ID, t>()) ab = a[m.i] * b[m.j];
          acc[m.o] += (coef.c[t] * wr[m.w]) * ab;
        });
      }
      store_row<T, NO>(y, e, u, c, acc);
    }
  }
}

// gx1 (per edge) and gx2 per edge (summed into the atoms afterwards). Per run
// r = Σ C g[o] W[w]; gx1[i] += x2[j] r, gx2[j] += x1[i] r.
template <typename T, int ID>
__global__ void __launch_bounds__(128, 2) first_rows_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ w,
    const int32_t* __restrict__ index, const T* __restrict__ g, T* __restrict__ gx1,
    T* __restrict__ gx2_edge, int64_t edges, int64_t atoms, int64_t u, int64_t chunk,
    const __grid_constant__ Coefficients<T, Table<ID>::NT> coef) {
  using Tb = Table<ID>;
  constexpr int NS1 = Tb::NS1, NS2 = Tb::NS2, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  int64_t first, last;
  chunk_of(edges, chunk, first, last);
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T wr[NB];
    load_row<T, NB>(w, 0, u, c, wr);
    for (int64_t e = first; e < last; ++e) {
      const int atom = index[e];
      T a1[NS1], a2[NS2];
      zero(a1);
      zero(a2);
      if (atom >= 0 && atom < atoms) {
        T a[NS1], b[NS2], gr[NO];
        load_row<T, NS1>(x1, e, u, c, a);
        load_row<T, NS2>(x2, atom, u, c, b);
        load_row<T, NO>(g, e, u, c, gr);
        T r = T(0);
        for_each_term<NT>([&]<int t>() {
          constexpr Term m = Tb::terms[t];
          if constexpr (starts_run<ID, t>()) r = T(0);
          r += (coef.c[t] * gr[m.o]) * wr[m.w];
          if constexpr (ends_run<ID, t>()) {
            a1[m.i] += b[m.j] * r;
            a2[m.j] += a[m.i] * r;
          }
        });
      }
      store_row<T, NS1>(gx1, e, u, c, a1);
      store_row<T, NS2>(gx2_edge, e, u, c, a2);
    }
  }
}

// gw (kSecond == 0: Σ C g[o] x1[i] x2[j]) or dw (kSecond == 1:
// Σ C g[o] (d1[i] x2[j] + x1[i] d2[j])), one partial per CTA.
template <typename T, int ID, bool kSecond>
__global__ void __launch_bounds__(128, 2) weight_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const int32_t* __restrict__ index,
    const T* __restrict__ g, const T* __restrict__ dgx1, const T* __restrict__ dgx2,
    T* __restrict__ partial, int64_t edges, int64_t atoms, int64_t u, int64_t chunk,
    const __grid_constant__ Coefficients<T, Table<ID>::NT> coef) {
  using Tb = Table<ID>;
  constexpr int NS1 = Tb::NS1, NS2 = Tb::NS2, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  int64_t first, last;
  chunk_of(edges, chunk, first, last);
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T aw[NB];
    zero(aw);
    for (int64_t e = first; e < last; ++e) {
      const int atom = index[e];
      if (atom < 0 || atom >= atoms) continue;
      T a[NS1], b[NS2], gr[NO];
      load_row<T, NS1>(x1, e, u, c, a);
      load_row<T, NS2>(x2, atom, u, c, b);
      load_row<T, NO>(g, e, u, c, gr);
      if constexpr (!kSecond) {
        T ab = T(0);
        for_each_term<NT>([&]<int t>() {
          constexpr Term m = Tb::terms[t];
          if constexpr (starts_run<ID, t>()) ab = a[m.i] * b[m.j];
          aw[m.w] += (coef.c[t] * gr[m.o]) * ab;
        });
      } else {
        T d1[NS1], d2[NS2];
        load_row<T, NS1>(dgx1, e, u, c, d1);
        load_row<T, NS2>(dgx2, atom, u, c, d2);
        T cross = T(0);
        for_each_term<NT>([&]<int t>() {
          constexpr Term m = Tb::terms[t];
          if constexpr (starts_run<ID, t>()) cross = d1[m.i] * b[m.j] + a[m.i] * d2[m.j];
          aw[m.w] += (coef.c[t] * gr[m.o]) * cross;
        });
      }
    }
    store_row<T, NB>(partial, blockIdx.x, u, c, aw);
  }
}

// dx1, dx2 (per edge) and dg from the cotangents dgx1, dgx2 (gathered), dgw.
// Per run: s = Σ C g[o] W[w], sd = Σ C g[o] dW[w]; dx1[i] += s d2[j] + sd x2[j],
// dx2[j] += s d1[i] + sd x1[i]; per term dg[o] += C (W[w] cross + dW[w] ab).
template <typename T, int ID>
__global__ void __launch_bounds__(128, 2) second_rows_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ w,
    const int32_t* __restrict__ index, const T* __restrict__ g, const T* __restrict__ dgx1,
    const T* __restrict__ dgx2, const T* __restrict__ dgw, T* __restrict__ dx1,
    T* __restrict__ dx2_edge, T* __restrict__ dg, int64_t edges, int64_t atoms, int64_t u,
    int64_t chunk, const __grid_constant__ Coefficients<T, Table<ID>::NT> coef) {
  using Tb = Table<ID>;
  constexpr int NS1 = Tb::NS1, NS2 = Tb::NS2, NB = Tb::NB, NO = Tb::NO, NT = Tb::NT;
  int64_t first, last;
  chunk_of(edges, chunk, first, last);
  for (int64_t c0 = 0; c0 < u; c0 += blockDim.x) {
    const int64_t c = c0 + threadIdx.x;
    if (c >= u) break;
    T wr[NB], dwr[NB];
    load_row<T, NB>(w, 0, u, c, wr);
    load_row<T, NB>(dgw, 0, u, c, dwr);
    for (int64_t e = first; e < last; ++e) {
      const int atom = index[e];
      T o1[NS1], o2[NS2], og[NO];
      zero(o1);
      zero(o2);
      zero(og);
      if (atom >= 0 && atom < atoms) {
        T a[NS1], b[NS2], gr[NO], d1[NS1], d2[NS2];
        load_row<T, NS1>(x1, e, u, c, a);
        load_row<T, NS2>(x2, atom, u, c, b);
        load_row<T, NO>(g, e, u, c, gr);
        load_row<T, NS1>(dgx1, e, u, c, d1);
        load_row<T, NS2>(dgx2, atom, u, c, d2);
        T ab = T(0), cross = T(0), s = T(0), sd = T(0);
        for_each_term<NT>([&]<int t>() {
          constexpr Term m = Tb::terms[t];
          if constexpr (starts_run<ID, t>()) {
            ab = a[m.i] * b[m.j];
            cross = d1[m.i] * b[m.j] + a[m.i] * d2[m.j];
            s = sd = T(0);
          }
          const T cf = coef.c[t], cg = cf * gr[m.o];
          og[m.o] += cf * (wr[m.w] * cross + dwr[m.w] * ab);
          s += cg * wr[m.w];
          sd += cg * dwr[m.w];
          if constexpr (ends_run<ID, t>()) {
            o1[m.i] += s * d2[m.j] + sd * b[m.j];
            o2[m.j] += s * d1[m.i] + sd * a[m.i];
          }
        });
      }
      store_row<T, NS1>(dx1, e, u, c, o1);
      store_row<T, NS2>(dx2_edge, e, u, c, o2);
      store_row<T, NO>(dg, e, u, c, og);
    }
  }
}

}  // namespace molnex::equivariance::ct
