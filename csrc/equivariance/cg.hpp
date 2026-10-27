// Compile-time Clebsch–Gordan classes: every coupling (l1, l2, l3) with
// l ≤ 3 that the triangle rule allows, each with the positions (m1, m2, m3)
// at which the real spherical-harmonic basis can carry a non-zero coefficient.
//
// A product of components m1 and m2 is non-zero only at
// m3 ∈ {±(|m1|+|m2|), ±(||m1|−|m2||)} and keeps one type: for an even
// l1+l2+l3 (the natural-parity couplings) factors of the same type (both
// cosine, m ≥ 0, or both sine) give cosine type and mixed factors sine type;
// for an odd l1+l2+l3 (the antisymmetric couplings, e.g. 1 × 1 → 1) the types
// swap. A zonal factor (m = 0) passes the other's m through, negated for an
// odd sum. Verified against cuEquivariance's O3 tables for every l ≤ 3:
// 34 classes, 611 non-zero coefficients inside 652 candidates
// (molrep.equivariance.descriptor.candidate_m3 is the same rule).
//
// Coefficient *values* travel as a kernel parameter — one flat table over the
// classes in lexicographic (l1, l2, l3) order, filled by the Python side from
// the descriptor, zero where a candidate position carries no coefficient —
// so a class contraction is a fully unrolled sequence of FMAs with
// constant-bank coefficients on register-resident blocks.
#pragma once

#include <c10/macros/Macros.h>

#include <array>
#include <utility>

namespace molnex::equivariance::cg {

constexpr int kLmax = 3;
constexpr int kBlock = 2 * kLmax + 1;  // components of one block, 7

constexpr int cabs(int x) { return x < 0 ? -x : x; }

// Candidate m3 of a product (m1, m2); returns the count, at most 2.
constexpr int candidates(int m1, int m2, bool odd, int (&m3)[2]) {
  if (m1 == 0) { m3[0] = odd ? -m2 : m2; return 1; }
  if (m2 == 0) { m3[0] = odd ? -m1 : m1; return 1; }
  const int s = cabs(m1) + cabs(m2), d = cabs(cabs(m1) - cabs(m2));
  const bool same = (m1 > 0) == (m2 > 0);
  if (same != odd) { m3[0] = s; m3[1] = d; return 2; }
  m3[0] = -s;
  if (d != 0) { m3[1] = -d; return 2; }
  return 1;
}

constexpr bool valid(int l1, int l2, int l3) {
  return l1 >= 0 && l2 >= 0 && l3 >= 0 && l1 <= kLmax && l2 <= kLmax && l3 <= kLmax &&
         l3 >= cabs(l1 - l2) && l3 <= l1 + l2;
}

template <class F>
constexpr void each_position(int l1, int l2, int l3, F&& f) {
  const bool odd = (l1 + l2 + l3) % 2 == 1;
  for (int m1 = -l1; m1 <= l1; ++m1)
    for (int m2 = -l2; m2 <= l2; ++m2) {
      int m3[2] = {0, 0};
      const int n = candidates(m1, m2, odd, m3);
      for (int q = 0; q < n; ++q)
        if (cabs(m3[q]) <= l3) f(m1 + l1, m2 + l2, m3[q] + l3);
    }
}

constexpr int count(int l1, int l2, int l3) {
  int n = 0;
  each_position(l1, l2, l3, [&](int, int, int) { ++n; });
  return n;
}

// Offset of a class in the flat coefficient table: classes in lexicographic
// (l1, l2, l3) order, invalid triples absent.
constexpr int offset(int l1, int l2, int l3) {
  int off = 0;
  for (int a = 0; a <= kLmax; ++a)
    for (int b = 0; b <= kLmax; ++b)
      for (int c = 0; c <= kLmax; ++c) {
        if (a == l1 && b == l2 && c == l3) return off;
        if (valid(a, b, c)) off += count(a, b, c);
      }
  return off;
}

constexpr int kTotal = offset(kLmax, kLmax, kLmax) + count(kLmax, kLmax, kLmax);
static_assert(kTotal == 652, "the candidate table of l <= 3 has 652 positions");

// Class id a path carries: (l1, l2, l3) packed in base 4.
constexpr int id(int l1, int l2, int l3) { return (l1 * (kLmax + 1) + l2) * (kLmax + 1) + l3; }

struct Position {
  int m1, m2, m3;
};

template <int L1, int L2, int L3>
struct Class {
  static_assert(valid(L1, L2, L3), "not a Clebsch-Gordan class of l <= 3");
  static constexpr int N1 = 2 * L1 + 1, N2 = 2 * L2 + 1, N3 = 2 * L3 + 1;
  static constexpr int NT = count(L1, L2, L3);
  static constexpr int OFFSET = offset(L1, L2, L3);
  static constexpr std::array<Position, NT> table() {
    std::array<Position, NT> t{};
    int n = 0;
    each_position(L1, L2, L3, [&](int a, int b, int c) { t[n++] = Position{a, b, c}; });
    return t;
  }
};

template <int L1, int L2, int L3>
inline constexpr std::array<Position, Class<L1, L2, L3>::NT> kTable = Class<L1, L2, L3>::table();

// The coefficient table as a kernel parameter (constant bank).
template <typename T>
struct Values {
  T v[kTotal];
};

// ``f.template operator()<m1, m2, m3>(coefficient)`` for every position of the class.
template <int L1, int L2, int L3, typename T, class F>
__device__ __forceinline__ void each(const Values<T>& values, F&& f) {
  [&]<int... Ts>(std::integer_sequence<int, Ts...>) {
    (f.template operator()<kTable<L1, L2, L3>[Ts].m1, kTable<L1, L2, L3>[Ts].m2,
                           kTable<L1, L2, L3>[Ts].m3>(values.v[Class<L1, L2, L3>::OFFSET + Ts]),
     ...);
  }(std::make_integer_sequence<int, Class<L1, L2, L3>::NT>{});
}

// Every valid class, for a dispatch switch.
#define MOLNEX_CG_CLASSES(X)                                                                   \
  X(0, 0, 0) X(0, 1, 1) X(0, 2, 2) X(0, 3, 3) X(1, 0, 1) X(1, 1, 0) X(1, 1, 1) X(1, 1, 2)      \
  X(1, 2, 1) X(1, 2, 2) X(1, 2, 3) X(1, 3, 2) X(1, 3, 3) X(2, 0, 2) X(2, 1, 1) X(2, 1, 2)      \
  X(2, 1, 3) X(2, 2, 0) X(2, 2, 1) X(2, 2, 2) X(2, 2, 3) X(2, 3, 1) X(2, 3, 2) X(2, 3, 3)      \
  X(3, 0, 3) X(3, 1, 2) X(3, 1, 3) X(3, 2, 1) X(3, 2, 2) X(3, 2, 3) X(3, 3, 0) X(3, 3, 1)      \
  X(3, 3, 2) X(3, 3, 3)

}  // namespace molnex::equivariance::cg
