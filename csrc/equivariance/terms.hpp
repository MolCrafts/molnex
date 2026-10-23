// Compile-time term tables of the natural-parity channelwise product; pure C++
// so a host build can check the enumeration against the Python descriptor.
#pragma once

#include <array>
#include <cstdint>

namespace molnex::equivariance {

struct Term {
  int16_t i;  // (l1, m1) row of the node operand, ``l1² + l1 + m1``
  int16_t j;  // (l2, m2) component of the angular operand
  int16_t p;  // path: weight segment
  int16_t k;  // (l3, m3) row of the output operand
};

constexpr int cabs(int x) { return x < 0 ? -x : x; }

template <int L1, int L2>
struct Config {
  static constexpr int L3 = L2;
  static constexpr int NA = (L1 + 1) * (L1 + 1);
  static constexpr int NB = (L2 + 1) * (L2 + 1);

  // Paths: enumerated as cuEquivariance does — ``l1`` outer, ``l2``, then ``l3``
  // ascending within the triangle, natural parity (``l1+l2+l3`` even) — and
  // then given the *segment* index of its sorted output irreps (a stable sort
  // by ``l3``), which the weight segments share. ``f(l1, l2, l3, seg, row)``
  // with ``row`` the first output row of the segment.
  template <class F>
  static constexpr void each_path(F&& f) {
    int l1s[64] = {}, l2s[64] = {}, l3s[64] = {};
    int n = 0;
    for (int l1 = 0; l1 <= L1; ++l1)
      for (int l2 = 0; l2 <= L2; ++l2)
        for (int l3 = cabs(l1 - l2); l3 <= (l1 + l2 < L3 ? l1 + l2 : L3); ++l3)
          if ((l1 + l2 + l3) % 2 == 0) {
            l1s[n] = l1;
            l2s[n] = l2;
            l3s[n] = l3;
            ++n;
          }
    for (int q = 0; q < n; ++q) {
      int seg = 0, row = 0;
      for (int r = 0; r < n; ++r)
        if (l3s[r] < l3s[q] || (l3s[r] == l3s[q] && r < q)) {
          ++seg;
          row += 2 * l3s[r] + 1;
        }
      f(l1s[q], l2s[q], l3s[q], seg, row);
    }
  }
  static constexpr int paths() {
    int n = 0;
    each_path([&](int, int, int, int, int) { ++n; });
    return n;
  }
  static constexpr int rows() {
    int n = 0;
    each_path([&](int, int, int l3, int, int) { n += 2 * l3 + 1; });
    return n;
  }
  static constexpr int NP = paths();
  static constexpr int NK = rows();

  template <class F>
  static constexpr void each_term(F&& f) {
    each_path([&](int l1, int l2, int l3, int p, int row) {
      for (int i = 0; i <= 2 * l1; ++i)
        for (int j = 0; j <= 2 * l2; ++j) {
          const int m1 = i - l1, m2 = j - l2;
          int m3[2] = {0, 0};
          int n = 0;
          if (m1 == 0) {
            m3[n++] = m2;
          } else if (m2 == 0) {
            m3[n++] = m1;
          } else {
            const int s = cabs(m1) + cabs(m2), d = cabs(cabs(m1) - cabs(m2));
            if ((m1 > 0) == (m2 > 0)) {
              m3[n++] = s;
              m3[n++] = d;
            } else {
              m3[n++] = -s;
              if (d != 0) m3[n++] = -d;
            }
          }
          for (int q = 0; q < n; ++q)
            if (cabs(m3[q]) <= l3)
              f(Term{int16_t(l1 * l1 + i), int16_t(l2 * l2 + j), int16_t(p),
                     int16_t(row + m3[q] + l3)});
        }
    });
  }
  static constexpr int terms() {
    int n = 0;
    each_term([&](Term) { ++n; });
    return n;
  }
  static constexpr int NT = terms();
  static constexpr std::array<Term, NT> table() {
    std::array<Term, NT> t{};
    int n = 0;
    each_term([&](Term x) { t[n++] = x; });
    return t;
  }
};

template <int L1, int L2>
inline constexpr std::array<Term, Config<L1, L2>::NT> kTable = Config<L1, L2>::table();

}  // namespace molnex::equivariance
