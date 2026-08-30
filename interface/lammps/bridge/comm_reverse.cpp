// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `comm_reverse.h` for the design notes and the preserved-defect record.

#include "bridge/comm_reverse.h"

namespace molnex::lammps_bridge {

int pack_reverse_forces(const LammpsView& view, int n, int first, double* buf) {
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; ++i) {
    buf[m++] = view.f[i][0];
    buf[m++] = view.f[i][1];
    buf[m++] = view.f[i][2];
  }
  return m;
}

void unpack_reverse_forces(const LammpsView& view, int n, const int* list,
                           const double* buf) {
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const int j = list[i];
    view.f[j][0] += buf[m++];
    view.f[j][1] += buf[m++];
    view.f[j][2] += buf[m++];
  }
}

}  // namespace molnex::lammps_bridge
