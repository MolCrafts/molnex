// C++ acceptance tests for the atom-side LAMMPS bridge components
// (spec lammps-iface-02-view): SpeciesMap, stage_nodes and ForceScatter,
// driven over a synthetic LammpsView.
//
// House style follows test_model_runner.cpp: no test framework, one function
// per case, a function-local static map dispatcher, and the [pass]/[fail]/
// [skip] stderr protocol. Like test_meta_schema.cpp these cases need NO
// fixtures — no exported model, no Python generation — and, crucially, no
// LAMMPS build: the whole point of LammpsView is that the bridge consumes
// borrowed row-pointer arrays, which a test can hand it out of its own
// std::vector storage. CMake therefore registers them without
// FIXTURES_REQUIRED.
//
// API contract under test. The components arrive with the next task and MUST
// satisfy exactly this (namespace molnex::lammps_bridge throughout; the
// include paths come from the molnex_lammps_bridge target's PUBLIC include
// directory, so `bridge/` is the prefix):
//
//   #include "bridge/lammps_view.h"
//
//   struct LammpsView {          // aggregate: no user-provided constructor
//     const double* const* x;    // borrowed, (nall, 3) angstrom
//     double* const* f;          // borrowed and writable, (nall, 3)
//     const int* type;           // borrowed, (nall,), 1-based LAMMPS types
//     int nlocal;
//     int nghost;
//     int nall() const;          // nlocal + nghost
//   };
//
//   #include "bridge/species_map.h"
//
//   class SpeciesMap {
//    public:
//     // z_by_type is 0-BASED and dense, length == ntypes, NO placeholder slot.
//     // Construction validates every entry: > 0 and present in meta_species,
//     // else std::runtime_error naming the type, its Z and every valid species.
//     SpeciesMap(std::vector<int> z_by_type, const std::vector<int>& meta_species);
//     int z_of(int lammps_type) const;   // 1-based in; throws outside [1, ntypes]
//     int ntypes() const;                // == z_by_type.size()
//   };
//
//   #include "bridge/node_staging.h"
//
//   struct NodeTensors { at::Tensor Z; at::Tensor pos; };
//   NodeTensors stage_nodes(const LammpsView&, const SpeciesMap&, int n_graph);
//
//   #include "bridge/force_scatter.h"
//
//   class ForceScatter {
//    public:
//     explicit ForceScatter(double force_conv);
//     void scatter(const LammpsView&, const at::Tensor& forces, int n_graph) const;
//   };
//
// The three invariants these cases exist to lock:
//
//   1. SpeciesMap validates at construction. The code being migrated
//      (pair_molnex.cpp) does not check `pair_coeff` Z values at all, so an
//      element the model was never exported for produces a confident, wrong
//      energy and no diagnostic. That silent path is what
//      bridge_species_map_rejects_unknown_z regresses.
//   2. ForceScatter ACCUMULATES (`+=`), never overwrites. Asserted from a
//      non-zero starting f, because a zero start cannot tell the two apart —
//      and every other pair style's contribution, plus virial_fdotr_compute,
//      depends on the accumulate semantics.
//   3. n_graph is validated before either component indexes x or f. Those
//      arrays belong to LAMMPS; an unchecked n_graph > nall() is a silent
//      out-of-bounds write into the middle of another object's memory, so the
//      two n_graph cases assert a throw, not a tolerated overrun.

#include <array>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/force_scatter.h"
#include "bridge/lammps_view.h"
#include "bridge/node_staging.h"
#include "bridge/species_map.h"
#include "expect.h"

using molnex::lammps_bridge::ForceScatter;
using molnex::lammps_bridge::LammpsView;
using molnex::lammps_bridge::NodeTensors;
using molnex::lammps_bridge::SpeciesMap;
using molnex::lammps_bridge::stage_nodes;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- the synthetic system --------------------------------------------------

// Three local atoms plus one ghost. The ghost exists so every case can assert
// that a component honouring n_graph == nlocal leaves it alone: that is the
// local-only semantics of a supports_pbc=false export.
constexpr int kNlocal = 3;
constexpr int kNghost = 1;
constexpr int kNall = kNlocal + kNghost;

// Atomic numbers the model was exported for: H, C, O.
const std::vector<int> kMetaSpecies = {1, 6, 8};

// The type-to-Z table, 0-based and dense: LAMMPS type 1 is O, type 2 is H,
// type 3 is C. Deliberately NOT the identity and not sorted, so a table read
// with the wrong base (or not read at all) cannot pass by coincidence.
const std::vector<int> kZByType = {8, 1, 6};

// Per-atom LAMMPS types, 1-based. Atom 3 is the ghost.
const std::vector<int> kType = {1, 2, 2, 3};

// Positions in angstrom, all distinct, the ghost's deliberately far away.
constexpr double kCoord[kNall][3] = {
    {0.0, 1.5, -2.25},
    {3.125, -0.5, 0.75},
    {-1.0, 2.0, 4.5},
    {9.0, 9.0, 9.0},
};

// Forces already standing in f when the pair style runs: a non-zero start, as
// LAMMPS would have after an earlier pair style contributed. Overwrite-instead-
// of-accumulate is invisible against a zero start, which is why this is not it.
constexpr double kInitialForce[kNall][3] = {
    {1.5, -2.25, 0.75},
    {-0.5, 3.125, -1.0},
    {0.0, 0.5, -0.125},
    {7.0, 7.0, 7.0},
};

// What the model returns for the three local atoms, in model units.
constexpr double kModelForce[kNlocal][3] = {
    {0.5, -1.25, 2.0},
    {1.25, 0.0, -0.75},
    {-2.5, 0.125, 1.0},
};

// Model energy unit expressed in the LAMMPS unit style (dimensionless), as
// LammpsContract::units_for would have handed it to settings(). eV -> kcal/mol.
constexpr double kForceConv = 23.060549;

// f_in + kForceConv * f_model, computed outside this file and pasted as
// literals. Recomputing the expectation from the same expression the component
// evaluates would assert nothing.
constexpr double kScatteredForce[kNlocal][3] = {
    {13.0302745, -31.075686250000004, 46.871098},
    {28.325686250000004, 3.125, -18.29541175},
    {-57.65137250000001, 3.382568625, 22.935549},
};

/// The storage a synthetic LammpsView borrows, standing in for atom->x / ->f.
///
/// LAMMPS owns its atom arrays as `double**`: an array of row pointers into
/// separately managed rows. This fixture reproduces exactly that shape out of
/// its own vectors, which is what makes the bridge drivable with no LAMMPS
/// build — the whole reason LammpsView exists.
///
/// Non-copyable: the row-pointer vectors point into this object's own rows, so
/// a copy would hand out pointers into the source object.
class SyntheticAtoms {
 public:
  SyntheticAtoms() : type_(kType) {
    for (int i = 0; i < kNall; ++i) {
      coords_.push_back({kCoord[i][0], kCoord[i][1], kCoord[i][2]});
      forces_.push_back(
          {kInitialForce[i][0], kInitialForce[i][1], kInitialForce[i][2]});
    }
    for (int i = 0; i < kNall; ++i) {
      x_rows_.push_back(coords_[i].data());
      f_rows_.push_back(forces_[i].data());
    }
  }

  SyntheticAtoms(const SyntheticAtoms&) = delete;
  SyntheticAtoms& operator=(const SyntheticAtoms&) = delete;

  /// A view borrowing this fixture's arrays; valid while the fixture lives.
  LammpsView view() {
    return LammpsView{x_rows_.data(), f_rows_.data(), type_.data(), kNlocal,
                      kNghost};
  }

  /// Current value of f[i][k] — what the component under test wrote.
  double force(int i, int k) const { return forces_[i][k]; }

 private:
  std::vector<std::array<double, 3>> coords_;
  std::vector<std::array<double, 3>> forces_;
  std::vector<int> type_;
  std::vector<double*> x_rows_;
  std::vector<double*> f_rows_;
};

/// The (n, 3) fp64 CPU force tensor a model returns, filled from `rows`.
///
/// fp64 and CPU because that is what the call site hands over: pair_molnex
/// brings the model output back with `.to(at::kDouble).to(at::kCPU)` before any
/// scatter, and this spec moves no dtype decision.
at::Tensor model_forces(int n) {
  at::Tensor forces = at::empty({n, 3}, at::TensorOptions().dtype(at::kDouble));
  auto rows = forces.accessor<double, 2>();
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      rows[i][k] = kModelForce[i % kNlocal][k];
    }
  }
  return forces;
}

/// Require every f row — ghost included — to still hold its starting value.
void expect_forces_untouched(const SyntheticAtoms& atoms, const std::string& tag) {
  for (int i = 0; i < kNall; ++i) {
    for (int k = 0; k < 3; ++k) {
      expect_near(atoms.force(i, k), kInitialForce[i][k], 0.0,
                  tag + ": f[" + std::to_string(i) + "][" + std::to_string(k) +
                      "] must be untouched");
    }
  }
}

// ---- happy path ------------------------------------------------------------

void test_bridge_species_map_maps_types() {
  const SpeciesMap map(kZByType, kMetaSpecies);

  // ntypes comes from the table's own length: there is no second ntypes
  // parameter to disagree with it, and no placeholder slot inflating it.
  expect_int(map.ntypes(), static_cast<long long>(kZByType.size()),
             "ntypes() must be the table length");

  // 1-based LAMMPS type in, 0-based table index out.
  for (int t = 1; t <= static_cast<int>(kZByType.size()); ++t) {
    expect_int(map.z_of(t), kZByType[static_cast<std::size_t>(t) - 1],
               "z_of(" + std::to_string(t) + ")");
  }
}

void test_bridge_stage_nodes_shapes_and_values() {
  SyntheticAtoms atoms;
  const SpeciesMap map(kZByType, kMetaSpecies);

  // n_graph == nlocal < nall: the local-only staging of a supports_pbc=false
  // export. The ghost must contribute nothing.
  const NodeTensors nodes = stage_nodes(atoms.view(), map, kNlocal);

  expect_true(nodes.Z.scalar_type() == at::kLong, "Z must be int64");
  expect_int(nodes.Z.dim(), 1, "Z must be 1-D");
  expect_int(nodes.Z.size(0), kNlocal, "Z must be (n_graph,)");
  expect_true(nodes.Z.device().is_cpu(),
              "Z must stay on the CPU: the device transfer belongs to the call "
              "site, not to staging");

  expect_true(nodes.pos.scalar_type() == at::kDouble,
              "pos must be float64: the model-dtype cast stays at the call site");
  expect_int(nodes.pos.dim(), 2, "pos must be 2-D");
  expect_int(nodes.pos.size(0), kNlocal, "pos must be (n_graph, 3)");
  expect_int(nodes.pos.size(1), 3, "pos must be (n_graph, 3)");
  expect_true(nodes.pos.device().is_cpu(), "pos must stay on the CPU");

  // Values are copied, not computed, so they must agree exactly. Asserting the
  // first kNlocal rows against the view's own coordinates is what pins "the
  // first n_graph atoms": row 2 holding the ghost's 9.0s would fail here.
  const auto Za = nodes.Z.accessor<int64_t, 1>();
  const auto pa = nodes.pos.accessor<double, 2>();
  for (int i = 0; i < kNlocal; ++i) {
    expect_int(Za[i], map.z_of(kType[static_cast<std::size_t>(i)]),
               "Z[" + std::to_string(i) + "]");
    for (int k = 0; k < 3; ++k) {
      expect_near(pa[i][k], kCoord[i][k], 0.0,
                  "pos[" + std::to_string(i) + "][" + std::to_string(k) + "]");
    }
  }
}

void test_bridge_force_scatter_accumulates() {
  SyntheticAtoms atoms;
  const ForceScatter scatter(kForceConv);

  scatter.scatter(atoms.view(), model_forces(kNlocal), kNlocal);

  // f_out == f_in + force_conv * f_model, against literals computed elsewhere.
  // An overwrite would land on force_conv * f_model and fail on every row.
  for (int i = 0; i < kNlocal; ++i) {
    for (int k = 0; k < 3; ++k) {
      expect_near(atoms.force(i, k), kScatteredForce[i][k], 1e-12,
                  "f[" + std::to_string(i) + "][" + std::to_string(k) + "]");
    }
  }

  // The ghost sits past n_graph and keeps its starting force untouched.
  for (int k = 0; k < 3; ++k) {
    expect_near(atoms.force(kNall - 1, k), kInitialForce[kNall - 1][k], 0.0,
                "ghost f[" + std::to_string(k) + "] must be untouched");
  }
}

// ---- edge cases ------------------------------------------------------------

void test_bridge_species_map_rejects_unknown_z() {
  // Type 3 is mapped to Z 7 (nitrogen), which the model was never exported for.
  // The migrated code accepts this silently and feeds the network an element it
  // has no embedding for; the resulting energy is wrong and nothing says so.
  const std::vector<int> unknown = {1, 6, 7};
  const std::string msg = expect_throws(
      [&] {
        const SpeciesMap map(unknown, kMetaSpecies);
        (void)map;
      },
      "unknown Z");
  expect_contains(msg, "3", "unknown Z must name the offending LAMMPS type");
  expect_contains(msg, "7", "unknown Z must name the offending Z");
  expect_contains(msg, "1, 6, 8", "unknown Z must list every valid species");
}

void test_bridge_species_map_rejects_bad_table_size() {
  // ntypes is the table's length, so the one size that cannot be right is zero:
  // a pair style with no atom types has nothing to map and must not construct
  // into a state where every z_of call throws instead.
  const std::vector<int> none;
  (void)expect_throws(
      [&] {
        const SpeciesMap map(none, kMetaSpecies);
        (void)map;
      },
      "empty type table");
}

void test_bridge_species_map_rejects_nonpositive_z() {
  // -7 is not an atomic number. Caught at construction, where the LAMMPS type
  // that produced it is still known — by z_of time the caller only has an index.
  const std::vector<int> nonpositive = {1, -7, 8};
  const std::string msg = expect_throws(
      [&] {
        const SpeciesMap map(nonpositive, kMetaSpecies);
        (void)map;
      },
      "non-positive Z");
  expect_contains(msg, "2", "non-positive Z must name the offending LAMMPS type");
  expect_contains(msg, "-7", "non-positive Z must name the offending Z");
  expect_contains(msg, "1, 6, 8", "non-positive Z must list every valid species");
}

void test_bridge_species_map_z_of_rejects_out_of_range_type() {
  const SpeciesMap map(kZByType, kMetaSpecies);
  const int ntypes = map.ntypes();

  // LAMMPS types are 1-based, so 0 is not "the first type" — reading it would
  // walk off the front of a 0-based dense table.
  (void)expect_throws([&] { (void)map.z_of(0); }, "z_of(0)");
  (void)expect_throws([&] { (void)map.z_of(-1); }, "z_of(-1)");
  (void)expect_throws([&] { (void)map.z_of(ntypes + 1); },
                      "z_of(ntypes + 1)");

  // The boundary itself is valid: ntypes is the last type, not one past it.
  expect_int(map.z_of(ntypes), kZByType[static_cast<std::size_t>(ntypes) - 1],
             "z_of(ntypes) must be the last entry");
}

void test_bridge_force_scatter_rejects_wrong_row_count() {
  SyntheticAtoms atoms;
  const ForceScatter scatter(kForceConv);

  // A model returning the wrong number of rows is a contract violation, not
  // something to scatter as far as it goes: the check must fire before any
  // write, so f is still untouched afterwards.
  (void)expect_throws(
      [&] { scatter.scatter(atoms.view(), model_forces(kNlocal - 1), kNlocal); },
      "forces with too few rows");
  expect_forces_untouched(atoms, "too few rows");

  (void)expect_throws(
      [&] { scatter.scatter(atoms.view(), model_forces(kNlocal + 1), kNlocal); },
      "forces with too many rows");
  expect_forces_untouched(atoms, "too many rows");
}

void test_bridge_stage_nodes_rejects_n_graph_over_nall() {
  SyntheticAtoms atoms;
  const SpeciesMap map(kZByType, kMetaSpecies);

  // x belongs to LAMMPS. Reading past nall() is an out-of-bounds read of
  // another object's memory, which is why this throws rather than staging
  // whatever happens to follow the array.
  (void)expect_throws(
      [&] { (void)stage_nodes(atoms.view(), map, kNall + 1); },
      "stage_nodes with n_graph > nall");
  (void)expect_throws([&] { (void)stage_nodes(atoms.view(), map, -1); },
                      "stage_nodes with negative n_graph");

  // The boundary itself is valid: staging every local and ghost atom is exactly
  // what a supports_pbc=true export asks for.
  const NodeTensors all = stage_nodes(atoms.view(), map, kNall);
  expect_int(all.Z.size(0), kNall, "stage_nodes(n_graph == nall) size");
}

void test_bridge_force_scatter_rejects_n_graph_over_nall() {
  SyntheticAtoms atoms;
  const ForceScatter scatter(kForceConv);

  // Same boundary on the write side, where the consequence is worse: an
  // unchecked n_graph > nall() writes past the end of LAMMPS' f array and
  // corrupts the heap, with the symptom surfacing somewhere unrelated. The row
  // count matches n_graph here, so it is unambiguously the n_graph check that
  // must fire.
  (void)expect_throws(
      [&] { scatter.scatter(atoms.view(), model_forces(kNall + 1), kNall + 1); },
      "scatter with n_graph > nall");
  expect_forces_untouched(atoms, "n_graph > nall");

  // A negative n_graph cannot be matched by any row count, so the two guards
  // are indistinguishable from out here; what this asserts is the part that
  // matters either way — it throws, and f is left alone.
  (void)expect_throws(
      [&] { scatter.scatter(atoms.view(), model_forces(0), -1); },
      "scatter with negative n_graph");
  expect_forces_untouched(atoms, "negative n_graph");
}

// ---- dispatcher -----------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"bridge_species_map_maps_types", test_bridge_species_map_maps_types},
      {"bridge_stage_nodes_shapes_and_values",
       test_bridge_stage_nodes_shapes_and_values},
      {"bridge_force_scatter_accumulates", test_bridge_force_scatter_accumulates},
      {"bridge_species_map_rejects_unknown_z",
       test_bridge_species_map_rejects_unknown_z},
      {"bridge_species_map_rejects_bad_table_size",
       test_bridge_species_map_rejects_bad_table_size},
      {"bridge_species_map_rejects_nonpositive_z",
       test_bridge_species_map_rejects_nonpositive_z},
      {"bridge_species_map_z_of_rejects_out_of_range_type",
       test_bridge_species_map_z_of_rejects_out_of_range_type},
      {"bridge_force_scatter_rejects_wrong_row_count",
       test_bridge_force_scatter_rejects_wrong_row_count},
      {"bridge_stage_nodes_rejects_n_graph_over_nall",
       test_bridge_stage_nodes_rejects_n_graph_over_nall},
      {"bridge_force_scatter_rejects_n_graph_over_nall",
       test_bridge_force_scatter_rejects_n_graph_over_nall},
  };
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <test_name>\n", argv[0]);
    std::fprintf(stderr, "available:\n");
    for (const auto& [k, _] : registry()) std::fprintf(stderr, "  %s\n", k.c_str());
    return 2;
  }
  std::string name = argv[1];
  auto it = registry().find(name);
  if (it == registry().end()) {
    std::fprintf(stderr, "unknown test: %s\n", name.c_str());
    return 2;
  }
  try {
    it->second();
    std::fprintf(stderr, "[pass] %s\n", name.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[fail] %s: %s\n", name.c_str(), e.what());
    return 1;
  }
}
