// C++ acceptance tests for the graph-side LAMMPS bridge components
// (spec lammps-iface-03-graph): GraphBridge, the reverse-comm free functions
// and the backend-neutral execution strategy (Exec / DynamicExec / GraphedExec
// / pad_edges / read_model_outputs), driven over a synthetic LammpsView
// carrying neighbour spans.
//
// House style follows test_bridge_atoms.cpp: no test framework, one function
// per case, a function-local static map dispatcher, and the [pass]/[fail]
// stderr protocol. Most cases need NO fixtures — no exported model, no Python
// generation — and no LAMMPS build and no CUDA: LammpsView hands the bridge the
// same row-pointer / neighbour-span shapes out of the test's own std::vector
// storage that LAMMPS hands out of `atom->x` and `NeighList`. CMake therefore
// registers them without FIXTURES_REQUIRED. The two exceptions are named in the
// execution-side block near the bottom of this file and carry
// FIXTURES_REQUIRED.
//
// API contract under test (namespace molnex::lammps_bridge throughout; the
// include paths come from the molnex_lammps_bridge target's PUBLIC include
// directory, so `bridge/` is the prefix):
//
//   #include "bridge/lammps_view.h"
//
//   struct LammpsView {            // aggregate: no user-provided constructor
//     const double* const* x;
//     double* const* f;
//     const int* type;
//     int nlocal;
//     int nghost;
//     const int* ilist;            // (inum + gnum,) indices of the centres
//     const int* numneigh;         // indexed by local i
//     const int* const* firstneigh;// firstneigh[i] -> (numneigh[i],)
//     int inum;
//     int gnum;                    // ghost centres, listed after the inum local
//     int neigh_mask;              // the facade fills this from LAMMPS
//     int nall() const;
//     int neighbor(int i, int jj) const;   // firstneigh[i][jj] & neigh_mask
//   };
//
//   #include "bridge/graph_bridge.h"
//
//   class GraphBridge {
//    public:
//     GraphBridge(double cutoff, bool requires_sorted_edges);
//     at::Tensor build(const LammpsView& view, int n_graph);   // (ne, 2) int64
//   };
//
//   #include "bridge/comm_reverse.h"
//
//   int  pack_reverse_forces(const LammpsView&, int n, int first, double* buf);
//   void unpack_reverse_forces(const LammpsView&, int n, const int* list,
//                              const double* buf);
//
//   #include "bridge/exec.h"
//
//   struct ModelOutputs { at::Tensor energy, forces, atomic_energy, virial; };
//   struct OutputExpectations { bool supports_virial; int64_t n_atoms; };
//   ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
//                                   const OutputExpectations& expect);
//   class Exec {
//    public:
//     virtual ~Exec();
//     virtual ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
//                              const at::Tensor& edge_index,
//                              const at::Tensor& local_mask) = 0;
//     const std::string& model_dtype_name() const;
//   };
//
//   #include "bridge/dynamic_exec.h"
//   #include "bridge/graphed_exec.h"
//
//   class DynamicExec : public Exec { DynamicExec(ModelRunner*, at::Device,
//                                                 at::ScalarType, bool); };
//   struct PaddedEdges { at::Tensor edges; at::Tensor mask; };
//   PaddedEdges pad_edges(const at::Tensor& edge_index, int64_t e_max);
//   class GraphedExec : public Exec { GraphedExec(ModelRunner*, at::Device,
//                                                 at::ScalarType, bool,
//                                                 int64_t n_fixed,
//                                                 int64_t e_max); };
//
// The invariants these cases exist to lock:
//
//   1. The cutoff REFILTER is load-bearing, not redundant. LAMMPS builds its
//      neighbour list at cutoff + skin, so the list always contains pairs the
//      model must not see; dropping the refilter silently changes the energy
//      with no diagnostic. bridge_graph_cutoff_refilter is the case that would
//      catch that "optimisation".
//   2. Edge order is a CONTRACT DECLARATION, not a universal promise. Scatter-
//      class reductions are permutation-invariant, so the default path builds in
//      raw neighbour-list order and pays nothing; a model that needs segmented /
//      CSR aggregation exports `requires_sorted_edges = true` and only then does
//      the builder sort by (source, target). bridge_graph_order_shuffle_invariance
//      and bridge_graph_sorts_only_when_contract_requires are the two halves.
//   3. Column 0 is the source (the centre atom out of ilist) and column 1 the
//      target (its neighbour), per the repo-wide Edge Convention. Under a ghost
//      request the source domain is local u ghost, not local alone — that is
//      what makes the edge set symmetrically closed and therefore usable by
//      target-aggregating (MACE) and source-aggregating (PiNet) models alike,
//      with no per-model branch (bridge_graph_ghost_closure).
//   4. n_graph is validated before anything indexes view.x. That array belongs
//      to LAMMPS; an unchecked n_graph > nall() reads another object's memory.
//   5. ne == 0 must yield a well-formed (0, 2) int64 tensor. The migrated code
//      handed a possibly-null empty-vector data() to at::from_blob on that path.
//   6. unpack_reverse_forces ACCUMULATES (`+=`). Asserted from a non-zero
//      starting f, because a zero start cannot tell accumulate from overwrite.
//   7. `virial` is well-formed OR undefined, decided inside read_model_outputs.
//      A defined-but-short virial reaching the facade is exactly what the
//      migrated `numel() < 6` check in the tally existed to catch, and the point
//      of moving it is that it can no longer be reached.
//   8. pad_edges is CPU-testable. GraphedExec's end-to-end path needs CUDA and a
//      fixed-shape export; the padding/mask arithmetic must not.
//   9. The v3 output ORDER is [energy, forces, atomic_energy, (virial)] — the
//      virial moved from index 2 to index 3. The two size guards
//      (`numel == 6` for the virial, `size(0) == n_atoms` for the per-atom
//      energy) are SIMULTANEOUSLY BLIND at N == 6, which is the size of benzene,
//      two waters and most hand-written fixtures, so
//      bridge_read_model_outputs_six_atoms_no_transpose asserts at the VALUE
//      level. It is the only defence against a swap at that size.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "bridge/comm_reverse.h"
#include "bridge/dynamic_exec.h"
#include "bridge/exec.h"
#include "bridge/graph_bridge.h"
#include "bridge/graphed_exec.h"
#include "bridge/lammps_view.h"
#include "expect.h"
#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;

using molnex::interface::ModelRunner;
using molnex::lammps_bridge::DynamicExec;
using molnex::lammps_bridge::GraphBridge;
using molnex::lammps_bridge::GraphedExec;
using molnex::lammps_bridge::LammpsView;
using molnex::lammps_bridge::ModelOutputs;
using molnex::lammps_bridge::pack_reverse_forces;
using molnex::lammps_bridge::pad_edges;
using molnex::lammps_bridge::PaddedEdges;
using molnex::lammps_bridge::read_model_outputs;
using molnex::lammps_bridge::unpack_reverse_forces;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_string;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- the synthetic system --------------------------------------------------

// Three local atoms plus one ghost. The ghost is what makes the n_graph cases
// mean something: with n_graph == nlocal it must vanish from the graph, with
// n_graph == nall it must appear as a target (never as a source — REQ_FULL
// lists only local atoms in ilist).
constexpr int kNlocal = 3;
constexpr int kNghost = 1;
constexpr int kNall = kNlocal + kNghost;

// Model cutoff. The neighbour lists below are deliberately built as if by
// LAMMPS at cutoff + skin, i.e. every atom sees every other one.
constexpr double kCutoff = 2.0;   // cutsq = 4.0

// LAMMPS' own NEIGHMASK value, travelling as DATA through LammpsView. The token
// NEIGHMASK itself must never appear under bridge/ — the bridge applies the
// mask it is handed and reproduces no LAMMPS constant.
constexpr int kNeighMask = 0x3FFFFFFF;
// A special-bond style flag bit LAMMPS sets in the high bits of a neighbour
// entry. Bit 30 only: bit 31 would make the entry negative in a 32-bit int and
// the point here is the masking, not signedness.
constexpr int kNeighFlag = 1 << 30;

// Per-atom LAMMPS types, 1-based. Unused by the graph path but part of the
// view; carried so the synthetic system is a complete stand-in.
const std::vector<int> kType = {1, 2, 2, 3};

// Positions in angstrom. Squared distances, against cutsq = 4.0:
//
//     0-1  0.36 + 0.64 + 0.00 =  1.00   inside
//     0-2  4.00 + 0.64 + 1.00 =  5.64   OUTSIDE (in the list via the skin)
//     0-3  9.00 + 0.64 + 1.00 = 10.64   OUTSIDE
//     1-2  1.96 + 0.00 + 1.00 =  2.96   inside
//     1-3  5.76 + 0.00 + 1.00 =  6.76   OUTSIDE
//     2-3  1.00 + 0.00 + 0.00 =  1.00   inside
//
// Three of the six undirected pairs survive the refilter and three do not, so a
// missing refilter cannot pass by coincidence.
constexpr double kCoord[kNall][3] = {
    {0.0, 0.0, 0.0},
    {0.6, 0.8, 0.0},
    {2.0, 0.8, 1.0},
    {3.0, 0.8, 1.0},
};

// Forces already standing in f when the pair style runs: a non-zero start, as
// LAMMPS would have after an earlier pair style contributed. Every literal is
// exactly representable in binary floating point, so the comm cases assert with
// a zero tolerance and an accumulation that is off by one ulp still fails.
constexpr double kInitialForce[kNall][3] = {
    {1.5, -2.25, 0.75},
    {-0.5, 3.125, -1.0},
    {0.0, 0.5, -0.125},
    {7.0, 7.0, 7.0},
};

using EdgePair = std::pair<int, int>;

// Every surviving directed pair with n_graph == nall, in (source, target)
// lexicographic order. Hand-derived from the distance table above:
//   centre 0: 1 in, 2 and 3 out          -> (0,1)
//   centre 1: 0 and 2 in, 3 out          -> (1,0) (1,2)
//   centre 2: 1 and 3 in, 0 out          -> (2,1) (2,3)
const std::vector<EdgePair> kEdgesAll = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}};

// The same with n_graph == nlocal: the ghost is not a graph node, so (2,3) is
// the one row that goes — centre 2 keeps its local neighbour 1.
const std::vector<EdgePair> kEdgesLocal = {{0, 1}, {1, 0}, {1, 2}, {2, 1}};

// A sparser system's graph: atom 0 lists nobody and is not a centre, so the
// FIRST row differs from kEdgesAll's rather than merely the last. That is what
// makes the buffer-reuse case able to see an aliased return: a result whose
// leading rows agree with the previous one cannot distinguish a copy from a
// view of the same reused storage.
const std::vector<EdgePair> kEdgesSparse = {{1, 2}, {2, 1}, {2, 3}};

/// Full neighbour lists as LAMMPS would build them at cutoff + skin.
///
/// Every atom lists every other atom, including pairs well outside the model
/// cutoff — that is precisely the skin, and the reason the refilter exists. Row
/// 3 is the ghost's and stays empty: REQ_FULL puts only local atoms in ilist.
std::vector<std::vector<int>> full_lists() {
  return {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {}};
}

/// The storage a synthetic LammpsView borrows: atom arrays plus neighbour spans.
///
/// LAMMPS owns `atom->x` / `atom->f` as `double**` (an array of row pointers)
/// and `NeighList::firstneigh` as `int**` (one jlist per local atom). This
/// fixture reproduces exactly those shapes out of its own vectors, which is what
/// makes the bridge drivable with no LAMMPS build.
///
/// Non-copyable: the pointer vectors point into this object's own storage, so a
/// copy would hand out pointers into the source object.
class SyntheticSystem {
 public:
  /// \param jlists      One neighbour list per atom, indexed by local index;
  ///                    entries may carry high-bit flags (see `neigh_mask`).
  /// \param ilist       The centres, in the order LAMMPS would present them:
  ///                    `inum` local ones first, then `gnum` ghost ones.
  /// \param gnum        How many trailing `ilist` entries are ghost centres.
  ///                    Zero reproduces the pre-ghost request exactly, which is
  ///                    what keeps every case written before the ghost closure
  ///                    meaningful rather than merely still-compiling.
  /// \param neigh_mask  Mask applied to every `firstneigh` entry.
  explicit SyntheticSystem(std::vector<std::vector<int>> jlists = full_lists(),
                           std::vector<int> ilist = {0, 1, 2}, int gnum = 0,
                           int neigh_mask = kNeighMask)
      : type_(kType),
        jlists_(std::move(jlists)),
        ilist_(std::move(ilist)),
        gnum_(gnum),
        neigh_mask_(neigh_mask) {
    for (int i = 0; i < kNall; ++i) {
      coords_.push_back({kCoord[i][0], kCoord[i][1], kCoord[i][2]});
      forces_.push_back(
          {kInitialForce[i][0], kInitialForce[i][1], kInitialForce[i][2]});
    }
    for (int i = 0; i < kNall; ++i) {
      x_rows_.push_back(coords_[static_cast<std::size_t>(i)].data());
      f_rows_.push_back(forces_[static_cast<std::size_t>(i)].data());
      numneigh_.push_back(static_cast<int>(jlists_[static_cast<std::size_t>(i)].size()));
      firstneigh_.push_back(jlists_[static_cast<std::size_t>(i)].data());
    }
  }

  SyntheticSystem(const SyntheticSystem&) = delete;
  SyntheticSystem& operator=(const SyntheticSystem&) = delete;

  /// A view borrowing this fixture's arrays; valid while the fixture lives.
  ///
  /// `inum` is `ilist.size() - gnum`: LAMMPS lays the ghost centres out AFTER
  /// the local ones in the same array, so the two counts partition one list.
  LammpsView view() {
    return LammpsView{x_rows_.data(),
                      f_rows_.data(),
                      type_.data(),
                      kNlocal,
                      kNghost,
                      ilist_.data(),
                      numneigh_.data(),
                      firstneigh_.data(),
                      static_cast<int>(ilist_.size()) - gnum_,
                      gnum_,
                      neigh_mask_};
  }

  /// Current value of f[i][k] — what the component under test wrote.
  double force(int i, int k) const {
    return forces_[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
  }

  /// The raw, still-flagged neighbour entry, for asserting the test's own input.
  int raw_neighbor(int i, int jj) const {
    return jlists_[static_cast<std::size_t>(i)][static_cast<std::size_t>(jj)];
  }

 private:
  std::vector<std::array<double, 3>> coords_;
  std::vector<std::array<double, 3>> forces_;
  std::vector<int> type_;
  std::vector<std::vector<int>> jlists_;
  std::vector<int> ilist_;
  std::vector<const double*> x_rows_;
  std::vector<double*> f_rows_;
  std::vector<int> numneigh_;
  std::vector<const int*> firstneigh_;
  int gnum_;
  int neigh_mask_;
};

// ---- assertions ------------------------------------------------------------

/// Require `edges` to be exactly `want`, row by row, with the pinned shape.
void expect_edges(const at::Tensor& edges, const std::vector<EdgePair>& want,
                  const std::string& tag) {
  expect_true(edges.defined(), tag + ": edge_index must be a defined tensor");
  expect_true(edges.scalar_type() == at::kLong,
              tag + ": edge_index must be int64");
  expect_true(edges.device().is_cpu(),
              tag + ": edge_index must stay on the CPU — the device transfer "
                    "belongs to the execution site");
  expect_int(edges.dim(), 2, tag + ": edge_index must be 2-D");
  expect_int(edges.size(0), static_cast<long long>(want.size()),
             tag + ": edge count");
  expect_int(edges.size(1), 2, tag + ": edge_index must be (ne, 2)");

  const auto ea = edges.accessor<int64_t, 2>();
  for (std::size_t k = 0; k < want.size(); ++k) {
    const std::string row = tag + ": row " + std::to_string(k);
    expect_int(ea[static_cast<int64_t>(k)][0], want[k].first, row + " source");
    expect_int(ea[static_cast<int64_t>(k)][1], want[k].second, row + " target");
  }
}

// ---- GraphBridge -----------------------------------------------------------

void test_bridge_graph_edge_convention() {
  SyntheticSystem sys;
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  const at::Tensor edges = graph.build(sys.view(), kNall);
  expect_edges(edges, kEdgesAll, "edge convention");

  // Column 0 is the SOURCE: the centre atom out of ilist. This fixture declares
  // gnum == 0, so every centre it offers is local and the bound below is
  // nlocal; under a ghost request (bridge_graph_ghost_closure) the source domain
  // widens to local u ghost, which is what closes the edge set symmetrically.
  // Column 1 is the TARGET, which may be a ghost either way. The last row is
  // what makes this more than a shape check — (2, 3) is the local centre 2 with
  // the ghost 3 as its neighbour, and a transposed convention would produce
  // (3, 2), a source that is not even in ilist here.
  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 0; k < edges.size(0); ++k) {
    expect_true(ea[k][0] >= 0 && ea[k][0] < kNlocal,
                "source must be a local centre out of ilist");
    expect_true(ea[k][1] >= 0 && ea[k][1] < kNall,
                "target must be a graph node");
  }
}

void test_bridge_graph_cutoff_refilter() {
  SyntheticSystem sys;
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  // Nine candidate entries reach the loop (3 centres x 3 neighbours), five
  // survive. The refilter is what removes (0,2), (2,0), (0,3), (1,3) — pairs
  // LAMMPS put in the list because it builds at cutoff + skin. Keeping them
  // would silently change the energy, so this count is hand-derived from the
  // distance table at the top of the file, not recomputed here.
  const at::Tensor edges = graph.build(sys.view(), kNall);
  expect_int(edges.size(0), 5, "five of nine candidates survive the refilter");
  expect_edges(edges, kEdgesAll, "cutoff refilter");

  // Named absences: the two pairs that are in the neighbour list and outside
  // the cutoff in both directions.
  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 0; k < edges.size(0); ++k) {
    const bool is_0_2 = ea[k][0] == 0 && ea[k][1] == 2;
    const bool is_2_0 = ea[k][0] == 2 && ea[k][1] == 0;
    const bool is_1_3 = ea[k][0] == 1 && ea[k][1] == 3;
    expect_true(!is_0_2, "pair (0,2) is 2.375 A apart and must be refiltered");
    expect_true(!is_2_0, "pair (2,0) is 2.375 A apart and must be refiltered");
    expect_true(!is_1_3, "pair (1,3) is 2.600 A apart and must be refiltered");
  }
}

void test_bridge_graph_sorted_by_source_target() {
  // Both the centre order and every jlist are shuffled, and the contract asks
  // for sorted edges: whatever permutation the neighbour list happened to hand
  // over, a `requires_sorted_edges = true` export must see (source, target)
  // order — that is the promise a segmented / CSR aggregation depends on.
  SyntheticSystem sys({{3, 2, 1}, {2, 3, 0}, {3, 1, 0}, {}}, {2, 0, 1});
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/true);

  const at::Tensor edges = graph.build(sys.view(), kNall);

  // Same pairs, canonical order — identical to the unshuffled expectation.
  expect_edges(edges, kEdgesAll, "sorted from shuffled input");

  // And the order itself, stated as the invariant rather than as a literal:
  // non-decreasing lexicographic by (source, target).
  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 1; k < edges.size(0); ++k) {
    const bool ordered = ea[k - 1][0] < ea[k][0] ||
                         (ea[k - 1][0] == ea[k][0] && ea[k - 1][1] <= ea[k][1]);
    expect_true(ordered, "rows must be non-decreasing in (source, target)");
  }
}

void test_bridge_graph_drops_beyond_n_graph() {
  SyntheticSystem sys;
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  // n_graph == nlocal is the supports_pbc=false graph. The migrated code had
  // two filters here (`!supports_pbc && j >= nlocal` and `j >= n_graph`); this
  // locks that the single surviving one is equivalent — the ghost 3 is inside
  // the cutoff of centre 2, so only the n_graph test can remove it.
  const at::Tensor edges = graph.build(sys.view(), kNlocal);
  expect_edges(edges, kEdgesLocal, "n_graph == nlocal");

  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 0; k < edges.size(0); ++k) {
    expect_true(ea[k][1] < kNlocal, "no target may be a ghost when n_graph == nlocal");
  }

  // The same view with n_graph == nall keeps it: the pair is real, and it is
  // n_graph alone that decides whether the ghost is a node.
  expect_edges(graph.build(sys.view(), kNall), kEdgesAll, "n_graph == nall");
}

void test_bridge_graph_applies_neigh_mask() {
  // Every entry carries a special-bond style flag in bit 30, exactly as LAMMPS
  // writes them. Unmasked, `2 | (1 << 30)` is 1073741826, which is >= n_graph
  // and would be silently dropped — the graph would come back empty and the
  // model would see a molecule with no interactions at all.
  std::vector<std::vector<int>> flagged = full_lists();
  for (auto& jl : flagged) {
    for (int& j : jl) j |= kNeighFlag;
  }
  SyntheticSystem sys(flagged, {0, 1, 2}, /*gnum=*/0, kNeighMask);

  // The test's own input really is flagged: without this the case could pass
  // against a mask that is never applied.
  expect_true(sys.raw_neighbor(0, 0) != 1,
              "the fixture must hand the bridge flagged neighbour entries");
  expect_int(sys.raw_neighbor(0, 0), 1 | kNeighFlag, "flagged entry value");

  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);
  expect_edges(graph.build(sys.view(), kNall), kEdgesAll,
               "masked neighbour entries");
}

void test_bridge_graph_empty_neighbor_list() {
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  // (a) Every jlist empty — an isolated set of atoms, or a first step before
  // any pair comes within range. The reused host buffers are still empty here,
  // so this is the path where the migrated code handed a possibly-null
  // std::vector::data() to at::from_blob.
  // Spelled out rather than braced: with `gnum` and `neigh_mask` both
  // defaultable, a bare `{{}, {}, {}, {}}` is equally readable as four empty
  // jlists or as four default-initialised arguments, and the compiler says so.
  SyntheticSystem bare(std::vector<std::vector<int>>{{}, {}, {}, {}});
  const at::Tensor none = graph.build(bare.view(), kNall);
  expect_edges(none, {}, "empty neighbour lists");
  expect_int(none.numel(), 0, "empty edge_index must hold no elements");

  // (b) n_graph == 0 — no atom is a graph node, so every candidate is dropped
  // after the lists have actually been walked.
  SyntheticSystem sys;
  const at::Tensor dropped = graph.build(sys.view(), 0);
  expect_edges(dropped, {}, "n_graph == 0");

  // The empty result must not leave the component wedged: a subsequent build
  // over the same bridge returns the full graph.
  expect_edges(graph.build(sys.view(), kNall), kEdgesAll, "build after empty");
}

void test_bridge_graph_reuses_buffers_across_calls() {
  SyntheticSystem sys;
  // A system whose graph disagrees with the full one from its very first row:
  // atom 0 lists nobody and is not a centre.
  SyntheticSystem sparse({{}, {2, 3}, {1, 3}, {}}, {1, 2});
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  // Five edges, then three, then five again: the reused host buffers shrink and
  // grow across steps, which is where stale trailing entries would show up.
  const at::Tensor first = graph.build(sys.view(), kNall);
  expect_edges(first, kEdgesAll, "first build");

  const at::Tensor second = graph.build(sparse.view(), kNall);
  expect_edges(second, kEdgesSparse, "second build");

  // The first result must be untouched by the second build. It would not be if
  // the returned tensor aliased the reused buffer instead of owning a copy —
  // the caller's edge_index would silently change under it mid-step, and a
  // shrink leaves the surplus capacity holding the previous step's tail rather
  // than anything the caller could notice going wrong.
  expect_edges(first, kEdgesAll, "first build after the second");

  const at::Tensor third = graph.build(sys.view(), kNall);
  expect_edges(third, kEdgesAll, "third build");
  expect_edges(second, kEdgesSparse, "second build after the third");

  // A dropped-ghost graph is a prefix of the full one, so it goes here only to
  // exercise the shrink; the aliasing evidence is the sparse pair above.
  expect_edges(graph.build(sys.view(), kNlocal), kEdgesLocal, "fourth build");
}

void test_bridge_graph_rejects_n_graph_over_nall() {
  SyntheticSystem sys;
  GraphBridge graph(kCutoff, /*requires_sorted_edges=*/false);

  // n_graph is both the neighbour filter threshold and the upper bound of every
  // view.x read. Unchecked, an n_graph past nall() reads coordinates out of
  // memory LAMMPS does not own and turns them into edges.
  (void)expect_throws([&] { (void)graph.build(sys.view(), kNall + 1); },
                      "build with n_graph > nall");
  (void)expect_throws([&] { (void)graph.build(sys.view(), -1); },
                      "build with negative n_graph");

  // The boundary itself is valid: n_graph == nall is exactly what a
  // supports_pbc=true export asks for.
  expect_edges(graph.build(sys.view(), kNall), kEdgesAll, "n_graph == nall");
}

// ---- the ghost-closure system ----------------------------------------------
//
// A SECOND synthetic system, purpose-built for the D4 cases and deliberately
// not a variation on the one above: those four atoms have no ghost with a
// neighbour list, and the whole subject here is what happens when ghosts are
// centres too.
//
// Two local atoms and three ghosts on a line, spaced 1.0 A apart, with the
// cutoff at 1.5 A so exactly the adjacent pairs are edges:
//
//     index:    2     0     1     3     4
//     x:      -1.0   0.0   1.0   2.0   3.0
//     role:   ghost local local ghost ghost
//
// so the undirected edges are 2-0, 0-1, 1-3, 3-4 and the full directed
// enumeration has eight rows. Atom 4 is the atom that makes this fixture worth
// building: it is TWO hops from local atom 1, so a two-layer model reaches it,
// and it is only in the graph because ghost 3 is a centre.
constexpr int kClosureNlocal = 2;
constexpr int kClosureNghost = 3;
constexpr int kClosureNall = kClosureNlocal + kClosureNghost;
constexpr double kClosureCutoff = 1.5;  // cutsq = 2.25

constexpr double kClosureCoord[kClosureNall][3] = {
    {0.0, 0.0, 0.0},   // 0 local
    {1.0, 0.0, 0.0},   // 1 local
    {-1.0, 0.0, 0.0},  // 2 ghost
    {2.0, 0.0, 0.0},   // 3 ghost
    {3.0, 0.0, 0.0},   // 4 ghost
};

// Per-atom LAMMPS types; unread by the graph path but part of a complete view.
const std::vector<int> kClosureType = {1, 1, 1, 1, 1};

/// Neighbour lists as LAMMPS builds them at cutoff + skin, for EVERY atom that
/// has one — the locals and all three ghosts. Each list carries at least one
/// entry 2.0 A away, i.e. outside the model cutoff, so the refilter stays load-
/// bearing here too.
std::vector<std::vector<int>> closure_lists() {
  return {
      {1, 2, 3},     // 0: 1 in, 2 in, 3 out (2.0)
      {0, 3, 2, 4},  // 1: 0 in, 3 in, 2 out (2.0), 4 out (2.0)
      {0, 1},        // 2: 0 in, 1 out (2.0)
      {1, 4, 0},     // 3: 1 in, 4 in, 0 out (2.0)
      {3, 1},        // 4: 3 in, 1 out (2.0)
  };
}

/// Storage for the closure system's view. Same borrowing shape as
/// `SyntheticSystem`; a separate type because the geometry, the atom count and
/// the local/ghost split are all different.
class ClosureSystem {
 public:
  /// \param ilist  Centres in presentation order: locals first, then ghosts.
  /// \param gnum   How many trailing `ilist` entries are ghost centres.
  ClosureSystem(std::vector<int> ilist, int gnum,
                std::vector<std::vector<int>> jlists = closure_lists())
      : type_(kClosureType),
        jlists_(std::move(jlists)),
        ilist_(std::move(ilist)),
        gnum_(gnum) {
    for (int i = 0; i < kClosureNall; ++i) {
      coords_.push_back(
          {kClosureCoord[i][0], kClosureCoord[i][1], kClosureCoord[i][2]});
      forces_.push_back({0.0, 0.0, 0.0});
    }
    for (int i = 0; i < kClosureNall; ++i) {
      x_rows_.push_back(coords_[static_cast<std::size_t>(i)].data());
      f_rows_.push_back(forces_[static_cast<std::size_t>(i)].data());
      numneigh_.push_back(
          static_cast<int>(jlists_[static_cast<std::size_t>(i)].size()));
      firstneigh_.push_back(jlists_[static_cast<std::size_t>(i)].data());
    }
  }

  ClosureSystem(const ClosureSystem&) = delete;
  ClosureSystem& operator=(const ClosureSystem&) = delete;

  LammpsView view() {
    return LammpsView{x_rows_.data(),
                      f_rows_.data(),
                      type_.data(),
                      kClosureNlocal,
                      kClosureNghost,
                      ilist_.data(),
                      numneigh_.data(),
                      firstneigh_.data(),
                      static_cast<int>(ilist_.size()) - gnum_,
                      gnum_,
                      kNeighMask};
  }

 private:
  std::vector<std::array<double, 3>> coords_;
  std::vector<std::array<double, 3>> forces_;
  std::vector<int> type_;
  std::vector<std::vector<int>> jlists_;
  std::vector<int> ilist_;
  std::vector<const double*> x_rows_;
  std::vector<double*> f_rows_;
  std::vector<int> numneigh_;
  std::vector<const int*> firstneigh_;
  int gnum_;
};

/// Every ordered pair within the closure cutoff, enumerated from the positions.
///
/// The REFERENCE the builder is judged against, and computed here from the
/// geometry alone — not from the neighbour lists — so it cannot inherit a
/// mistake in the fixture's own jlists.
std::vector<EdgePair> closure_reference_edges() {
  std::vector<EdgePair> out;
  for (int i = 0; i < kClosureNall; ++i) {
    for (int j = 0; j < kClosureNall; ++j) {
      if (i == j) continue;
      double d2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double d = kClosureCoord[j][k] - kClosureCoord[i][k];
        d2 += d * d;
      }
      if (d2 < kClosureCutoff * kClosureCutoff) out.emplace_back(i, j);
    }
  }
  return out;
}

/// The rows of `edges`, as an order-independent sorted list of pairs.
std::vector<EdgePair> edge_set(const at::Tensor& edges) {
  std::vector<EdgePair> out;
  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 0; k < edges.size(0); ++k) {
    out.emplace_back(static_cast<int>(ea[k][0]), static_cast<int>(ea[k][1]));
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Require two edge collections to hold the same pairs, whatever their order.
void expect_same_edge_set(const at::Tensor& got, const std::vector<EdgePair>& want,
                          const std::string& tag) {
  std::vector<EdgePair> canonical = want;
  std::sort(canonical.begin(), canonical.end());
  const std::vector<EdgePair> mine = edge_set(got);
  expect_int(static_cast<long long>(mine.size()),
             static_cast<long long>(canonical.size()), tag + ": edge count");
  for (std::size_t k = 0; k < canonical.size(); ++k) {
    expect_int(mine[k].first, canonical[k].first,
               tag + ": pair " + std::to_string(k) + " source");
    expect_int(mine[k].second, canonical[k].second,
               tag + ": pair " + std::to_string(k) + " target");
  }
}

/// Two rounds of message passing over `edges`, aggregating onto one endpoint.
///
/// `to_target` picks the direction: true aggregates each edge onto its TARGET
/// (MACE's `_scatter_sum`), false onto its SOURCE (PiNet's `index_add_(0,
/// src)`). Two rounds, not one, because the two directions fail differently and
/// only depth exposes both: a target-aggregating model is already wrong at
/// layer 1 without ghost->local edges, while a source-aggregating one is right
/// at layer 1 and wrong from layer 2, when a local node reads a ghost's frozen
/// features.
///
/// \param edges      `(ne, 2)` int64 edge rows.
/// \param h0         Per-node seed values `(n,)`.
/// \param to_target  Aggregation direction; see above.
/// \returns The layer-2 node values `(n,)`.
std::vector<double> two_layer_aggregate(const std::vector<EdgePair>& edges,
                                        const std::vector<double>& h0,
                                        bool to_target) {
  const std::size_t n = h0.size();
  std::vector<double> h1(n, 0.0);
  for (const EdgePair& e : edges) {
    const std::size_t sink = static_cast<std::size_t>(to_target ? e.second : e.first);
    const std::size_t src = static_cast<std::size_t>(to_target ? e.first : e.second);
    h1[sink] += h0[src];
  }
  std::vector<double> h2(n, 0.0);
  for (const EdgePair& e : edges) {
    const std::size_t sink = static_cast<std::size_t>(to_target ? e.second : e.first);
    const std::size_t src = static_cast<std::size_t>(to_target ? e.first : e.second);
    h2[sink] += h1[src];
  }
  return h2;
}

void test_bridge_graph_ghost_closure() {
  // All five atoms are centres: two local, then three ghosts. This is the
  // `REQ_FULL | REQ_GHOST` list a supports_pbc export gets, and walking it whole
  // is the entire content of D4.2.
  ClosureSystem sys({0, 1, 2, 3, 4}, /*gnum=*/3);
  GraphBridge graph(kClosureCutoff, /*requires_sorted_edges=*/false);

  const at::Tensor edges = graph.build(sys.view(), kClosureNall);
  const std::vector<EdgePair> reference = closure_reference_edges();
  expect_int(static_cast<long long>(reference.size()), 8,
             "the reference enumeration is the four adjacent pairs, both ways");
  expect_same_edge_set(edges, reference, "ghost closure");

  // (1) The structural claim, stated per local atom: every INCOMING edge of a
  // local atom is present. That is what a target-aggregating model needs at its
  // very first layer, and precisely what a local-only walk loses — edge (2,0)
  // exists only because ghost 2 is a centre.
  const std::vector<EdgePair> built = edge_set(edges);
  for (const EdgePair& want : reference) {
    if (want.second >= kClosureNlocal) continue;  // incoming to a local only
    expect_true(std::find(built.begin(), built.end(), want) != built.end(),
                "incoming edge (" + std::to_string(want.first) + "," +
                    std::to_string(want.second) + ") of a local atom is missing");
  }

  // (2) The direction-independence claim, executed. Seeds are powers of two, so
  // every subset sum is unique and a single missing edge changes the result
  // exactly — there is no coincidental agreement available.
  const std::vector<double> h0 = {1.0, 2.0, 4.0, 8.0, 16.0};
  for (const bool to_target : {true, false}) {
    const std::string dir = to_target ? "target-aggregating (MACE)"
                                      : "source-aggregating (PiNet)";
    const std::vector<double> got = two_layer_aggregate(built, h0, to_target);
    const std::vector<double> want = two_layer_aggregate(reference, h0, to_target);
    for (int i = 0; i < kClosureNlocal; ++i) {
      expect_near(got[static_cast<std::size_t>(i)],
                  want[static_cast<std::size_t>(i)], 0.0,
                  dir + ": local atom " + std::to_string(i) +
                      " must reproduce the full-enumeration value");
    }
  }

  // (3) The negative control, and the reason this case is evidence rather than
  // decoration: the SAME builder over the SAME atoms with only the local
  // centres walked (gnum == 0, ilist truncated to the locals) disagrees with the
  // reference in BOTH directions. Without it, (2) would pass against a builder
  // that never looked at a ghost, since two identical computations over one
  // edge list always agree.
  ClosureSystem locals_only({0, 1}, /*gnum=*/0);
  const std::vector<EdgePair> narrow =
      edge_set(graph.build(locals_only.view(), kClosureNall));
  expect_true(narrow.size() < built.size(),
              "a local-only walk must produce strictly fewer edges");
  for (const bool to_target : {true, false}) {
    const std::vector<double> got = two_layer_aggregate(narrow, h0, to_target);
    const std::vector<double> want = two_layer_aggregate(reference, h0, to_target);
    bool differs = false;
    for (int i = 0; i < kClosureNlocal; ++i) {
      differs = differs || got[static_cast<std::size_t>(i)] !=
                               want[static_cast<std::size_t>(i)];
    }
    expect_true(differs,
                std::string("a local-only walk must corrupt the ") +
                    (to_target ? "target" : "source") +
                    "-aggregating result on a local atom — otherwise this case "
                    "proves nothing about the ghost traversal");
  }
}

void test_bridge_graph_order_shuffle_invariance() {
  // requires_sorted_edges == false, so the builder hands back whatever order
  // the neighbour list had. Two presentations of the same system: the same
  // centres and the same pairs, permuted.
  ClosureSystem plain({0, 1, 2, 3, 4}, /*gnum=*/3);
  ClosureSystem shuffled({3, 1, 4, 0, 2}, /*gnum=*/3,
                         {{2, 3, 1}, {4, 2, 3, 0}, {1, 0}, {0, 4, 1}, {1, 3}});
  GraphBridge graph(kClosureCutoff, /*requires_sorted_edges=*/false);

  // `build` returns an owned copy, so both results stay valid side by side —
  // which is what lets the two orders be compared row for row.
  const at::Tensor plain_edges = graph.build(plain.view(), kClosureNall);
  const at::Tensor shuffled_edges = graph.build(shuffled.view(), kClosureNall);

  // Same edges, and — the part that makes the case non-vacuous — a genuinely
  // different row order coming out of the builder.
  expect_same_edge_set(plain_edges, closure_reference_edges(),
                       "listed presentation yields the reference edge set");
  expect_same_edge_set(shuffled_edges, closure_reference_edges(),
                       "shuffled presentation yields the same edge set");

  const auto pa = plain_edges.accessor<int64_t, 2>();
  const auto sa = shuffled_edges.accessor<int64_t, 2>();
  bool row_order_differs = false;
  for (int64_t k = 0; k < plain_edges.size(0); ++k) {
    row_order_differs =
        row_order_differs || pa[k][0] != sa[k][0] || pa[k][1] != sa[k][1];
  }
  expect_true(row_order_differs,
              "the two presentations must actually differ in row order, or "
              "this case is not testing permutation invariance at all");

  // The physical claim: a scatter-class reduction over these edges is
  // permutation-invariant. Weights are irrational-ish on purpose — with exactly
  // representable values fp addition would be associative here and the
  // tolerance would prove nothing. 1e-12 in fp64 follows the (n-1)u bound for a
  // handful of edges.
  auto scatter_energy = [](const at::Tensor& edges) {
    std::vector<double> acc(kClosureNall, 0.0);
    const auto ea = edges.accessor<int64_t, 2>();
    for (int64_t k = 0; k < edges.size(0); ++k) {
      const int s = static_cast<int>(ea[k][0]);
      const int t = static_cast<int>(ea[k][1]);
      double d2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double d = kClosureCoord[t][c] - kClosureCoord[s][c];
        d2 += d * d;
      }
      acc[static_cast<std::size_t>(s)] += 1.0 / (d2 + 0.3);
    }
    double e = 0.0;
    for (const double v : acc) e += v * v;
    return e;
  };

  const double e_plain = scatter_energy(plain_edges);
  const double e_shuffled = scatter_energy(shuffled_edges);
  expect_true(e_plain > 0.0, "the stub energy must be non-trivial");
  expect_near(e_shuffled, e_plain, 1e-12,
              "a scatter-reduced energy must not depend on edge order");
}

void test_bridge_graph_sorts_only_when_contract_requires() {
  // One presentation, two contracts. The neighbour list is deliberately NOT in
  // (source, target) order: centre 3 comes first, and every jlist is shuffled.
  const std::vector<int> ilist = {3, 1, 4, 0, 2};
  const std::vector<std::vector<int>> jlists = {
      {2, 3, 1}, {4, 2, 3, 0}, {1, 0}, {0, 4, 1}, {1, 3}};

  // (a) The default contract: raw neighbour-list order, hand-derived from the
  // walk above. Centre 3 first (its list is {0, 4, 1}: 0 is 2.0 A away and goes,
  // 4 and 1 stay, in that order), then centre 1, and so on.
  ClosureSystem raw(ilist, /*gnum=*/3, jlists);
  GraphBridge unsorted(kClosureCutoff, /*requires_sorted_edges=*/false);
  const at::Tensor as_listed = unsorted.build(raw.view(), kClosureNall);
  const std::vector<EdgePair> want_raw = {
      {3, 4}, {3, 1},          // centre 3: {0 out, 4 in, 1 in}
      {1, 3}, {1, 0},          // centre 1: {4 out, 2 out, 3 in, 0 in}
      {4, 3},                  // centre 4: {1 out, 3 in}
      {0, 2}, {0, 1},          // centre 0: {2 in, 3 out, 1 in}
      {2, 0},                  // centre 2: {1 out, 0 in}
  };
  expect_edges(as_listed, want_raw, "raw neighbour order is preserved");

  // (b) The same input under a contract that declares it needs sorted edges.
  ClosureSystem sorted_sys(ilist, /*gnum=*/3, jlists);
  GraphBridge sorting(kClosureCutoff, /*requires_sorted_edges=*/true);
  const at::Tensor as_sorted = sorting.build(sorted_sys.view(), kClosureNall);
  const auto sa = as_sorted.accessor<int64_t, 2>();
  for (int64_t k = 1; k < as_sorted.size(0); ++k) {
    const bool ordered = sa[k - 1][0] < sa[k][0] ||
                         (sa[k - 1][0] == sa[k][0] && sa[k - 1][1] <= sa[k][1]);
    expect_true(ordered,
                "a requires_sorted_edges contract must see rows non-decreasing "
                "in (source, target)");
  }

  // (c) And the two differ ONLY in order: the edge set is identical, so the
  // flag buys ordering and never changes the physics.
  expect_same_edge_set(as_listed, closure_reference_edges(), "unsorted edge set");
  expect_same_edge_set(as_sorted, closure_reference_edges(), "sorted edge set");
  bool row_order_differs = false;
  const auto ua = as_listed.accessor<int64_t, 2>();
  for (int64_t k = 0; k < as_listed.size(0); ++k) {
    row_order_differs = row_order_differs || ua[k][0] != sa[k][0] || ua[k][1] != sa[k][1];
  }
  expect_true(row_order_differs,
              "the sorted and unsorted results must actually differ in row "
              "order, or the flag is doing nothing either way");
}

// ---- reverse comm ----------------------------------------------------------

void test_bridge_comm_pack_reverse_layout() {
  SyntheticSystem sys;

  // LAMMPS hands a buffer and a contiguous [first, first + n) range of atoms.
  // The sentinel past 3n catches a pack that writes more than it reports.
  constexpr double kSentinel = -12345.0;
  std::vector<double> buf(8, kSentinel);
  const int m = pack_reverse_forces(sys.view(), 2, 1, buf.data());

  expect_int(m, 6, "pack must return the number of doubles written, 3n");

  // fx, fy, fz per atom, atoms in order, starting at `first` — atom 1 then 2.
  expect_near(buf[0], kInitialForce[1][0], 0.0, "buf[0] == f[1][0]");
  expect_near(buf[1], kInitialForce[1][1], 0.0, "buf[1] == f[1][1]");
  expect_near(buf[2], kInitialForce[1][2], 0.0, "buf[2] == f[1][2]");
  expect_near(buf[3], kInitialForce[2][0], 0.0, "buf[3] == f[2][0]");
  expect_near(buf[4], kInitialForce[2][1], 0.0, "buf[4] == f[2][1]");
  expect_near(buf[5], kInitialForce[2][2], 0.0, "buf[5] == f[2][2]");

  expect_near(buf[6], kSentinel, 0.0, "pack must not write past 3n");
  expect_near(buf[7], kSentinel, 0.0, "pack must not write past 3n");

  // Packing reads f and must not change it.
  for (int i = 0; i < kNall; ++i) {
    for (int k = 0; k < 3; ++k) {
      expect_near(sys.force(i, k), kInitialForce[i][k], 0.0,
                  "pack must leave f[" + std::to_string(i) + "][" +
                      std::to_string(k) + "] untouched");
    }
  }
}

void test_bridge_comm_unpack_accumulates() {
  SyntheticSystem sys;

  // The list is out of order on purpose: the i-th triple in buf belongs to
  // atom list[i], not to atom i.
  const std::vector<int> list = {3, 0};
  const std::vector<double> buf = {0.25, -0.5, 1.0, -2.0, 0.125, 4.0};

  unpack_reverse_forces(sys.view(), 2, list.data(), buf.data());

  // Hand-computed from the non-zero starting f. Every literal is exact in
  // binary floating point, so the tolerance is zero: an overwrite lands on the
  // buf values themselves and fails every one of these.
  //   f[3] = (7.0, 7.0, 7.0)      + (0.25, -0.5, 1.0)   = (7.25, 6.5, 8.0)
  //   f[0] = (1.5, -2.25, 0.75)   + (-2.0, 0.125, 4.0)  = (-0.5, -2.125, 4.75)
  expect_near(sys.force(3, 0), 7.25, 0.0, "f[3][0] += buf[0]");
  expect_near(sys.force(3, 1), 6.5, 0.0, "f[3][1] += buf[1]");
  expect_near(sys.force(3, 2), 8.0, 0.0, "f[3][2] += buf[2]");
  expect_near(sys.force(0, 0), -0.5, 0.0, "f[0][0] += buf[3]");
  expect_near(sys.force(0, 1), -2.125, 0.0, "f[0][1] += buf[4]");
  expect_near(sys.force(0, 2), 4.75, 0.0, "f[0][2] += buf[5]");

  // Atoms absent from the list keep their starting force.
  for (const int i : {1, 2}) {
    for (int k = 0; k < 3; ++k) {
      expect_near(sys.force(i, k), kInitialForce[i][k], 0.0,
                  "f[" + std::to_string(i) + "][" + std::to_string(k) +
                      "] is not in the list and must be untouched");
    }
  }
}

// ---- execution strategy: no-fixture cases ----------------------------------

/// A `(ne, 2)` int64 CPU edge tensor from literal rows, as GraphBridge yields.
at::Tensor make_edges(const std::vector<EdgePair>& rows) {
  const auto ne = static_cast<int64_t>(rows.size());
  at::Tensor t = at::zeros({ne, 2}, at::TensorOptions().dtype(at::kLong));
  auto a = t.accessor<int64_t, 2>();
  for (int64_t k = 0; k < ne; ++k) {
    a[k][0] = rows[static_cast<std::size_t>(k)].first;
    a[k][1] = rows[static_cast<std::size_t>(k)].second;
  }
  return t;
}

/// Assert one padding result: real rows, inert tail, mask, shapes and dtypes.
void expect_padded(const PaddedEdges& p, const std::vector<EdgePair>& rows,
                   int64_t e_max, const std::string& tag) {
  const auto ne = static_cast<int64_t>(rows.size());

  expect_true(p.edges.defined(), tag + ": edges must be defined");
  expect_true(p.mask.defined(), tag + ": mask must be defined");
  expect_true(p.edges.scalar_type() == at::kLong, tag + ": edges must be int64");
  expect_true(p.mask.scalar_type() == at::kBool, tag + ": mask must be bool");
  expect_true(p.edges.device().is_cpu(), tag + ": edges must stay on the CPU");
  expect_true(p.mask.device().is_cpu(), tag + ": mask must stay on the CPU");
  expect_int(p.edges.dim(), 2, tag + ": edges must be 2-D");
  expect_int(p.edges.size(0), e_max, tag + ": edges must have e_max rows");
  expect_int(p.edges.size(1), 2, tag + ": edges must be (e_max, 2)");
  expect_int(p.mask.dim(), 1, tag + ": mask must be 1-D");
  expect_int(p.mask.size(0), e_max, tag + ": mask must have e_max entries");

  const auto ea = p.edges.accessor<int64_t, 2>();
  const auto ma = p.mask.accessor<bool, 1>();
  for (int64_t k = 0; k < e_max; ++k) {
    const std::string row = tag + ": row " + std::to_string(k);
    if (k < ne) {
      expect_int(ea[k][0], rows[static_cast<std::size_t>(k)].first, row + " source");
      expect_int(ea[k][1], rows[static_cast<std::size_t>(k)].second, row + " target");
      expect_true(ma[k], row + " must be masked in");
    } else {
      // The inert tail is (0, 0): a self-loop on atom 0, which the model must
      // never see as an edge — the mask is the only thing making it harmless.
      expect_int(ea[k][0], 0, row + " padding source must be 0");
      expect_int(ea[k][1], 0, row + " padding target must be 0");
      expect_true(!ma[k], row + " must be masked out");
    }
  }
}

void test_bridge_pad_edges() {
  // (a) ne < e_max — the ordinary step: real edges up front, inert tail behind.
  expect_padded(pad_edges(make_edges(kEdgesSparse), 8), kEdgesSparse, 8,
                "ne < e_max");

  // (b) ne == e_max — the boundary the CUDA-graph export was sized for. An
  // off-by-one in the guard shows up here and nowhere else.
  expect_padded(pad_edges(make_edges(kEdgesAll), 5), kEdgesAll, 5, "ne == e_max");

  // (c) ne == 0 — no pair within cutoff. Every row is padding and the mask is
  // all false, so the replayed graph contributes nothing rather than folding in
  // e_max self-loops on atom 0.
  expect_padded(pad_edges(make_edges({}), 4), {}, 4, "ne == 0");

  // (d) ne == 0 with e_max == 0 — degenerate but well-formed, not a null deref.
  expect_padded(pad_edges(make_edges({}), 0), {}, 0, "ne == 0, e_max == 0");

  // (e) Overflow is refused, not truncated. Silently dropping edges past e_max
  // would change the energy with no diagnostic, which is the whole reason the
  // fixed-shape path carries an explicit bound.
  const std::string msg = expect_throws(
      [] { (void)pad_edges(make_edges(kEdgesAll), 2); }, "ne > e_max");
  expect_contains(msg, "5", "overflow message must state the observed edge count");
  expect_contains(msg, "2", "overflow message must state e_max");

  // (f) The input is not consumed or aliased: padding a tensor must leave it
  // usable, and a caller holding it must not see the padded copy's rows.
  const at::Tensor src = make_edges(kEdgesSparse);
  const PaddedEdges p = pad_edges(src, 6);
  expect_int(src.size(0), 3, "the source edge tensor keeps its own row count");
  p.edges.zero_();
  const auto sa = src.accessor<int64_t, 2>();
  expect_int(sa[0][0], kEdgesSparse[0].first, "source rows survive a write to the padded copy");
  expect_int(sa[0][1], kEdgesSparse[0].second, "source rows survive a write to the padded copy");
}

/// A `(n,)` float64 CPU tensor of `value`, standing in for a model output.
at::Tensor f64_of(int64_t n, double value) {
  return at::full({n}, value, at::TensorOptions().dtype(at::kDouble));
}

/// A `(n,)` float64 CPU tensor holding `values`, in order.
at::Tensor f64_from(const std::vector<double>& values) {
  at::Tensor t = at::zeros({static_cast<int64_t>(values.size())},
                           at::TensorOptions().dtype(at::kDouble));
  auto a = t.accessor<double, 1>();
  for (std::size_t k = 0; k < values.size(); ++k) {
    a[static_cast<int64_t>(k)] = values[k];
  }
  return t;
}

/// The expectation struct every case below hands the decoder.
///
/// It exists because a third contract switch reaching `read_model_outputs` as
/// another bool would make the call sites unreadable — `(raw, true, 6)` says
/// nothing — while the same two facts named in a struct do.
molnex::lammps_bridge::OutputExpectations expect_of(bool supports_virial,
                                                    int64_t n_atoms) {
  return molnex::lammps_bridge::OutputExpectations{supports_virial, n_atoms};
}

void test_bridge_read_model_outputs_rejects_short_virial() {
  // A model that declares supports_virial but emits fewer than six components
  // used to reach the facade's tally, which then read six anyway. The check
  // belongs here: the virial either comes out well-formed or does not come out.
  // Under v3 the virial is output THREE, behind the per-atom energy.
  const std::vector<at::Tensor> raw = {
      f64_of(1, -3.5),
      at::zeros({4, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(4, 0.5),
      f64_of(3, 1.0),
  };

  const std::string msg = expect_throws(
      [&] { (void)read_model_outputs(raw, expect_of(true, 4)); },
      "virial with numel 3");
  expect_contains(msg, "3", "the diagnostic must state the observed numel");
  expect_contains(msg, "6", "the diagnostic must state the required numel");

  // A seven-element virial is just as wrong as a three-element one: the check
  // is an equality, not a lower bound (`>= 6` would let an off-by-one layout
  // through and silently read the wrong six components).
  const std::vector<at::Tensor> too_long = {raw[0], raw[1], raw[2], f64_of(7, 1.0)};
  (void)expect_throws(
      [&] { (void)read_model_outputs(too_long, expect_of(true, 4)); },
      "virial with numel 7");

  // Exactly six is accepted, so the case above is not passing by rejecting
  // every virial there is.
  const std::vector<at::Tensor> good = {raw[0], raw[1], raw[2], f64_of(6, 2.0)};
  const ModelOutputs out = read_model_outputs(good, expect_of(true, 4));
  expect_true(out.virial.defined(), "a six-element virial must be accepted");
  expect_int(out.virial.numel(), 6, "the accepted virial keeps its six elements");
  expect_true(out.virial.scalar_type() == at::kDouble, "virial must be float64");
  expect_true(out.virial.device().is_cpu(), "virial must be a CPU tensor");
}

void test_bridge_read_model_outputs_absent_virial_undefined() {
  // supports_virial is true but the model emitted only the three mandatory
  // outputs. That is tolerated, exactly as the migrated `out.size() >= 3` was:
  // the flag says the facade may tally a virial, not that one is present on
  // every call.
  const std::vector<at::Tensor> raw = {
      f64_of(1, 7.25),
      at::ones({3, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(3, 2.5),
  };

  const ModelOutputs out = read_model_outputs(raw, expect_of(true, 3));
  expect_true(!out.virial.defined(),
              "an absent virial must stay undefined, not become an empty tensor");
  expect_true(out.energy.defined(), "energy must still be read");
  expect_true(out.forces.defined(), "forces must still be read");
  expect_true(out.atomic_energy.defined(), "atomic_energy must still be read");
  expect_near(out.energy.item<double>(), 7.25, 0.0, "energy value");
  expect_true(out.forces.is_contiguous(), "forces must be contiguous");
}

void test_bridge_read_model_outputs_ignores_virial_when_unsupported() {
  // The model emitted a fourth tensor but the export never declared a virial.
  // Reading it would tally a quantity the contract does not claim — and a
  // malformed fourth output must not even be inspected, so this deliberately
  // carries a numel the supports_virial=true path would reject.
  const std::vector<at::Tensor> raw = {
      f64_of(1, -1.0),
      at::zeros({2, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(2, -0.5),
      f64_of(3, 99.0),
  };

  const ModelOutputs out = read_model_outputs(raw, expect_of(false, 2));
  expect_true(!out.virial.defined(),
              "supports_virial=false must leave virial undefined");
  expect_true(out.energy.defined(), "energy must still be read");
  expect_true(out.forces.defined(), "forces must still be read");
  expect_true(out.atomic_energy.defined(),
              "atomic_energy is mandatory under v3, virial flag or not");
}

void test_bridge_read_model_outputs_requires_atomic_energy() {
  // Per-atom energy is NOT optional under v3: it is the only thing the partition
  // sum can be computed from, so a model returning the v2 pair would produce a
  // zero eng_vdwl rather than an error. Two outputs is refused.
  const std::vector<at::Tensor> v2_shaped = {
      f64_of(1, 1.0),
      at::zeros({3, 3}, at::TensorOptions().dtype(at::kDouble)),
  };
  const std::string few = expect_throws(
      [&] { (void)read_model_outputs(v2_shaped, expect_of(false, 3)); },
      "a v2-shaped two-output model");
  expect_contains(few, "2", "the diagnostic must state the observed count");
  expect_contains(few, "3", "the diagnostic must state the required count");
  expect_contains(few, "atomic_energy",
                  "the diagnostic must name the output that is missing");

  // And its LENGTH is checked against the atom count the caller staged. A
  // per-atom energy that is not per-atom would be sliced to nlocal downstream
  // and silently sum the wrong sites.
  const std::vector<at::Tensor> wrong_length = {
      f64_of(1, 1.0),
      at::zeros({5, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(4, 0.25),
  };
  const std::string msg = expect_throws(
      [&] { (void)read_model_outputs(wrong_length, expect_of(false, 5)); },
      "atomic_energy with the wrong length");
  expect_contains(msg, "4", "the diagnostic must state the observed length");
  expect_contains(msg, "5", "the diagnostic must state the expected atom count");
}

void test_bridge_read_model_outputs_six_atoms_no_transpose() {
  // SIX ATOMS: the one size at which BOTH size guards go blind at once. A
  // (6,) virial passes `size(0) == n_atoms` and a (6,) per-atom energy passes
  // `numel == 6`, so a swapped pair of indices produces two well-formed,
  // completely wrong tensors — and six atoms is benzene, two waters, and most
  // hand-written fixtures. This is therefore a VALUE-level assertion, and it is
  // the only defence there is at N == 6.
  constexpr int64_t kSixAtoms = 6;
  const std::vector<double> site_energies = {-1.5, -2.5, -3.5, -4.5, -5.5, -6.5};
  const std::vector<double> virial_components = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

  const std::vector<at::Tensor> raw = {
      f64_of(1, -21.0),
      at::zeros({kSixAtoms, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_from(site_energies),
      f64_from(virial_components),
  };

  const ModelOutputs out = read_model_outputs(raw, expect_of(true, kSixAtoms));

  expect_true(out.atomic_energy.defined(), "atomic_energy must be defined");
  expect_true(out.virial.defined(), "virial must be defined");
  expect_int(out.atomic_energy.size(0), kSixAtoms, "atomic_energy is (6,)");
  expect_int(out.virial.numel(), 6, "virial is (6,)");

  // Disjoint value ranges: the site energies are all negative, the virial
  // components all positive and an order of magnitude larger. A transposition
  // fails every one of the twelve comparisons below, not just one.
  const auto ae = out.atomic_energy.accessor<double, 1>();
  const auto va = out.virial.accessor<double, 1>();
  for (int64_t k = 0; k < kSixAtoms; ++k) {
    expect_near(ae[k], site_energies[static_cast<std::size_t>(k)], 0.0,
                "atomic_energy[" + std::to_string(k) +
                    "] must come from output index 2, not from the virial");
    expect_near(va[k], virial_components[static_cast<std::size_t>(k)], 0.0,
                "virial[" + std::to_string(k) +
                    "] must come from output index 3, not from the per-atom "
                    "energy");
  }
}

void test_bridge_graphed_exec_cpu_only_ctor_throws() {
#ifdef MOLNEX_INTERFACE_CUDA
  // The diagnostic under test is unreachable in this build by construction.
  // The case still exists, and still runs, so a CPU-only build (which is what
  // CI configures) gates it.
  std::fprintf(stderr,
               "[skip] built with CUDA — the CPU-only ctor diagnostic is "
               "unreachable here\n");
#else
  // GraphedExec's class is declared UNCONDITIONALLY so a CPU-only build still
  // compiles, links and can instantiate it; what is guarded is the run body.
  // Constructing one must therefore fail loudly rather than hand back an object
  // whose run() is missing. Note the runner pointer is never dereferenced: the
  // refusal precedes every other check.
  const std::string msg = expect_throws(
      [] {
        GraphedExec exec(/*runner=*/nullptr, at::Device(at::kCPU), at::kDouble,
                         /*supports_virial=*/false, /*n_fixed=*/4,
                         /*e_max=*/8);
        (void)exec;
      },
      "GraphedExec on a CPU-only build");
  expect_contains(msg, "CPU-only", "the diagnostic must name the CPU-only build");
#endif
}

// ---- execution strategy: fixture-backed cases ------------------------------
//
// These two need the exported schema-v3 graph-potential package from
// generate_fixtures.py, so CMake registers them with FIXTURES_REQUIRED and the
// MOLNEX_INTERFACE_FIXTURE_DIR environment variable — the same wiring as
// contract_reads_generated_v3_fixture.

fs::path fixture_dir() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr || std::string(env).empty()) {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  return fs::path(env);
}

/// Load a `torch.save({name: tensor, ...})` pickle as a name -> tensor map.
std::map<std::string, at::Tensor> load_tensor_dict(const fs::path& pt_path) {
  std::ifstream is(pt_path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + pt_path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                          std::istreambuf_iterator<char>());
  auto iv = torch::pickle_load(bytes);
  std::map<std::string, at::Tensor> out;
  for (const auto& kv : iv.toGenericDict()) {
    out.emplace(kv.key().toStringRef(), kv.value().toTensor());
  }
  return out;
}

/// Fetch `key` from a reference dict, naming the file when it is missing.
at::Tensor reference_entry(const std::map<std::string, at::Tensor>& ref,
                           const std::string& key, const fs::path& from) {
  auto it = ref.find(key);
  if (it == ref.end()) {
    throw std::runtime_error("reference " + from.string() + " has no '" + key + "'");
  }
  return it->second;
}

void expect_allclose_f64(const at::Tensor& got, const at::Tensor& ref,
                         double atol, const std::string& tag) {
  auto a = got.detach().to(at::kCPU).to(at::kDouble).contiguous();
  auto b = ref.detach().to(at::kCPU).to(at::kDouble).contiguous();
  if (!a.sizes().equals(b.sizes())) {
    throw std::runtime_error(tag + ": shape mismatch " + c10::str(a.sizes()) +
                             " vs " + c10::str(b.sizes()));
  }
  const double diff = (a - b).abs().max().item<double>();
  if (diff > atol) {
    throw std::runtime_error(tag + ": max abs diff " + std::to_string(diff) +
                             " exceeds atol " + std::to_string(atol));
  }
}

void test_bridge_dynamic_exec_matches_reference() {
  const fs::path dir = fixture_dir();
  const fs::path ref_path = dir / "reference_graph_v3_cpu.pt";
  const auto ref = load_tensor_dict(ref_path);

  const at::Tensor Z = reference_entry(ref, "Z", ref_path);
  const at::Tensor pos = reference_entry(ref, "pos", ref_path);
  const at::Tensor edge_index = reference_entry(ref, "edge_index", ref_path);
  // All-ones: the fixture's reference outputs were captured at full mask, which
  // is the case whose energy equals the undecomposed total. A partial mask is
  // the impl's subject (impl_energy_sums_locals_only), not the execution
  // strategy's — this case is about the tensor plumbing.
  const at::Tensor local_mask = reference_entry(ref, "local_mask", ref_path);

  ModelRunner runner((dir / "model_graph_v3_cpu").string());
  expect_string(runner.device(), "cpu", "the graph fixture is a CPU export");

  DynamicExec exec(&runner, at::Device(at::kCPU), at::kDouble,
                   /*supports_virial=*/false);

  // The dtype name the facade prints in its logmesg. It lives here because the
  // execution site is the only reader of the model dtype; a copy on the facade
  // could only drift from this one.
  expect_string(exec.model_dtype_name(), "float64",
                "the fixture is an fp64 export");

  const ModelOutputs out = exec.run(Z, pos, edge_index, local_mask);

  // fp64 in, fp64 through, fp64 out: the fixture's outputs are a closed-form
  // expression of its inputs, so 1e-12 is a real parity claim and not a
  // tolerance chosen to make a mismatch fit.
  expect_allclose_f64(out.energy, reference_entry(ref, "energy", ref_path), 1e-12,
                      "dynamic exec energy");
  expect_allclose_f64(out.forces, reference_entry(ref, "forces", ref_path), 1e-12,
                      "dynamic exec forces");
  // The third output, asserted at the same tolerance as the other two: under v3
  // it is the one the impl actually partitions and tallies, so it cannot be the
  // one output nobody checks.
  expect_allclose_f64(out.atomic_energy,
                      reference_entry(ref, "atomic_energy", ref_path), 1e-12,
                      "dynamic exec atomic_energy");

  // The output contract, asserted rather than assumed: the facade's tally does
  // `e_t.item<double>()` and ForceScatter does `forces.accessor<double, 2>()`,
  // both of which are undefined on a non-CPU or non-float64 tensor.
  expect_true(out.energy.defined(), "energy must be defined");
  expect_true(out.energy.device().is_cpu(), "energy must come back on the CPU");
  expect_true(out.energy.scalar_type() == at::kDouble, "energy must be float64");
  expect_int(out.energy.numel(), 1, "energy must be a single scalar");
  expect_true(out.forces.defined(), "forces must be defined");
  expect_true(out.forces.device().is_cpu(), "forces must come back on the CPU");
  expect_true(out.forces.scalar_type() == at::kDouble, "forces must be float64");
  expect_true(out.forces.is_contiguous(), "forces must be contiguous");
  expect_int(out.forces.dim(), 2, "forces must be (N, 3)");
  expect_int(out.forces.size(0), Z.size(0), "one force row per atom");
  expect_int(out.forces.size(1), 3, "three force components per atom");

  // supports_virial=false: no virial, and specifically not an empty tensor that
  // would read as "present but zero" to a defined() check.
  expect_true(!out.virial.defined(),
              "supports_virial=false must leave the virial undefined");

  // Re-running must not hand back the previous call's tensors. This is the
  // owned-CPU-copy contract stated as a test: the graphed path returns replay
  // buffers that the next replay overwrites in place, and the dynamic path is
  // held to the same promise so the facade can rely on it uniformly.
  const at::Tensor first_forces = out.forces;
  const ModelOutputs again = exec.run(Z, pos, edge_index, local_mask);
  expect_true(again.forces.data_ptr() != first_forces.data_ptr(),
              "a second run must not alias the first run's force storage");
  expect_allclose_f64(again.forces, first_forces, 0.0,
                      "a second run on the same input is bit-identical");
}

void test_bridge_graphed_exec_matches_dynamic() {
  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "[skip] CUDA not available\n");
    return;
  }
  const fs::path dir = fixture_dir();
  if (!fs::exists(dir / "model_graph_v3_cuda") ||
      !fs::exists(dir / "model_graph_fixed_v3_cuda")) {
    // generate_fixtures.py emits the CUDA graph-potential exports only on a
    // host with a visible GPU; without them there is nothing to compare.
    std::fprintf(stderr, "[skip] cuda graph-potential fixture missing\n");
    return;
  }

  const fs::path ref_path = dir / "reference_graph_v3_cuda.pt";
  const auto ref = load_tensor_dict(ref_path);
  const at::Tensor Z = reference_entry(ref, "Z", ref_path).to(at::kCPU);
  const at::Tensor pos = reference_entry(ref, "pos", ref_path).to(at::kCPU);
  const at::Tensor edge_index = reference_entry(ref, "edge_index", ref_path).to(at::kCPU);
  const at::Tensor local_mask =
      reference_entry(ref, "local_mask", ref_path).to(at::kCPU);

  const int64_t n_fixed = Z.size(0);
  // Read, not recomputed: the fixed-shape package was COMPILED for this
  // capacity, so a literal here would be a second copy of a number that only
  // generate_fixtures.py can decide. It is deliberately larger than the edge
  // count — the padded tail is the part of the fixed-shape path a same-sized
  // export would never exercise.
  const int64_t e_max =
      reference_entry(ref, "e_max", ref_path).item<int64_t>();

  ModelRunner dyn_runner((dir / "model_graph_v3_cuda").string());
  DynamicExec dynamic(&dyn_runner, at::Device(at::kCUDA), at::kDouble,
                      /*supports_virial=*/false);
  const ModelOutputs want = dynamic.run(Z, pos, edge_index, local_mask);

  ModelRunner graph_runner((dir / "model_graph_fixed_v3_cuda").string());
  GraphedExec graphed(&graph_runner, at::Device(at::kCUDA), at::kDouble,
                      /*supports_virial=*/false, n_fixed, e_max);
  const ModelOutputs got = graphed.run(Z, pos, edge_index, local_mask);

  // rel 1e-6 is this chain's parity judgement; energies here are O(1), so an
  // absolute tolerance of the same size is the stricter reading of it.
  expect_allclose_f64(got.energy, want.energy, 1e-6, "graphed vs dynamic energy");
  expect_allclose_f64(got.forces, want.forces, 1e-6, "graphed vs dynamic forces");
  // The per-atom energy travels the fixed-shape path too — and it is the output
  // whose index moved under v3, so an unpadded-vs-padded disagreement here is
  // exactly the class of bug the five-input replay could introduce.
  expect_allclose_f64(got.atomic_energy, want.atomic_energy, 1e-6,
                      "graphed vs dynamic atomic_energy");
  expect_true(got.forces.device().is_cpu(), "graphed forces must come back on the CPU");
  expect_true(got.forces.scalar_type() == at::kDouble, "graphed forces must be float64");
  expect_true(got.forces.is_contiguous(), "graphed forces must be contiguous");
  expect_true(!got.virial.defined(), "supports_virial=false leaves virial undefined");

  // The replay-aliasing hazard, stated as an assertion: run_graphed hands back
  // tensors at FIXED addresses that the next replay overwrites in place. A
  // GraphedExec returning those directly would silently rewrite the caller's
  // previous-step forces.
  const at::Tensor first_forces = got.forces;
  const ModelOutputs second = graphed.run(Z, pos, edge_index, local_mask);
  expect_true(second.forces.data_ptr() != first_forces.data_ptr(),
              "a replay must not hand back the previous call's storage");
  expect_allclose_f64(first_forces, want.forces, 1e-6,
                      "the first result must survive the second replay intact");

  // Beyond e_max the fixed-shape path refuses rather than truncating.
  std::vector<EdgePair> too_many;
  for (int64_t k = 0; k < e_max + 1; ++k) too_many.emplace_back(0, 0);
  (void)expect_throws(
      [&] { (void)graphed.run(Z, pos, make_edges(too_many), local_mask); },
      "graphed run past e_max");

  // And a node count other than the exported fixed N is refused too.
  (void)expect_throws(
      [&] {
        (void)graphed.run(Z.narrow(0, 0, n_fixed - 1),
                          pos.narrow(0, 0, n_fixed - 1), make_edges({}),
                          local_mask.narrow(0, 0, n_fixed - 1));
      },
      "graphed run with the wrong atom count");
}

// ---- dispatcher -----------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"bridge_graph_edge_convention", test_bridge_graph_edge_convention},
      {"bridge_graph_cutoff_refilter", test_bridge_graph_cutoff_refilter},
      {"bridge_graph_sorted_by_source_target",
       test_bridge_graph_sorted_by_source_target},
      {"bridge_graph_drops_beyond_n_graph",
       test_bridge_graph_drops_beyond_n_graph},
      {"bridge_graph_applies_neigh_mask", test_bridge_graph_applies_neigh_mask},
      {"bridge_graph_empty_neighbor_list",
       test_bridge_graph_empty_neighbor_list},
      {"bridge_graph_reuses_buffers_across_calls",
       test_bridge_graph_reuses_buffers_across_calls},
      {"bridge_graph_rejects_n_graph_over_nall",
       test_bridge_graph_rejects_n_graph_over_nall},
      {"bridge_graph_ghost_closure", test_bridge_graph_ghost_closure},
      {"bridge_graph_order_shuffle_invariance",
       test_bridge_graph_order_shuffle_invariance},
      {"bridge_graph_sorts_only_when_contract_requires",
       test_bridge_graph_sorts_only_when_contract_requires},
      {"bridge_comm_pack_reverse_layout", test_bridge_comm_pack_reverse_layout},
      {"bridge_comm_unpack_accumulates", test_bridge_comm_unpack_accumulates},
      {"bridge_pad_edges", test_bridge_pad_edges},
      {"bridge_read_model_outputs_rejects_short_virial",
       test_bridge_read_model_outputs_rejects_short_virial},
      {"bridge_read_model_outputs_absent_virial_undefined",
       test_bridge_read_model_outputs_absent_virial_undefined},
      {"bridge_read_model_outputs_ignores_virial_when_unsupported",
       test_bridge_read_model_outputs_ignores_virial_when_unsupported},
      {"bridge_read_model_outputs_requires_atomic_energy",
       test_bridge_read_model_outputs_requires_atomic_energy},
      {"bridge_read_model_outputs_six_atoms_no_transpose",
       test_bridge_read_model_outputs_six_atoms_no_transpose},
      {"bridge_graphed_exec_cpu_only_ctor_throws",
       test_bridge_graphed_exec_cpu_only_ctor_throws},
      {"bridge_dynamic_exec_matches_reference",
       test_bridge_dynamic_exec_matches_reference},
      {"bridge_graphed_exec_matches_dynamic",
       test_bridge_graphed_exec_matches_dynamic},
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
