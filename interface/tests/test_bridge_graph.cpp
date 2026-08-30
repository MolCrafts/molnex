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
//     const int* ilist;            // (inum,) local indices of the centres
//     const int* numneigh;         // indexed by local i
//     const int* const* firstneigh;// firstneigh[i] -> (numneigh[i],)
//     int inum;
//     int neigh_mask;              // the facade fills this from LAMMPS
//     int nall() const;
//     int neighbor(int i, int jj) const;   // firstneigh[i][jj] & neigh_mask
//   };
//
//   #include "bridge/graph_bridge.h"
//
//   class GraphBridge {
//    public:
//     explicit GraphBridge(double cutoff);
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
//   struct ModelOutputs { at::Tensor energy, forces, virial; };
//   ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
//                                   bool supports_virial);
//   class Exec {
//    public:
//     virtual ~Exec();
//     virtual ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
//                              const at::Tensor& edge_index) = 0;
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
//   2. The output is SORTED by (source, target). molnex aggregation requires
//      that order — it is what the reference data pipeline's nonzero() yields —
//      and the raw neighbour-list order gives the same pairs in a different
//      order, i.e. wrong energies and forces with no error. Within this spec's
//      lifetime the sort is unconditional; lammps-iface-05-pbc demotes it to
//      the requires_sorted_edges contract flag.
//   3. Column 0 is the source (the centre atom out of ilist) and column 1 the
//      target (its neighbour), per the repo-wide Edge Convention.
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
  /// \param ilist       The centres, in the order LAMMPS would present them.
  /// \param neigh_mask  Mask applied to every `firstneigh` entry.
  explicit SyntheticSystem(std::vector<std::vector<int>> jlists = full_lists(),
                           std::vector<int> ilist = {0, 1, 2},
                           int neigh_mask = kNeighMask)
      : type_(kType),
        jlists_(std::move(jlists)),
        ilist_(std::move(ilist)),
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
  LammpsView view() {
    return LammpsView{x_rows_.data(),
                      f_rows_.data(),
                      type_.data(),
                      kNlocal,
                      kNghost,
                      ilist_.data(),
                      numneigh_.data(),
                      firstneigh_.data(),
                      static_cast<int>(ilist_.size()),
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
  GraphBridge graph(kCutoff);

  const at::Tensor edges = graph.build(sys.view(), kNall);
  expect_edges(edges, kEdgesAll, "edge convention");

  // Column 0 is the SOURCE: the centre atom, which under REQ_FULL is always
  // local. Column 1 is the TARGET, which may be a ghost. The last row is what
  // makes this more than a shape check — (2, 3) is the local centre 2 with the
  // ghost 3 as its neighbour, and a transposed convention would produce (3, 2),
  // a source that is not even in ilist.
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
  GraphBridge graph(kCutoff);

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
  // Both the centre order and every jlist are shuffled: the raw neighbour-list
  // order molnex would receive without the sort is a different permutation of
  // the same pairs, which produces wrong energies and forces with no error.
  SyntheticSystem sys({{3, 2, 1}, {2, 3, 0}, {3, 1, 0}, {}}, {2, 0, 1});
  GraphBridge graph(kCutoff);

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
  GraphBridge graph(kCutoff);

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
  SyntheticSystem sys(flagged, {0, 1, 2}, kNeighMask);

  // The test's own input really is flagged: without this the case could pass
  // against a mask that is never applied.
  expect_true(sys.raw_neighbor(0, 0) != 1,
              "the fixture must hand the bridge flagged neighbour entries");
  expect_int(sys.raw_neighbor(0, 0), 1 | kNeighFlag, "flagged entry value");

  GraphBridge graph(kCutoff);
  expect_edges(graph.build(sys.view(), kNall), kEdgesAll,
               "masked neighbour entries");
}

void test_bridge_graph_empty_neighbor_list() {
  GraphBridge graph(kCutoff);

  // (a) Every jlist empty — an isolated set of atoms, or a first step before
  // any pair comes within range. The reused host buffers are still empty here,
  // so this is the path where the migrated code handed a possibly-null
  // std::vector::data() to at::from_blob.
  SyntheticSystem bare({{}, {}, {}, {}});
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
  GraphBridge graph(kCutoff);

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
  GraphBridge graph(kCutoff);

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

void test_bridge_read_model_outputs_rejects_short_virial() {
  // A model that declares supports_virial but emits fewer than six components
  // used to reach the facade's tally, which then read six anyway. The check
  // belongs here: the virial either comes out well-formed or does not come out.
  const std::vector<at::Tensor> raw = {
      f64_of(1, -3.5),
      at::zeros({4, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(3, 1.0),
  };

  const std::string msg = expect_throws(
      [&] { (void)read_model_outputs(raw, /*supports_virial=*/true); },
      "virial with numel 3");
  expect_contains(msg, "3", "the diagnostic must state the observed numel");
  expect_contains(msg, "6", "the diagnostic must state the required numel");

  // A seven-element virial is just as wrong as a three-element one: the check
  // is an equality, not a lower bound (`>= 6` would let an off-by-one layout
  // through and silently read the wrong six components).
  const std::vector<at::Tensor> too_long = {raw[0], raw[1], f64_of(7, 1.0)};
  (void)expect_throws(
      [&] { (void)read_model_outputs(too_long, /*supports_virial=*/true); },
      "virial with numel 7");

  // Exactly six is accepted, so the case above is not passing by rejecting
  // every virial there is.
  const std::vector<at::Tensor> good = {raw[0], raw[1], f64_of(6, 2.0)};
  const ModelOutputs out = read_model_outputs(good, /*supports_virial=*/true);
  expect_true(out.virial.defined(), "a six-element virial must be accepted");
  expect_int(out.virial.numel(), 6, "the accepted virial keeps its six elements");
  expect_true(out.virial.scalar_type() == at::kDouble, "virial must be float64");
  expect_true(out.virial.device().is_cpu(), "virial must be a CPU tensor");
}

void test_bridge_read_model_outputs_absent_virial_undefined() {
  // supports_virial is true but the model emitted only energy and forces. That
  // is tolerated, exactly as the migrated `out.size() >= 3` was: the flag says
  // the facade may tally a virial, not that one is present on every call.
  const std::vector<at::Tensor> raw = {
      f64_of(1, 7.25),
      at::ones({3, 3}, at::TensorOptions().dtype(at::kDouble)),
  };

  const ModelOutputs out = read_model_outputs(raw, /*supports_virial=*/true);
  expect_true(!out.virial.defined(),
              "an absent virial must stay undefined, not become an empty tensor");
  expect_true(out.energy.defined(), "energy must still be read");
  expect_true(out.forces.defined(), "forces must still be read");
  expect_near(out.energy.item<double>(), 7.25, 0.0, "energy value");
  expect_true(out.forces.is_contiguous(), "forces must be contiguous");
}

void test_bridge_read_model_outputs_ignores_virial_when_unsupported() {
  // The model emitted a third tensor but the export never declared a virial.
  // Reading it would tally a quantity the contract does not claim — and a
  // malformed third output must not even be inspected, so this deliberately
  // carries a numel the supports_virial=true path would reject.
  const std::vector<at::Tensor> raw = {
      f64_of(1, -1.0),
      at::zeros({2, 3}, at::TensorOptions().dtype(at::kDouble)),
      f64_of(3, 99.0),
  };

  const ModelOutputs out = read_model_outputs(raw, /*supports_virial=*/false);
  expect_true(!out.virial.defined(),
              "supports_virial=false must leave virial undefined");
  expect_true(out.energy.defined(), "energy must still be read");
  expect_true(out.forces.defined(), "forces must still be read");
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
// These two need the exported graph-potential package from
// generate_fixtures.py, so CMake registers them with FIXTURES_REQUIRED and the
// MOLNEX_INTERFACE_FIXTURE_DIR environment variable — the same wiring as
// contract_reads_generated_v2_fixture.

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
  const fs::path ref_path = dir / "reference_graph_cpu.pt";
  const auto ref = load_tensor_dict(ref_path);

  const at::Tensor Z = reference_entry(ref, "Z", ref_path);
  const at::Tensor pos = reference_entry(ref, "pos", ref_path);
  const at::Tensor edge_index = reference_entry(ref, "edge_index", ref_path);

  ModelRunner runner((dir / "model_graph_cpu").string());
  expect_string(runner.device(), "cpu", "the graph fixture is a CPU export");

  DynamicExec exec(&runner, at::Device(at::kCPU), at::kDouble,
                   /*supports_virial=*/false);

  // The dtype name the facade prints in its logmesg. It lives here because the
  // execution site is the only reader of the model dtype; a copy on the facade
  // could only drift from this one.
  expect_string(exec.model_dtype_name(), "float64",
                "the fixture is an fp64 export");

  const ModelOutputs out = exec.run(Z, pos, edge_index);

  // fp64 in, fp64 through, fp64 out: the fixture's energy and forces are a
  // closed-form expression of its inputs, so 1e-12 is a real parity claim and
  // not a tolerance chosen to make a mismatch fit.
  expect_allclose_f64(out.energy, reference_entry(ref, "energy", ref_path), 1e-12,
                      "dynamic exec energy");
  expect_allclose_f64(out.forces, reference_entry(ref, "forces", ref_path), 1e-12,
                      "dynamic exec forces");

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
  const ModelOutputs again = exec.run(Z, pos, edge_index);
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
  if (!fs::exists(dir / "model_graph_cuda") ||
      !fs::exists(dir / "model_graph_fixed_cuda")) {
    // generate_fixtures.py emits the CUDA graph-potential exports only on a
    // host with a visible GPU; without them there is nothing to compare.
    std::fprintf(stderr, "[skip] cuda graph-potential fixture missing\n");
    return;
  }

  const fs::path ref_path = dir / "reference_graph_cuda.pt";
  const auto ref = load_tensor_dict(ref_path);
  const at::Tensor Z = reference_entry(ref, "Z", ref_path).to(at::kCPU);
  const at::Tensor pos = reference_entry(ref, "pos", ref_path).to(at::kCPU);
  const at::Tensor edge_index = reference_entry(ref, "edge_index", ref_path).to(at::kCPU);

  const int64_t n_fixed = Z.size(0);
  // Read, not recomputed: the fixed-shape package was COMPILED for this
  // capacity, so a literal here would be a second copy of a number that only
  // generate_fixtures.py can decide. It is deliberately larger than the edge
  // count — the padded tail is the part of the fixed-shape path a same-sized
  // export would never exercise.
  const int64_t e_max =
      reference_entry(ref, "e_max", ref_path).item<int64_t>();

  ModelRunner dyn_runner((dir / "model_graph_cuda").string());
  DynamicExec dynamic(&dyn_runner, at::Device(at::kCUDA), at::kDouble,
                      /*supports_virial=*/false);
  const ModelOutputs want = dynamic.run(Z, pos, edge_index);

  ModelRunner graph_runner((dir / "model_graph_fixed_cuda").string());
  GraphedExec graphed(&graph_runner, at::Device(at::kCUDA), at::kDouble,
                      /*supports_virial=*/false, n_fixed, e_max);
  const ModelOutputs got = graphed.run(Z, pos, edge_index);

  // rel 1e-6 is this chain's parity judgement; energies here are O(1), so an
  // absolute tolerance of the same size is the stricter reading of it.
  expect_allclose_f64(got.energy, want.energy, 1e-6, "graphed vs dynamic energy");
  expect_allclose_f64(got.forces, want.forces, 1e-6, "graphed vs dynamic forces");
  expect_true(got.forces.device().is_cpu(), "graphed forces must come back on the CPU");
  expect_true(got.forces.scalar_type() == at::kDouble, "graphed forces must be float64");
  expect_true(got.forces.is_contiguous(), "graphed forces must be contiguous");
  expect_true(!got.virial.defined(), "supports_virial=false leaves virial undefined");

  // The replay-aliasing hazard, stated as an assertion: run_graphed hands back
  // tensors at FIXED addresses that the next replay overwrites in place. A
  // GraphedExec returning those directly would silently rewrite the caller's
  // previous-step forces.
  const at::Tensor first_forces = got.forces;
  const ModelOutputs second = graphed.run(Z, pos, edge_index);
  expect_true(second.forces.data_ptr() != first_forces.data_ptr(),
              "a replay must not hand back the previous call's storage");
  expect_allclose_f64(first_forces, want.forces, 1e-6,
                      "the first result must survive the second replay intact");

  // Beyond e_max the fixed-shape path refuses rather than truncating.
  std::vector<EdgePair> too_many;
  for (int64_t k = 0; k < e_max + 1; ++k) too_many.emplace_back(0, 0);
  (void)expect_throws(
      [&] { (void)graphed.run(Z, pos, make_edges(too_many)); },
      "graphed run past e_max");

  // And a node count other than the exported fixed N is refused too.
  (void)expect_throws(
      [&] {
        (void)graphed.run(Z.narrow(0, 0, n_fixed - 1),
                          pos.narrow(0, 0, n_fixed - 1), make_edges({}));
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
      {"bridge_comm_pack_reverse_layout", test_bridge_comm_pack_reverse_layout},
      {"bridge_comm_unpack_accumulates", test_bridge_comm_unpack_accumulates},
      {"bridge_pad_edges", test_bridge_pad_edges},
      {"bridge_read_model_outputs_rejects_short_virial",
       test_bridge_read_model_outputs_rejects_short_virial},
      {"bridge_read_model_outputs_absent_virial_undefined",
       test_bridge_read_model_outputs_absent_virial_undefined},
      {"bridge_read_model_outputs_ignores_virial_when_unsupported",
       test_bridge_read_model_outputs_ignores_virial_when_unsupported},
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
