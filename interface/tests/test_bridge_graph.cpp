// C++ tests for GraphBridge and the reverse-comm free functions, over a
// synthetic LammpsView. No LAMMPS build, no package.
//
// Invariants:
//
//   1. The cutoff refilter drops pairs LAMMPS kept because it builds the
//      neighbour list at cutoff + skin.
//   2. Edge order is the neighbour-list order. The builder does not sort.
//   3. Column 0 is the source and column 1 the target. Walking ghost centres
//      closes the edge set.
//   4. n_graph is validated before any read of view.x.
//   5. An empty edge set is a well-formed (0, 2) int64 tensor.
//   6. unpack_reverse_forces accumulates.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/comm_reverse.h"
#include "bridge/graph_bridge.h"
#include "bridge/lammps_view.h"
#include "expect.h"

using molnex::lammps_bridge::GraphBridge;
using molnex::lammps_bridge::LammpsView;
using molnex::lammps_bridge::pack_reverse_forces;
using molnex::lammps_bridge::unpack_reverse_forces;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- the synthetic system --------------------------------------------------

// Three local atoms plus one ghost. The ghost is what makes the n_graph cases
// mean something: with n_graph == nlocal it must vanish from the graph, with
// n_graph == nall it must appear as a target (never as a source — REQ_FULL
// lists only local atoms in ilist). The pair style always passes nall().
constexpr int kNlocal = 3;
constexpr int kNghost = 1;
constexpr int kNall = kNlocal + kNghost;

// Model cutoff. The neighbour lists below are deliberately built as if by
// LAMMPS at cutoff + skin, i.e. every atom sees every other one.
constexpr double kCutoff = 2.0;  // cutsq = 4.0

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
const std::vector<EdgePair> kEdgesAll = {
    {0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}};

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

/// The storage a synthetic LammpsView borrows: atom arrays plus neighbour
/// spans.
///
/// LAMMPS owns `atom->x` / `atom->f` as `double**` (an array of row pointers)
/// and `NeighList::firstneigh` as `int**` (one jlist per local atom). This
/// fixture reproduces exactly those shapes out of its own vectors, which is
/// what makes the bridge drivable with no LAMMPS build.
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
      numneigh_.push_back(
          static_cast<int>(jlists_[static_cast<std::size_t>(i)].size()));
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

  /// The raw, still-flagged neighbour entry, for asserting the test's own
  /// input.
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
  expect_true(edges.defined(), tag + ": pair_index must be a defined tensor");
  expect_true(edges.scalar_type() == at::kLong,
              tag + ": pair_index must be int64");
  expect_true(edges.device().is_cpu(),
              tag +
                  ": pair_index must stay on the CPU — the device transfer "
                  "belongs to the execution site");
  expect_int(edges.dim(), 2, tag + ": pair_index must be 2-D");
  expect_int(edges.size(0), static_cast<long long>(want.size()),
             tag + ": edge count");
  expect_int(edges.size(1), 2, tag + ": pair_index must be (ne, 2)");

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

  // Column 0 is the SOURCE: the centre atom out of ilist. This fixture declares
  // gnum == 0, so every centre it offers is local and the bound below is
  // nlocal; under a ghost request (bridge_graph_ghost_closure) the source
  // domain widens to local u ghost, which is what closes the edge set
  // symmetrically. Column 1 is the TARGET, which may be a ghost either way. The
  // last row is what makes this more than a shape check — (2, 3) is the local
  // centre 2 with the ghost 3 as its neighbour, and a transposed convention
  // would produce (3, 2), a source that is not even in ilist here.
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

void test_bridge_graph_drops_beyond_n_graph() {
  SyntheticSystem sys;
  GraphBridge graph(kCutoff);

  // n_graph == nlocal drops the ghost. The pair style passes nall(); this
  // locks the filter the builder still owns. The ghost 3 is inside the
  // cutoff of centre 2, so only the n_graph test can remove it.
  const at::Tensor edges = graph.build(sys.view(), kNlocal);
  expect_edges(edges, kEdgesLocal, "n_graph == nlocal");

  const auto ea = edges.accessor<int64_t, 2>();
  for (int64_t k = 0; k < edges.size(0); ++k) {
    expect_true(ea[k][1] < kNlocal,
                "no target may be a ghost when n_graph == nlocal");
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
  // Spelled out rather than braced: with `gnum` and `neigh_mask` both
  // defaultable, a bare `{{}, {}, {}, {}}` is equally readable as four empty
  // jlists or as four default-initialised arguments, and the compiler says so.
  SyntheticSystem bare(std::vector<std::vector<int>>{{}, {}, {}, {}});
  const at::Tensor none = graph.build(bare.view(), kNall);
  expect_edges(none, {}, "empty neighbour lists");
  expect_int(none.numel(), 0, "empty pair_index must hold no elements");

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
  // the caller's pair_index would silently change under it mid-step, and a
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

/// One owned atom and two of its own periodic images in a box under twice
/// the cutoff: the folded graph is two self-pairs (0, 0) with opposite shifts.
/// The image translation is deliberately not axis-aligned — the shift is read
/// off the coordinates, so a triclinic image needs nothing extra.
void test_bridge_graph_folds_self_images() {
  const std::vector<std::array<double, 3>> coords = {
      {0.25, 0.5, 0.0}, {1.75, 1.0, 0.0}, {-1.25, 0.0, 0.0}};
  std::vector<const double*> x_rows;
  for (const auto& c : coords) x_rows.push_back(c.data());
  std::vector<std::array<double, 3>> f(3, {0.0, 0.0, 0.0});
  std::vector<double*> f_rows;
  for (auto& r : f) f_rows.push_back(r.data());
  const std::vector<int> type = {1, 1, 1};
  const std::vector<int> ilist = {0};
  const std::vector<int> neigh0 = {1, 2};
  const std::vector<int> numneigh = {2, 0, 0};
  const std::vector<const int*> firstneigh = {neigh0.data(), nullptr, nullptr};
  const std::vector<int> owner = {0, 0, 0};
  const LammpsView view{x_rows.data(),   f_rows.data(),     type.data(),
                        /*nlocal=*/1,    /*nghost=*/2,      ilist.data(),
                        numneigh.data(), firstneigh.data(), /*inum=*/1,
                        /*gnum=*/0,      0x3FFFFFFF,        owner.data()};

  GraphBridge bridge(/*cutoff=*/2.0);
  const auto graph = bridge.build_folded(view);
  expect_int(graph.pair_index.size(0), 2, "both images are neighbours");
  const auto ea = graph.pair_index.accessor<int64_t, 2>();
  const auto sa = graph.shifts.accessor<double, 2>();
  for (int k = 0; k < 2; ++k) {
    expect_int(ea[k][0], 0, "source is the owned atom");
    expect_int(ea[k][1], 0, "an image folds onto its own original");
  }
  expect_near(sa[0][0], 1.5, 1e-15, "shift x of image 1");
  expect_near(sa[0][1], 0.5, 1e-15, "shift y of image 1 (not axis-aligned)");
  expect_near(sa[1][0], -1.5, 1e-15, "shift x of image 2");
  expect_near(sa[1][1], -0.5, 1e-15, "shift y of image 2");

  // Without an owner table a ghost neighbour cannot be folded.
  LammpsView no_owner = view;
  no_owner.owner = nullptr;
  (void)expect_throws([&] { (void)bridge.build_folded(no_owner); },
                      "a ghost with no owner table");
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

  // The boundary itself is valid: n_graph == nall is the count the pair
  // style passes.
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
void expect_same_edge_set(const at::Tensor& got,
                          const std::vector<EdgePair>& want,
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
    const std::size_t sink =
        static_cast<std::size_t>(to_target ? e.second : e.first);
    const std::size_t src =
        static_cast<std::size_t>(to_target ? e.first : e.second);
    h1[sink] += h0[src];
  }
  std::vector<double> h2(n, 0.0);
  for (const EdgePair& e : edges) {
    const std::size_t sink =
        static_cast<std::size_t>(to_target ? e.second : e.first);
    const std::size_t src =
        static_cast<std::size_t>(to_target ? e.first : e.second);
    h2[sink] += h1[src];
  }
  return h2;
}

void test_bridge_graph_ghost_closure() {
  // All five atoms are centres: two local, then three ghosts. This is the
  // full list with ghosts the pair style requests. Walking the ghost
  // centres is what closes the edge set.
  ClosureSystem sys({0, 1, 2, 3, 4}, /*gnum=*/3);
  GraphBridge graph(kClosureCutoff);

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
                    std::to_string(want.second) +
                    ") of a local atom is missing");
  }

  // (2) The direction-independence claim, executed. Seeds are powers of two, so
  // every subset sum is unique and a single missing edge changes the result
  // exactly — there is no coincidental agreement available.
  const std::vector<double> h0 = {1.0, 2.0, 4.0, 8.0, 16.0};
  for (const bool to_target : {true, false}) {
    const std::string dir =
        to_target ? "target-aggregating (MACE)" : "source-aggregating (PiNet)";
    const std::vector<double> got = two_layer_aggregate(built, h0, to_target);
    const std::vector<double> want =
        two_layer_aggregate(reference, h0, to_target);
    for (int i = 0; i < kClosureNlocal; ++i) {
      expect_near(got[static_cast<std::size_t>(i)],
                  want[static_cast<std::size_t>(i)], 0.0,
                  dir + ": local atom " + std::to_string(i) +
                      " must reproduce the full-enumeration value");
    }
  }

  // (3) The negative control, and the reason this case is evidence rather than
  // decoration: the SAME builder over the SAME atoms with only the local
  // centres walked (gnum == 0, ilist truncated to the locals) disagrees with
  // the reference in BOTH directions. Without it, (2) would pass against a
  // builder that never looked at a ghost, since two identical computations over
  // one edge list always agree.
  ClosureSystem locals_only({0, 1}, /*gnum=*/0);
  const std::vector<EdgePair> narrow =
      edge_set(graph.build(locals_only.view(), kClosureNall));
  expect_true(narrow.size() < built.size(),
              "a local-only walk must produce strictly fewer edges");
  for (const bool to_target : {true, false}) {
    const std::vector<double> got = two_layer_aggregate(narrow, h0, to_target);
    const std::vector<double> want =
        two_layer_aggregate(reference, h0, to_target);
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
  // The builder does not sort: it hands back whatever order the neighbour
  // list had. Two presentations of the same system: the same
  // centres and the same pairs, permuted.
  ClosureSystem plain({0, 1, 2, 3, 4}, /*gnum=*/3);
  ClosureSystem shuffled({3, 1, 4, 0, 2}, /*gnum=*/3,
                         {{2, 3, 1}, {4, 2, 3, 0}, {1, 0}, {0, 4, 1}, {1, 3}});
  GraphBridge graph(kClosureCutoff);

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

// ---- dispatcher -----------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"bridge_graph_edge_convention", test_bridge_graph_edge_convention},
      {"bridge_graph_cutoff_refilter", test_bridge_graph_cutoff_refilter},
      {"bridge_graph_drops_beyond_n_graph",
       test_bridge_graph_drops_beyond_n_graph},
      {"bridge_graph_applies_neigh_mask", test_bridge_graph_applies_neigh_mask},
      {"bridge_graph_empty_neighbor_list",
       test_bridge_graph_empty_neighbor_list},
      {"bridge_graph_reuses_buffers_across_calls",
       test_bridge_graph_reuses_buffers_across_calls},
      {"bridge_graph_rejects_n_graph_over_nall",
       test_bridge_graph_rejects_n_graph_over_nall},
      {"bridge_graph_folds_self_images", test_bridge_graph_folds_self_images},
      {"bridge_graph_ghost_closure", test_bridge_graph_ghost_closure},
      {"bridge_graph_order_shuffle_invariance",
       test_bridge_graph_order_shuffle_invariance},
      {"bridge_comm_pack_reverse_layout", test_bridge_comm_pack_reverse_layout},
      {"bridge_comm_unpack_accumulates", test_bridge_comm_unpack_accumulates},
  };
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <test_name>\n", argv[0]);
    std::fprintf(stderr, "available:\n");
    for (const auto& [k, _] : registry())
      std::fprintf(stderr, "  %s\n", k.c_str());
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
