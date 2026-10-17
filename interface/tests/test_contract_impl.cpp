// C++ tests for CutoffGraphImpl, the one pair-style implementation.
//
// The execution is injected over a stub ModelArtifact, so these cases run with
// no LAMMPS build, no CUDA and no package. CutoffGraphImpl is constructed
// directly.
//
// Invariants:
//
//   1. NeighborNeeds always requests a full neighbour list and ghosts. The
//      cutoff comes from the metadata.
//   2. compute() composes stage_nodes, GraphBridge, AtomEnergyExec and
//      ForceScatter, and applies the energy factor once. A model length unit
//      other than angstrom is refused at construction.
//   3. n_graph is nall() (locals and ghosts). compute() requires init() first:
//      newton on is checked there and re-checked in compute().
//   4. The partition energy sums local atom energies only.
//   5. needs_pair_reverse_comm is constantly false.
//   6. pair_coeff tokens are parsed by full consumption.
//   7. special_bonds that drop 1-2 / 1-3 / 1-4 pairs are refused on a molecular
//      system.
//   8. newton off is refused. The ghost cutoff must be >= T * cutoff + skin.
//      An unset ghost cutoff (0) means cutoff + skin, and that reading lives
//      only in init().

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/comm_reverse.h"
#include "bridge/lammps_view.h"
#include "expect.h"
#include "impls/contract_impl.h"
#include "impls/cutoff_graph_impl.h"
#include "molnex/runtime/metadata.h"
#include "molnex/runtime/model_artifact.h"
#include "molnex/runtime/atom_energy_exec.h"

using molnex::lammps::ContractImpl;
using molnex::lammps::CutoffGraphImpl;
using molnex::lammps::EvalResult;
using molnex::lammps::GraphMode;
using molnex::lammps::LammpsRunEnv;
using molnex::lammps::NeighborNeeds;
using molnex::lammps_bridge::LammpsView;
using molnex::lammps_bridge::pack_reverse_forces;
using molnex::lammps_bridge::unpack_reverse_forces;
using molnex::runtime::Metadata;
using molnex::runtime::ModelArtifact;
using molnex::runtime::AtomEnergyExec;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- the model's metadata ---------------------------------------------------

/// The eV -> kcal/mol factor the facade would hand the impl under `units real`.
/// Hard-coded: a fixture may not be computed by the code under test.
constexpr double kEnergyConv = 23.060549;

/// An eV model with a 2 A cutoff (matching the synthetic system below) and
/// T == 1: with the default ghost cutoff (cutoff + skin) that is exactly the
/// fat-ghost threshold, so every case that calls init() without caring about
/// the shell passes it. The deeper model lives in deep_metadata().
Metadata model_metadata() {
  Metadata meta;
  meta.format = "aoti";
  meta.cutoff = 2.0;
  meta.species = {1, 6, 8};
  meta.dtype = at::kDouble;
  meta.receptive_field_layers = 1;
  meta.energy_unit = "eV";
  meta.length_unit = "angstrom";
  return meta;
}

// ---- the synthetic system --------------------------------------------------

// Three local atoms plus one ghost slot. A caller that passes nghost == 0 makes
// nall() equal nlocal, so the fourth atom is storage the view does not expose.
constexpr int kNlocal = 3;
constexpr int kNall = 4;

/// Model cutoff in angstrom; equals model_metadata().cutoff.
constexpr double kCutoff = 2.0;

/// LAMMPS' own NEIGHMASK, travelling as data through LammpsView.
constexpr int kNeighMask = 0x3FFFFFFF;

/// Per-atom LAMMPS types, 1-based: three types over four atoms.
const std::vector<int> kType = {1, 2, 2, 3};

/// pair_coeff residual tokens for those three types, mapping onto the meta's
/// species list [1, 6, 8]. Atom Z therefore reads (1, 6, 6, 8).
const std::vector<std::string> kCoeffTokens = {"1", "6", "8"};
const std::vector<int64_t> kAtomZ = {1, 6, 6, 8};

// Positions in angstrom. Squared distances against cutsq = 4.0 (identical to
// the 03-graph fixture, so the edge sets below are the ones that spec pinned):
//     0-1  1.00 in | 0-2  5.64 out | 0-3 10.64 out
//     1-2  2.96 in | 1-3  6.76 out | 2-3  1.00 in
constexpr double kCoord[kNall][3] = {
    {0.0, 0.0, 0.0},
    {0.6, 0.8, 0.0},
    {2.0, 0.8, 1.0},
    {3.0, 0.8, 1.0},
};

/// Forces already standing in atom->f when the pair style runs — NON-ZERO, so a
/// scatter that overwrites instead of accumulating cannot pass.
constexpr double kInitialForce[kNall][3] = {
    {1.5, -2.25, 0.75},
    {-0.5, 3.125, -1.0},
    {0.0, 0.5, -0.125},
    {7.0, 7.0, 7.0},
};

/// Forces the stub artifact returns, in MODEL units. Every entry is a power of
/// two so `kEnergyConv * entry` is an exact binary scaling and the expectations
/// below are true decimal literals rather than rounded ones.
constexpr double kModelForce[kNall][3] = {
    {0.5, -0.25, 0.125},
    {1.0, 2.0, -4.0},
    {-0.0625, 8.0, 0.03125},
    {16.0, -0.5, 0.25},
};

/// atom->f after the impl has run: kInitialForce + kEnergyConv * kModelForce,
/// hand-computed. E.g. row 0 column 0: 1.5 + 23.060549 * 0.5 = 13.0302745.
constexpr double kScatteredForce[kNall][3] = {
    {13.0302745, -8.01513725, 3.632568625},
    {22.560549, 49.246098, -93.242196},
    {-1.4412843125, 184.984392, 0.59564215625},
    {375.968784, -4.5302745, 12.76513725},
};

/// Scalar energy the stub artifact returns, in MODEL units (eV).
///
/// DELIBERATELY UNRELATED to the atom energies below. A real model's scalar is
/// its masked atom-energy sum, but this stub is a probe, not a model: making
/// the two disagree is what lets these cases tell WHICH output the impl
/// tallied. An impl that adds `out.energy` to eng_vdwl gets 184.48 where the
/// partition energy is 34.59, and every energy assertion below fails.
/// (A real export cannot ship this inconsistency: molix.engine.export checks
/// decomposability eagerly on the example system before tracing.)
constexpr double kStubEnergy = 8.0;

/// The strain derivative the stub returns, in MODEL units: symmetric, as the
/// contract's, with six distinct entries so the Voigt order is checked.
constexpr double kModelVirial[3][3] = {
    {1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}};

/// Per-atom energies the stub returns, in MODEL units (eV), one per graph node.
///
/// Distinct rather than constant, so `e_atom` cannot pass by carrying the right
/// length with the wrong contents, and so the local sum differs from the
/// all-atoms sum by an amount no rounding could produce.
constexpr double kModelAtomEnergy[kNall] = {0.25, 0.5, 0.75, 1.0};

/// The three LOCAL atom energies in LAMMPS `real` units, by hand:
/// 0.25 * 23.060549, 0.5 * 23.060549, 0.75 * 23.060549.
constexpr double kExpectedEAtom[kNlocal] = {5.76513725, 11.5302745,
                                            17.29541175};

/// Partition energy: the sum of the three LOCAL atom energies, converted.
/// (0.25 + 0.5 + 0.75) * 23.060549 = 1.5 * 23.060549.
constexpr double kExpectedETotal = 34.5908235;

/// What summing over local + ghost would give — 2.5 * 23.060549. This is the
/// D1 bug's value, written down so the assertion against it is explicit rather
/// than implied by the absence of a number.
constexpr double kGhostInclusiveETotal = 57.6513725;

using EdgePair = std::pair<int, int>;

/// Every surviving directed pair with n_graph == 4 (the ghost is a node).
const std::vector<EdgePair> kEdgesAll = {
    {0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}};
/// The same with n_graph == 3: the ghost is not a node, so (2,3) goes.
const std::vector<EdgePair> kEdgesLocal = {{0, 1}, {1, 0}, {1, 2}, {2, 1}};

/// Full neighbour lists as LAMMPS builds them at cutoff + skin: every atom
/// lists every other one, so the model cutoff refilter is doing real work.
std::vector<std::vector<int>> full_lists() {
  return {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {}};
}

/// The storage a synthetic LammpsView borrows: atom arrays plus neighbour
/// spans.
///
/// Non-copyable: the pointer vectors point into this object's own storage.
class SyntheticSystem {
 public:
  /// \param nghost  Ghost count the view reports. Storage is always four atoms;
  ///                nghost == 0 means the fourth is not addressable: nall()
  ///                equals nlocal, and the fourth atom is unused storage.
  /// \param gnum    Ghost centres presented after the local ones in `ilist`.
  ///                Zero here even when nghost is one: the ghost is a graph
  ///                NODE and a neighbour, but this fixture does not give it a
  ///                neighbour list of its own. That is the impl's business to
  ///                pass through, not to invent — the ghost-closure walk itself
  ///                is locked in test_bridge_graph.cpp, where the geometry is
  ///                built for it.
  explicit SyntheticSystem(int nghost, int gnum = 0)
      : type_(kType),
        jlists_(full_lists()),
        ilist_({0, 1, 2}),
        nghost_(nghost),
        gnum_(gnum) {
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
  LammpsView view() {
    return LammpsView{x_rows_.data(),
                      f_rows_.data(),
                      type_.data(),
                      kNlocal,
                      nghost_,
                      ilist_.data(),
                      numneigh_.data(),
                      firstneigh_.data(),
                      static_cast<int>(ilist_.size()) - gnum_,
                      gnum_,
                      kNeighMask,
                      owner_.empty() ? nullptr : owner_.data()};
  }

  /// Declare which owned atom each atom copies (folded mode), `(nall,)`.
  void set_owner(std::vector<int> owner) { owner_ = std::move(owner); }

  /// Current value of f[i][k] — what the component under test wrote.
  double force(int i, int k) const {
    return forces_[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
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
  int nghost_;
  int gnum_;
  std::vector<int> owner_;
};

// ---- the injected stub artifact ---------------------------------------------

/// A `ModelArtifact` that answers with closed-form outputs and records what it
/// was fed. No package format at all: the execution names none, so a test
/// artifact attaches exactly where an AOTI one does.
class StubArtifact : public ModelArtifact {
 public:
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) override {
    calls += 1;
    Z_seen = inputs[0].clone();
    pos_seen = inputs[1].clone();
    edges_seen = inputs[2].clone();
    shifts_seen = inputs[3].clone();
    local_mask_seen = inputs[4].clone();

    const auto opts = at::TensorOptions().dtype(at::kDouble);
    const int64_t n = Z_seen.size(0);
    at::Tensor forces = at::zeros({n, 3}, opts);
    auto fa = forces.accessor<double, 2>();
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t k = 0; k < 3; ++k) {
        fa[i][k] = kModelForce[i][k];
      }
    }

    // One per GRAPH NODE — locals and ghosts alike. Which of them belong to the
    // partition energy is the impl's decision, and the whole subject of
    // impl_energy_sums_locals_only; a stub that returned only the local ones
    // would be answering that question on the impl's behalf.
    at::Tensor atom_energy = at::zeros({n}, opts);
    auto ea = atom_energy.accessor<double, 1>();
    for (int64_t i = 0; i < n; ++i) ea[i] = kModelAtomEnergy[i];
    at::Tensor virial = at::zeros({3, 3}, opts);
    auto va = virial.accessor<double, 2>();
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) va[a][b] = kModelVirial[a][b];
    return {at::full({1}, kStubEnergy, opts), forces, atom_energy, virial};
  }

  at::Device device() const override { return at::Device(at::kCPU); }

  /// How many times the impl called through, and with what.
  int calls = 0;
  at::Tensor Z_seen;
  at::Tensor pos_seen;
  at::Tensor edges_seen;
  at::Tensor shifts_seen;
  at::Tensor local_mask_seen;
};

/// A CutoffGraphImpl over `meta`, executing through a stub artifact whose probe
/// is handed back.
///
/// CutoffGraphImpl's PUBLIC constructor is the only construction path any
/// impl_* case uses. `stub` is declared first so it outlives `impl`, which
/// borrows it.
struct ImplUnderTest {
  std::unique_ptr<StubArtifact> stub;
  std::unique_ptr<CutoffGraphImpl> impl;
  StubArtifact* exec;  ///< The probe; owned by `stub`.
};

ImplUnderTest impl_over(const Metadata& meta,
                        GraphMode mode = GraphMode::kGhost) {
  auto stub = std::make_unique<StubArtifact>();
  StubArtifact* probe = stub.get();
  auto impl = std::make_unique<CutoffGraphImpl>(
      meta, kEnergyConv, AtomEnergyExec(*stub, meta.dtype), mode);
  return ImplUnderTest{std::move(stub), std::move(impl), probe};
}

/// A three-layer model: T == 3, so an atom's energy depends on everything within
/// 3 * cutoff and the default ghost cutoff (cutoff + skin) is too small.
Metadata deep_metadata() {
  Metadata meta = model_metadata();
  meta.receptive_field_layers = 3;
  return meta;
}

/// LAMMPS' special-bond arrays, index 0 unused, [1..3] the 1-2 / 1-3 / 1-4
/// scale factors. `1.0 1.0 1.0` is the only setting that leaves the neighbour
/// list intact; LAMMPS' DEFAULT is 0.0 0.0 0.0, which deletes those pairs.
constexpr double kSpecialKept[4] = {1.0, 1.0, 1.0, 1.0};
constexpr double kSpecialDefault[4] = {1.0, 0.0, 0.0, 0.0};

/// `neighbor->skin` as LAMMPS defaults it for real / metal units, in angstrom.
constexpr double kSkin = 2.0;

/// `comm->cutghostuser` when the input script never ran `comm_modify cutoff`.
///
/// LAMMPS' own value for "unset" is a literal 0.0, and it is passed through
/// RAW: what zero MEANS — that the effective ghost cutoff is cutoff + skin — is
/// interpreted inside CutoffGraphImpl::init and nowhere else, which is why this
/// fixture can state the LAMMPS value without also stating the interpretation.
constexpr double kGhostCutoffUnset = 0.0;

/// A run environment with the specials that keep every bonded neighbour.
LammpsRunEnv env_with(int newton_pair, int nprocs, int molecular,
                      const double* special_lj = kSpecialKept,
                      const double* special_coul = kSpecialKept,
                      double ghost_cutoff = kGhostCutoffUnset,
                      double skin = kSkin) {
  return LammpsRunEnv{newton_pair,  nprocs,       molecular, special_lj,
                      special_coul, ghost_cutoff, skin};
}

/// Newton on, several ranks. init() refuses newton off: the integrator's
/// global reverse comm only runs under newton, and it is the sole reducer of
/// ghost forces.
LammpsRunEnv pbc_env() {
  return env_with(/*newton_pair=*/1, /*nprocs=*/2, /*molecular=*/0);
}

/// Require the recorded edge tensor to be exactly `want`.
void expect_edges(const at::Tensor& edges, const std::vector<EdgePair>& want,
                  const std::string& tag) {
  expect_true(edges.defined(),
              tag + ": the impl must have built an pair_index");
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

/// Require atom `i` to carry the scattered force, or its untouched initial one.
void expect_force_row(const SyntheticSystem& sys, int i, bool scattered,
                      const std::string& tag) {
  for (int k = 0; k < 3; ++k) {
    const double want = scattered ? kScatteredForce[i][k] : kInitialForce[i][k];
    expect_near(sys.force(i, k), want, 1e-12,
                tag + ": f[" + std::to_string(i) + "][" + std::to_string(k) +
                    (scattered ? "] += force_conv * model force"
                               : "] must be untouched"));
  }
}

// ---- needs -----------------------------------------------------------------

void test_impl_needs_matches_legacy_request() {
  const ImplUnderTest under = impl_over(model_metadata());
  const NeighborNeeds needs = under.impl->needs();

  // Value for value, the request pair_molnex.cpp made before the refactor:
  // REQ_FULL (cpp:265), comm_reverse / comm_reverse_off = 3 (cpp:56-57) and the
  // cutoff init_one() reported (cpp:256).
  expect_true(needs.full, "a many-body model needs the FULL neighbour list");
  expect_near(needs.cutoff, kCutoff, 1e-12, "needs.cutoff is the meta cutoff");
  expect_int(needs.comm_reverse, 3, "reverse comm packs fx, fy, fz");
  expect_int(needs.comm_reverse_off, 3, "newton-off variant packs the same 3");

  // Ghost centres are always requested so the edge set can close. A system
  // with no ghosts makes the request a no-op.
  expect_true(needs.ghost, "the pair style always requests ghost neighbours");

  // The cutoff is READ from the metadata, not a constant of the impl: a second
  // export with a different cutoff must report it.
  Metadata wide = model_metadata();
  wide.cutoff = 5.5;
  expect_near(impl_over(wide).impl->needs().cutoff, 5.5, 1e-12,
              "needs.cutoff follows the metadata");

  // needs() carries no predicate: the single-rank constraint is executed by
  // init(), never handed to the facade to interpret (ac-002). That is a
  // compile-time fact about the struct — there is no field to read here.
}

// ---- compute ---------------------------------------------------------------

void test_impl_compute_composes_pipeline() {
  SyntheticSystem sys(/*nghost=*/0);
  ImplUnderTest under = impl_over(model_metadata());
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0));

  const EvalResult result = under.impl->compute(sys.view());

  // (1) stage_nodes ran: Z came through the SpeciesMap the tokens built, and
  // pos is the fp64 coordinates as LAMMPS holds them.
  expect_int(under.exec->calls, 1, "the impl must run the model exactly once");
  expect_int(under.exec->Z_seen.size(0), kNlocal, "one Z per graph node");
  const auto za = under.exec->Z_seen.accessor<int64_t, 1>();
  for (int i = 0; i < kNlocal; ++i) {
    expect_int(za[i], kAtomZ[static_cast<std::size_t>(i)],
               "Z[" + std::to_string(i) + "] is the mapped atomic number");
  }
  expect_true(under.exec->pos_seen.scalar_type() == at::kDouble,
              "positions reach the model in fp64; the model-dtype cast is the "
              "execution site's own business");
  expect_near(under.exec->pos_seen.accessor<double, 2>()[1][0], kCoord[1][0],
              0.0, "pos[1][0] is LAMMPS' own coordinate");

  // (2) GraphBridge ran, with the cutoff refilter. The builder keeps
  // neighbour-list order, and this fixture's lists are already in
  // (source, target) order. A shuffled list is covered in
  // test_bridge_graph.cpp.
  expect_edges(under.exec->edges_seen, kEdgesLocal, "composed pair_index");

  // (3) The energy is the sum of the LOCAL atom energies, times the factor,
  // ONCE: (0.25 + 0.5 + 0.75) eV in a `real` run is 1.5 * 23.060549
  // = 34.5908235 kcal/mol. A missing conversion gives 1.5 and a doubled one
  // 797.7..., so neither can pass — and neither can tallying the stub's
  // scalar 8.0 eV.
  expect_near(result.e_total, kExpectedETotal, 1e-12,
              "e_total is the local atom-energy sum in LAMMPS units");

  // (4) ForceScatter ran: atom->f ACCUMULATED the converted model forces, from
  // a non-zero starting value. This is the asymmetry EvalResult's docstring
  // records — the forces are already in atom->f when compute() returns, because
  // that is the array comm->reverse_comm reads. There is no `result.forces` to
  // read here, and that is a COMPILE-TIME fact: EvalResult declares no such
  // member, so `result.forces` would not build.
  for (int i = 0; i < kNlocal; ++i) {
    expect_force_row(sys, i, /*scattered=*/true, "composed scatter");
  }
  // The fourth row is not a graph node here (nghost == 0), so nothing wrote it.
  expect_force_row(sys, 3, /*scattered=*/false, "beyond n_graph");

  // (5) The reduction-ownership seam: the integrator's global reverse comm owns
  // the reduction, so the pair style is never asked to issue one of its own.
  expect_true(
      !result.needs_pair_reverse_comm,
      "the cutoff-graph contract never asks the facade to reverse comm");
}

void test_impl_compute_n_graph_with_ghost() {
  // One ghost: n_graph is nall(), so the ghost is a graph node, gains an edge
  // and receives a force.
  SyntheticSystem sys(/*nghost=*/1);
  ImplUnderTest under = impl_over(model_metadata());
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(pbc_env());

  const EvalResult result = under.impl->compute(sys.view());

  expect_int(under.exec->Z_seen.size(0), kNall,
             "a periodic export stages local + ghost nodes");
  expect_int(under.exec->Z_seen.accessor<int64_t, 1>()[3], kAtomZ[3],
             "the ghost's Z is mapped like any other atom");
  expect_edges(under.exec->edges_seen, kEdgesAll, "ghost-inclusive pair_index");

  // EVERY row, ghost included, carries the converted model force — and that is
  // not an accident of the stub. The partition energy sums local atoms only,
  // but it is differentiated with respect to ALL coordinates, so a ghost's
  // force is the cross-term of its local neighbours' atom energies. Dropping
  // it would lose a real contribution; the integrator's global reverse comm is
  // what carries it home to the owning rank.
  for (int i = 0; i < kNall; ++i) {
    expect_force_row(sys, i, /*scattered=*/true, "ghost-inclusive scatter");
  }

  // The energy, however, does NOT grow with the ghost: it stays the local
  // three-atom sum even though four atoms were evaluated.
  expect_near(result.e_total, kExpectedETotal, 1e-12,
              "the partition energy is unchanged by the presence of a ghost");
}

void test_impl_reports_no_pair_reverse_comm() {
  // THE ANSWER IS NOW CONSTANT, and the constant is the whole point: ghost
  // forces are scattered into atom->f like any other, and Verlet::run's global
  // reverse comm — which runs whenever force->newton is set — reduces them onto
  // their owners. A pair-level comm->reverse_comm(this) on top of that would
  // send the same contribution home TWICE.
  //
  // The precondition is CutoffGraphImpl::init's newton-pair enforcement: the
  // global reduction only happens under newton, so without that refusal this
  // constant would silently drop every ghost force instead of double-counting
  // it. The two are one decision and must be read together.
  //
  // Ghosts or none, the answer is the same: the pair style never issues its
  // own reverse comm. init() has already required newton on.
  {
    SyntheticSystem sys(/*nghost=*/0);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    under.impl->init(
        env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0));
    expect_true(!under.impl->compute(sys.view()).needs_pair_reverse_comm,
                "nghost=0 -> no pair-level reverse comm");
  }
  {
    SyntheticSystem sys(/*nghost=*/0);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    under.impl->init(pbc_env());
    expect_true(!under.impl->compute(sys.view()).needs_pair_reverse_comm,
                "no ghost -> nothing to communicate");
  }
  {
    // Ghosts present, forces on them, and still no pair-level reverse comm.
    SyntheticSystem sys(/*nghost=*/1);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    under.impl->init(pbc_env());
    const EvalResult result = under.impl->compute(sys.view());
    expect_true(!result.needs_pair_reverse_comm,
                "ghosts present -> still false; the integrator's global "
                "reverse comm owns the reduction");
    // And the ghost really did receive a force, so the assertion above is about
    // reduction OWNERSHIP and not about there being nothing to reduce.
    expect_force_row(
        sys, 3, /*scattered=*/true,
        "the ghost carries a force that the integrator must reduce");
  }
  // The field and the facade's `if (r.needs_pair_reverse_comm)` call site both
  // survive untouched: a future contract that scatters ghost forces outside
  // newton-reduced integration fills it true and re-activates
  // ContractImpl::pack_reverse / unpack_reverse, which have no other entrance.
}

void test_impl_energy_sums_locals_only() {
  // D1, locked against a hard-coded literal. One ghost is a graph node and gets
  // an atom energy like any other atom; the partition energy must nonetheless be
  // the sum over the three LOCAL atoms.
  SyntheticSystem sys(/*nghost=*/1);
  ImplUnderTest under = impl_over(model_metadata());
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(pbc_env());

  const EvalResult result = under.impl->compute(sys.view());

  // Four atoms were evaluated ...
  expect_int(under.exec->Z_seen.size(0), kNall,
             "the model is evaluated over local + ghost");
  // ... and three of them are summed.
  expect_near(result.e_total, kExpectedETotal, 1e-12,
              "e_total is 1.5 eV * 23.060549 = 34.5908235 kcal/mol, the sum "
              "over the three LOCAL atoms");
  expect_true(
      result.e_total != kGhostInclusiveETotal,
      "e_total must not be the local+ghost sum 57.6513725 — that error "
      "is dominated by the per-species offset and grows with rank count");

  // The mask the model was handed is the other half of D1: the C++ side sums
  // the local slice, and the model is TOLD which atoms are local so its own
  // masked sum agrees. All-ones here would make a correct model return the
  // ghost-inclusive energy.
  expect_true(under.exec->local_mask_seen.defined(),
              "the impl must hand the model a local_mask");
  expect_true(under.exec->local_mask_seen.scalar_type() == at::kBool,
              "local_mask is a bool tensor");
  expect_int(under.exec->local_mask_seen.size(0), kNall,
             "local_mask has one entry per graph node, not per local atom");
  const auto ma = under.exec->local_mask_seen.accessor<bool, 1>();
  for (int i = 0; i < kNall; ++i) {
    expect_true(ma[i] == (i < kNlocal),
                "local_mask[" + std::to_string(i) + "] must be " +
                    (i < kNlocal ? "true (owned)" : "false (ghost)"));
  }
}

void test_impl_e_atom_locals_only() {
  // The per-atom energy the facade tallies into eatom. Its LENGTH is the
  // contract — nlocal, never nall — because the facade iterates the returned
  // tensor's own extent and must not consult atom->nlocal itself.
  SyntheticSystem sys(/*nghost=*/1);
  ImplUnderTest under = impl_over(model_metadata());
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(pbc_env());

  const EvalResult result = under.impl->compute(sys.view());

  expect_true(result.e_atom.defined(), "e_atom must be a defined tensor");
  expect_true(result.e_atom.device().is_cpu(), "e_atom must be a CPU tensor");
  expect_true(result.e_atom.scalar_type() == at::kDouble, "e_atom is float64");
  expect_int(result.e_atom.dim(), 1, "e_atom is 1-D");
  expect_int(result.e_atom.size(0), kNlocal,
             "e_atom has exactly nlocal entries even though nall atoms were "
             "evaluated");

  // Contents, converted once, against hand-computed literals. Distinct values,
  // so a right-length tensor with the wrong rows cannot pass.
  const auto ea = result.e_atom.accessor<double, 1>();
  for (int i = 0; i < kNlocal; ++i) {
    expect_near(ea[i], kExpectedEAtom[i], 1e-12,
                "e_atom[" + std::to_string(i) +
                    "] is the atom energy in LAMMPS "
                    "units");
  }

  // e_total and e_atom are ONE number and its parts: the facade adds the former
  // to eng_vdwl and the latter to eatom, and LAMMPS' own thermo compares them.
  // They must not come from two independent computations that could drift.
  expect_near(result.e_atom.sum().item<double>(), result.e_total, 1e-12,
              "e_total must equal the sum of e_atom");
}

void test_impl_compute_reasserts_newton_invariant() {
  // The cheap re-assertion. It reads the value CutoffGraphImpl::init
  // snapshotted, so what it catches is an internal ordering fault — compute()
  // reached on a periodic contract whose newton precondition was never
  // validated — and NOT a LAMMPS-side change after init, which init_style
  // re-running at every run setup already covers.
  //
  // (a) init never ran at all.
  {
    SyntheticSystem sys(/*nghost=*/1);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    const std::string msg =
        expect_throws([&] { (void)under.impl->compute(sys.view()); },
                      "periodic compute without a validated newton setting");
    expect_contains(msg, "newton", "the diagnostic must name newton");
    expect_int(under.exec->calls, 0,
               "the re-assertion must precede the model call — a half-applied "
               "step is worse than a refused one");
    for (int i = 0; i < kNall; ++i) {
      expect_force_row(sys, i, /*scattered=*/false, "refused compute");
    }
  }

  // (b) init ran and REFUSED, and compute is reached anyway. The snapshot was
  // never taken, so the same refusal fires rather than a stale "on".
  {
    SyntheticSystem sys(/*nghost=*/1);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    (void)expect_throws(
        [&] {
          under.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/2,
                                    /*molecular=*/0));
        },
        "init refuses newton off on a periodic contract");
    (void)expect_throws([&] { (void)under.impl->compute(sys.view()); },
                        "compute after a refused init");
  }

  // (c) nghost == 0 does not skip the check. compute without init throws.
  {
    SyntheticSystem sys(/*nghost=*/0);
    ImplUnderTest under = impl_over(model_metadata());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    const std::string msg =
        expect_throws([&] { (void)under.impl->compute(sys.view()); },
                      "compute without init, no ghosts");
    expect_contains(msg, "newton", "the diagnostic must name newton");
  }
}

// ---- coeff -----------------------------------------------------------------

void test_impl_coeff_token_count() {
  ImplUnderTest under = impl_over(model_metadata());

  // One Z per atom type, no more and no fewer. The facade strips `* *` and
  // forwards the rest, so this count is the impl's to check (ac-007).
  const std::string few = expect_throws(
      [&] { under.impl->coeff({"1", "6"}, /*ntypes=*/3); }, "too few tokens");
  expect_contains(few, "3", "the diagnostic must state the expected count");

  const std::string many = expect_throws(
      [&] { under.impl->coeff({"1", "6", "8", "1"}, /*ntypes=*/3); },
      "too many tokens");
  expect_contains(many, "4", "the diagnostic must state the observed count");

  // The matching count is accepted, so the cases above are not passing by
  // rejecting every input.
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
}

void test_impl_coeff_rejects_trailing_garbage() {
  ImplUnderTest under = impl_over(model_metadata());

  // std::stoi("6abc") silently returns 6, turning a typo into a different
  // element the model was never asked about. Full consumption refuses it.
  const std::string msg = expect_throws(
      [&] { under.impl->coeff({"1", "6abc", "8"}, /*ntypes=*/3); },
      "trailing garbage");
  expect_contains(msg, "6abc", "the diagnostic must quote the original token");

  // And the impl is not half-configured afterwards: no SpeciesMap was built, so
  // a compute() cannot proceed on a partially parsed table.
  SyntheticSystem sys(/*nghost=*/0);
  (void)expect_throws([&] { (void)under.impl->compute(sys.view()); },
                      "compute after a rejected pair_coeff");
}

void test_impl_coeff_rejects_empty_token() {
  ImplUnderTest under = impl_over(model_metadata());

  const std::string msg = expect_throws(
      [&] { under.impl->coeff({"1", "", "8"}, /*ntypes=*/3); }, "empty token");
  // The token is empty, so what the diagnostic must show is the empty quotes
  // and which LAMMPS type they came from.
  expect_contains(msg, "''", "the diagnostic must quote the original token");
  expect_contains(msg, "2", "the diagnostic must name the offending type");
}

void test_impl_coeff_rejects_out_of_range() {
  ImplUnderTest under = impl_over(model_metadata());

  // Outside int. std::from_chars reports result_out_of_range rather than
  // wrapping, so this is refused instead of becoming some other element.
  const std::string msg = expect_throws(
      [&] { under.impl->coeff({"1", "99999999999", "8"}, /*ntypes=*/3); },
      "out of range");
  expect_contains(msg, "99999999999",
                  "the diagnostic must quote the original token");
}

// ---- init ------------------------------------------------------------------

void test_impl_init_rejects_special_bonds_on_molecular() {
  ImplUnderTest under = impl_over(model_metadata());

  // LAMMPS' DEFAULT `special_bonds lj/coul 0 0 0` deletes every 1-2 / 1-3 / 1-4
  // pair from the neighbour list. GraphBridge would then never see the nearest
  // neighbours of a bonded system and the many-body model would be evaluated on
  // a mutilated graph — silently, with no diagnostic anywhere.
  const std::string lj = expect_throws(
      [&] {
        under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1,
                                  /*molecular=*/1, kSpecialDefault,
                                  kSpecialKept));
      },
      "special_bonds lj excludes bonded neighbours");
  expect_contains(lj, "special_bonds lj/coul 1.0 1.0 1.0",
                  "the diagnostic must state the required setting");

  // The coulomb half is checked too: either array alone deletes the pairs.
  constexpr double kScaledCoul[4] = {1.0, 1.0, 1.0, 0.5};
  const std::string coul = expect_throws(
      [&] {
        under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1,
                                  /*molecular=*/1, kSpecialKept, kScaledCoul));
      },
      "special_bonds coul scales bonded neighbours");
  expect_contains(coul, "special_bonds lj/coul 1.0 1.0 1.0",
                  "the diagnostic must state the required setting");

  // A molecular system that keeps every pair is accepted.
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/1));

  // THE OTHER HALF, and the one that keeps the fix from breaking working runs:
  // atom_style atomic has no bonds, so there are no special pairs to exclude
  // and the very same coefficients must NOT throw. The aspirin parity case is
  // molecular == 0, so this pair of assertions is the ONLY coverage the
  // special_bonds fix has anywhere (spec ac-008).
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0,
                            kSpecialDefault, kSpecialDefault));
}

void test_impl_init_rejects_newton_off() {
  // THE TIGHTENING lammps-iface-04-facade deferred to this spec, and the
  // precondition of EvalResult::needs_pair_reverse_comm's constant false.
  // Ghost forces are left in atom->f for Verlet::run to reduce, and that
  // reduction happens only `if (force->newton)`. With newton off, nothing
  // reduces them: every ghost's contribution is silently discarded and the
  // owning rank integrates forces that are missing their periodic cross-terms.
  ImplUnderTest under = impl_over(model_metadata());

  const std::string msg = expect_throws(
      [&] {
        under.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/2,
                                  /*molecular=*/0));
      },
      "newton off on a periodic contract");
  expect_contains(msg, "newton on",
                  "the diagnostic must name the setting the user must change");

  // Single rank does not make it safe: a periodic box has ghosts even on one
  // rank, and they are reduced by the same global reverse comm.
  (void)expect_throws(
      [&] {
        under.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/1,
                                  /*molecular=*/0));
      },
      "newton off on a periodic contract, one rank");

  // With newton on it is accepted, so the case is not passing by refusing every
  // periodic run.
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/2, /*molecular=*/0));

  // Newton off is refused on every system, including one rank and a run
  // that happens to have no ghosts. The message names newton and does not
  // offer a flag to turn the check off.
  ImplUnderTest any = impl_over(model_metadata());
  const std::string local = expect_throws(
      [&] {
        any.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/2,
                                /*molecular=*/0));
      },
      "newton off is refused even when the system has no ghosts");
  expect_contains(local, "newton on", "the diagnostic must require newton on");
  expect_true(local.find("supports_pbc") == std::string::npos,
              "the diagnostic must not name a deleted capability flag");
}

void test_impl_init_rejects_insufficient_ghost_cutoff() {
  // D3.2, fail-closed. A T-layer network's atom energy depends on everything
  // within T * cutoff, so a single-shot evaluation needs a copy of every atom
  // out to that radius. LAMMPS' default ghost cutoff is cutoff + skin, which is
  // enough only at T == 1; at T == 3 it is less than half of what is required,
  // and the model would be evaluated on a graph whose outer atoms are missing
  // neighbours — quietly, with plausible-looking forces.
  ImplUnderTest under = impl_over(deep_metadata());

  // Required: T * cutoff + skin = 3 * 2.0 + 2.0 = 8.0 A.
  // Default effective: cutoff + skin = 4.0 A.
  const std::string msg =
      expect_throws([&] { under.impl->init(env_with(1, 1, 0)); },
                    "T = 3 under the default ghost cutoff");
  expect_contains(msg, "comm_modify cutoff",
                  "the diagnostic must give the command that fixes it");
  expect_contains(msg, "8", "the diagnostic must state the required value");
  expect_contains(msg, "4", "the diagnostic must state the measured value");

  // A user-set cutoff that is still too small is refused just the same: the
  // rule is the threshold, not the presence of a comm_modify line.
  (void)expect_throws(
      [&] {
        under.impl->init(env_with(1, 1, 0, kSpecialKept, kSpecialKept,
                                  /*ghost_cutoff=*/7.9));
      },
      "a user ghost cutoff below the threshold");

  // Exactly at the threshold passes — the condition is >=, so a model is not
  // asked for a margin it does not need.
  under.impl->init(env_with(1, 1, 0, kSpecialKept, kSpecialKept,
                            /*ghost_cutoff=*/8.0));
  // And comfortably above it, which is what the diagnostic tells the user to
  // do.
  under.impl->init(env_with(1, 1, 0, kSpecialKept, kSpecialKept,
                            /*ghost_cutoff=*/12.0));

  // THE INTERPRETATION OF ZERO LIVES HERE AND NOWHERE ELSE. The facade passes
  // comm->cutghostuser raw, and LAMMPS writes 0.0 when the user never ran
  // comm_modify; only this layer turns that into "the effective cutoff is
  // cutoff + skin". A zero read literally would compare 0.0 against 8.0 and
  // refuse for the wrong reason — and, worse, would refuse a T == 1 model too.
  // Periodic, so the check is actually reached: on a non-periodic contract it
  // does not run at all and this would assert nothing.
  // T == 1 meets the default exactly: required 1 * 2 + 2 = 4, effective 4.
  impl_over(model_metadata()).impl->init(env_with(1, 1, 0));

  // A larger cutoff scales both sides and still passes: 5.5 + 2 == 1 * 5.5 + 2.
  Metadata wide = model_metadata();
  wide.cutoff = 5.5;
  impl_over(wide).impl->init(env_with(1, 1, 0));

  // T == 2 does not pass on the default (required 6, effective 4).
  Metadata two_layer = model_metadata();
  two_layer.receptive_field_layers = 2;
  (void)expect_throws(
      [&] { impl_over(two_layer).impl->init(env_with(1, 1, 0)); },
      "T = 2 under the default ghost cutoff");

  // Unset ghost cutoff is the literal 0, read as cutoff + skin, and a T == 1
  // model passes that reading.
  ImplUnderTest shallow = impl_over(model_metadata());
  shallow.impl->init(env_with(1, 1, 0, kSpecialKept, kSpecialKept,
                              /*ghost_cutoff=*/kGhostCutoffUnset));
}

// ---- reverse comm ----------------------------------------------------------

void test_impl_reverse_comm_forwards() {
  // The two comm callbacks are pure forwarding (binding decision 2): the impl
  // adds nothing, so its results must be bit-identical to calling the 03-graph
  // free functions directly. Anything else would mean the reverse-comm defect
  // routed to 05-pbc had quietly changed shape in transit.
  SyntheticSystem through_impl(/*nghost=*/1);
  SyntheticSystem direct(/*nghost=*/1);
  ImplUnderTest under = impl_over(model_metadata());

  constexpr double kSentinel = -12345.0;
  std::vector<double> via_impl(8, kSentinel);
  std::vector<double> via_free(8, kSentinel);

  const int m_impl =
      under.impl->pack_reverse(through_impl.view(), 2, 1, via_impl.data());
  const int m_free = pack_reverse_forces(direct.view(), 2, 1, via_free.data());

  expect_int(m_impl, m_free,
             "pack_reverse returns what the free function does");
  for (std::size_t k = 0; k < via_impl.size(); ++k) {
    expect_near(via_impl[k], via_free[k], 0.0,
                "pack buffer entry " + std::to_string(k) + " is bit-identical");
  }

  const std::vector<int> list = {3, 0};
  const std::vector<double> buf = {0.25, -0.5, 1.0, -2.0, 0.125, 4.0};
  under.impl->unpack_reverse(through_impl.view(), 2, list.data(), buf.data());
  unpack_reverse_forces(direct.view(), 2, list.data(), buf.data());

  for (int i = 0; i < kNall; ++i) {
    for (int k = 0; k < 3; ++k) {
      expect_near(through_impl.force(i, k), direct.force(i, k), 0.0,
                  "unpack_reverse leaves f[" + std::to_string(i) + "][" +
                      std::to_string(k) +
                      "] bit-identical to the free function");
    }
  }
  // Accumulation, not assignment, on both sides — asserted from the non-zero
  // starting force so an overwrite cannot pass.
  expect_near(through_impl.force(3, 0), kInitialForce[3][0] + 0.25, 0.0,
              "f[3][0] += buf[0]");
}

// ---- folded mode ------------------------------------------------------------

void test_impl_fold_requests_no_ghost_centres() {
  const ImplUnderTest under = impl_over(model_metadata(), GraphMode::kFold);
  const NeighborNeeds needs = under.impl->needs();
  expect_true(needs.full, "the folded mode still needs the full list");
  expect_true(!needs.ghost,
              "every folded pair is reached from an owned centre: no "
              "REQ_GHOST");
}

void test_impl_fold_refuses_several_ranks() {
  ImplUnderTest under = impl_over(model_metadata(), GraphMode::kFold);
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  const std::string msg = expect_throws(
      [&] {
        under.impl->init(
            env_with(/*newton_pair=*/1, /*nprocs=*/2, /*molecular=*/0));
      },
      "fold on two ranks");
  expect_contains(msg, "mode ghost", "the diagnostic names the alternative");
  // One rank is accepted, and newton is not the folded mode's condition: no
  // ghost force is written, so there is nothing to reduce.
  under.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/1, /*molecular=*/0));
}

void test_impl_fold_compute_folds_images_onto_owners() {
  // Atom 3 is the ghost image of atom 0. Of the surviving pairs only (2, 3)
  // touches it; folded, it is (2, 0) with shift x3 - x0.
  SyntheticSystem sys(/*nghost=*/1);
  sys.set_owner({0, 1, 2, 0});
  ImplUnderTest under = impl_over(model_metadata(), GraphMode::kFold);
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0));
  const EvalResult result = under.impl->compute(sys.view());

  expect_int(under.exec->Z_seen.size(0), kNlocal,
             "the model sees the owned atoms only");
  const auto mask = under.exec->local_mask_seen.accessor<bool, 1>();
  for (int i = 0; i < kNlocal; ++i)
    expect_true(mask[i], "every folded atom is local");

  const at::Tensor edges = under.exec->edges_seen;
  const at::Tensor shifts = under.exec->shifts_seen;
  const std::vector<EdgePair> expected = {
      {0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 0}};
  expect_int(edges.size(0), static_cast<long long>(expected.size()),
             "one folded pair per surviving neighbour");
  const auto ea = edges.accessor<int64_t, 2>();
  const auto sa = shifts.accessor<double, 2>();
  for (std::size_t k = 0; k < expected.size(); ++k) {
    expect_int(ea[static_cast<int64_t>(k)][0], expected[k].first, "source");
    expect_int(ea[static_cast<int64_t>(k)][1], expected[k].second, "target");
    for (int c = 0; c < 3; ++c) {
      const double want = k == 4 ? kCoord[3][c] - kCoord[0][c] : 0.0;
      expect_near(sa[static_cast<int64_t>(k)][c], want, 1e-15,
                  "shift = x[image] - x[owner], zero between owned atoms");
    }
  }

  for (int i = 0; i < kNlocal; ++i)
    for (int c = 0; c < 3; ++c)
      expect_near(sys.force(i, c), kScatteredForce[i][c], 1e-9,
                  "owned forces are accumulated");
  for (int c = 0; c < 3; ++c)
    expect_near(sys.force(3, c), kInitialForce[3][c], 0.0,
                "no force is written to a ghost row");

  expect_near(result.e_total, kExpectedETotal, 1e-9,
              "energy: the owned atom energies, converted");
  expect_true(result.virial_from_model, "the folded virial is the model's");
  const double voigt[6] = {1.0, 4.0, 6.0, 2.0, 3.0, 5.0};
  for (int k = 0; k < 6; ++k)
    expect_near(result.virial[static_cast<std::size_t>(k)],
                -kEnergyConv * voigt[k], 1e-9,
                "LAMMPS virial = -conv * dE/deps, xx yy zz xy xz yz");
}

void test_impl_fold_refuses_a_foreign_ghost() {
  // A ghost with no owned original (owner -1): another rank's atom.
  SyntheticSystem sys(/*nghost=*/1);
  sys.set_owner({0, 1, 2, -1});
  ImplUnderTest under = impl_over(model_metadata(), GraphMode::kFold);
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0));
  expect_contains(expect_throws([&] { (void)under.impl->compute(sys.view()); },
                                "a ghost owned elsewhere"),
                  "ghost mode", "the diagnostic names the alternative");
}

// ---- units ------------------------------------------------------------------

void test_impl_refuses_non_angstrom_length() {
  // One factor serves energy and force only while the model measures length in
  // LAMMPS' own unit; a Bohr model must be refused, not run mis-scaled.
  Metadata bohr = model_metadata();
  bohr.length_unit = "bohr";
  StubArtifact stub;
  const std::string msg = expect_throws(
      [&] {
        CutoffGraphImpl impl(bohr, kEnergyConv,
                             AtomEnergyExec(stub, bohr.dtype),
                             GraphMode::kGhost);
      },
      "a bohr model");
  expect_contains(msg, "bohr", "the diagnostic quotes the unit");
  expect_contains(msg, "angstrom", "the diagnostic names the required unit");
}

// ---- dispatcher ------------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"impl_refuses_non_angstrom_length",
       test_impl_refuses_non_angstrom_length},
      {"impl_needs_matches_legacy_request",
       test_impl_needs_matches_legacy_request},
      {"impl_compute_composes_pipeline", test_impl_compute_composes_pipeline},
      {"impl_compute_n_graph_with_ghost", test_impl_compute_n_graph_with_ghost},
      {"impl_reports_no_pair_reverse_comm",
       test_impl_reports_no_pair_reverse_comm},
      {"impl_energy_sums_locals_only", test_impl_energy_sums_locals_only},
      {"impl_e_atom_locals_only", test_impl_e_atom_locals_only},
      {"impl_compute_reasserts_newton_invariant",
       test_impl_compute_reasserts_newton_invariant},
      {"impl_coeff_token_count", test_impl_coeff_token_count},
      {"impl_coeff_rejects_trailing_garbage",
       test_impl_coeff_rejects_trailing_garbage},
      {"impl_coeff_rejects_empty_token", test_impl_coeff_rejects_empty_token},
      {"impl_coeff_rejects_out_of_range", test_impl_coeff_rejects_out_of_range},
      {"impl_init_rejects_special_bonds_on_molecular",
       test_impl_init_rejects_special_bonds_on_molecular},
      {"impl_init_rejects_newton_off", test_impl_init_rejects_newton_off},
      {"impl_init_rejects_insufficient_ghost_cutoff",
       test_impl_init_rejects_insufficient_ghost_cutoff},
      {"impl_reverse_comm_forwards", test_impl_reverse_comm_forwards},
      {"impl_fold_requests_no_ghost_centres",
       test_impl_fold_requests_no_ghost_centres},
      {"impl_fold_refuses_several_ranks", test_impl_fold_refuses_several_ranks},
      {"impl_fold_compute_folds_images_onto_owners",
       test_impl_fold_compute_folds_images_onto_owners},
      {"impl_fold_refuses_a_foreign_ghost",
       test_impl_fold_refuses_a_foreign_ghost},
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
