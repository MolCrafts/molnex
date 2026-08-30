// C++ acceptance tests for the contract-implementation layer
// (spec lammps-iface-04-facade): the attribute-only contract types
// (NeighborNeeds / EvalResult / LammpsRunEnv), CutoffGraphImpl as the
// composition point of every 02-view and 03-graph component, and the
// exhaustive-switch dispatch in ContractImpl::for_contract.
//
// House style follows test_bridge_graph.cpp: no test framework, one function
// per case, a function-local static map dispatcher, and the [pass]/[fail]
// stderr protocol.
//
// WHAT MAKES THIS LAYER TESTABLE AT ALL — and it is the methodological result
// of the spec, not an accident. `Exec` is INJECTED into CutoffGraphImpl, so
// every impl_* case below runs with:
//
//   * no LAMMPS build   — LammpsView hands the impl the same row-pointer /
//                         neighbour-span shapes out of this file's own
//                         std::vector storage that LAMMPS hands out of atom->x,
//   * no CUDA           — the stub Exec is plain CPU arithmetic,
//   * no .pt2 package   — the stub derives from molnex::lammps_bridge::Exec
//                         DIRECTLY and holds no ModelRunner, which is also the
//                         evidence that the Exec interface is backend-neutral
//                         (ac-006): a TorchScript or any other backend attaches
//                         the same way.
//
// Every impl_* case CONSTRUCTS CutoffGraphImpl THROUGH ITS PUBLIC CONSTRUCTOR
// and never through ContractImpl::for_contract (ac-005). Going through the
// dispatch would drag in a LammpsContract-backed ModelRunner and turn a
// composition test into an integration test — and would leave "direct
// construction is the primary constructor story" (CLAUDE.md) uncovered. Only
// the two registry_* cases exercise the dispatch, and they are the only two
// here that need a fixture, because ContractImpl::for_contract takes a
// ModelRunner& and a ModelRunner cannot exist without a real package.
//
// API contract under test (namespace molnex::lammps; the include paths come
// from the molnex_lammps_impls target's PUBLIC include directory, which is
// interface/lammps, so `impls/` is the prefix):
//
//   #include "impls/contract_impl.h"
//
//   struct NeighborNeeds { bool full; bool ghost; double cutoff;
//                          int comm_reverse; int comm_reverse_off; };
//   struct EvalResult    { double e_total; at::Tensor virial;
//                          bool needs_pair_reverse_comm; };
//   struct LammpsRunEnv  { int newton_pair; int nprocs; int molecular;
//                          const double* special_lj; const double* special_coul; };
//
//   class ContractImpl {
//    public:
//     virtual ~ContractImpl();
//     virtual NeighborNeeds needs() const = 0;
//     virtual void coeff(const std::vector<std::string>& tokens, int ntypes) = 0;
//     virtual void init(const LammpsRunEnv& env) = 0;
//     virtual EvalResult compute(const molnex::lammps_bridge::LammpsView&) = 0;
//     virtual int  pack_reverse(const LammpsView&, int n, int first, double* buf) = 0;
//     virtual void unpack_reverse(const LammpsView&, int n, const int* list,
//                                 const double* buf) = 0;
//     static std::unique_ptr<ContractImpl> for_contract(
//         Contract, const LammpsContract&, const std::string& lammps_unit_style,
//         molnex::interface::ModelRunner&);
//   };
//
//   #include "impls/cutoff_graph_impl.h"
//
//   class CutoffGraphImpl : public ContractImpl {
//    public:
//     CutoffGraphImpl(const LammpsContract&, const std::string& lammps_unit_style,
//                     std::unique_ptr<molnex::lammps_bridge::Exec>);
//   };
//
// The invariants these cases exist to lock:
//
//   1. NeighborNeeds reproduces the pre-refactor neighbour request exactly
//      (REQ_FULL, comm_reverse 3 / 3, cutoff from the meta).
//   2. compute() composes stage_nodes -> GraphBridge -> Exec -> ForceScatter and
//      applies energy_conv ONCE, to a hand-computed literal.
//   3. n_graph is `supports_pbc ? nall() : nlocal` — the semantics migrated out
//      of the facade, and the ghost guard that makes the two agree when
//      supports_pbc is false.
//   4. EvalResult carries NO forces: the impl has already scattered them into
//      atom->f, because that is the array comm->reverse_comm reads.
//   5. needs_pair_reverse_comm is `supports_pbc && nghost > 0`, produced by the
//      impl, so lammps-iface-05-pbc changes only the answer.
//   6. pair_coeff residual tokens are parsed by FULL CONSUMPTION: "6abc" is an
//      error, not the element 6 that std::stoi would silently return.
//   7. special_bonds that excludes 1-2 / 1-3 / 1-4 neighbours is intercepted on
//      a molecular system, and NOT on an atomic one. These two halves are the
//      ONLY coverage of that fix anywhere: the aspirin parity case is
//      molecular == 0 and exercises none of it (spec ac-008).
//   8. The single-domain check is carried over verbatim, newton condition
//      included — tightening it is a behaviour change owed to 05-pbc.
//   9. The contract dispatch is a closed set: cutoff-graph builds, fixed-graph
//      is a loud refusal that lists every valid contract name.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/comm_reverse.h"
#include "bridge/exec.h"
#include "bridge/lammps_view.h"
#include "expect.h"
#include "impls/contract_impl.h"
#include "impls/cutoff_graph_impl.h"
#include "impls/registry.h"
#include "lammps_contract.h"
#include "molnex/interface/meta_schema.h"
#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;

using molnex::interface::MetaSchema;
using molnex::interface::ModelRunner;
using molnex::lammps::Contract;
using molnex::lammps::ContractImpl;
using molnex::lammps::CutoffGraphImpl;
using molnex::lammps::EvalResult;
using molnex::lammps::LammpsContract;
using molnex::lammps::LammpsRunEnv;
using molnex::lammps::NeighborNeeds;
using molnex::lammps_bridge::Exec;
using molnex::lammps_bridge::LammpsView;
using molnex::lammps_bridge::ModelOutputs;
using molnex::lammps_bridge::pack_reverse_forces;
using molnex::lammps_bridge::unpack_reverse_forces;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_string;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- the export document ---------------------------------------------------
//
// A field-by-field schema-v2 `lammps` block, so one case can perturb exactly
// one field of an otherwise well-formed document. Deliberately a second,
// smaller copy of the builder in test_lammps_contract.cpp rather than a shared
// test header: the two want different defaults (this one is an eV model with a
// 2 A cutoff, matching the synthetic system below), and extracting at the
// second use would couple every future contract case to every future impl case.

struct MetaFields {
  std::string contract = "cutoff-graph";
  std::string execution = "dynamic";
  // Matches kCutoff below: the impl hands this straight to GraphBridge.
  std::string cutoff = "2.0";
  // A metal (eV) model, so running it under LAMMPS `real` gives a conversion
  // factor that is not 1.0 and a missing multiplication cannot pass unnoticed.
  std::string units = "metal";
  std::string energy_unit = "eV";
  std::string length_unit = "angstrom";
  // 1 model energy unit (eV) expressed in each style. The `real` value is the
  // eV -> kcal/mol ratio and MUST equal kEnergyConv below; hard-coded on both
  // sides on purpose, since a fixture may not be computed by the code under
  // test.
  std::string energy_conv = R"({"real": 23.060549, "metal": 1.0})";
  std::string species = "[1, 6, 8]";
  std::string model_dtype = "float64";
  std::string inputs = R"(["Z", "pos", "edge_index"])";
  std::string outputs = R"(["energy", "forces", "virial"])";
  std::string adapter = "molnex-tensordict";
  std::string supports_pbc = "false";
  std::string supports_virial = "true";
};

/// The eV -> kcal/mol factor the documents above declare; see `energy_conv`.
constexpr double kEnergyConv = 23.060549;

/// The LAMMPS unit style every case runs under: a `real` simulation of an eV
/// model, which is what makes kEnergyConv != 1.
const std::string kUnitStyle = "real";

std::string v2_meta(const MetaFields& f) {
  return std::string(R"json({
  "device": "cpu",
  "model_class": "Sonata",
  "lammps": {
    "schema_version": 2,
    "contract": ")json") +
         f.contract + R"json(",
    "execution": ")json" + f.execution + R"json(",
    "cutoff": )json" + f.cutoff + R"json(,
    "units": ")json" + f.units + R"json(",
    "energy_unit": ")json" + f.energy_unit + R"json(",
    "length_unit": ")json" + f.length_unit + R"json(",
    "energy_conv": )json" + f.energy_conv + R"json(,
    "species": )json" + f.species + R"json(,
    "model_dtype": ")json" + f.model_dtype + R"json(",
    "inputs": )json" + f.inputs + R"json(,
    "outputs": )json" + f.outputs + R"json(,
    "adapter": ")json" + f.adapter + R"json(",
    "supports_pbc": )json" + f.supports_pbc + R"json(,
    "supports_virial": )json" + f.supports_virial + R"json(
  }
})json";
}

/// The typed contract for one document. The contract owns its copy of the block
/// text, so the MetaSchema temporary dying here is deliberate.
LammpsContract contract_of(const MetaFields& f) {
  return LammpsContract{MetaSchema{v2_meta(f)}};
}

// ---- the synthetic system --------------------------------------------------

// Three local atoms plus one ghost slot. Whether that fourth atom is a ghost or
// merely unused storage is the view's business: a `supports_pbc=false` export
// must be handed nghost == 0, which is exactly what the ghost guard enforces.
constexpr int kNlocal = 3;
constexpr int kNall = 4;

/// Model cutoff in angstrom; equals MetaFields::cutoff.
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

/// Forces the stub Exec returns, in MODEL units. Every entry is a power of two
/// so `kEnergyConv * entry` is an exact binary scaling and the expectations
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

/// Total energy the stub Exec returns, in MODEL units (eV).
constexpr double kStubEnergy = 8.0;
/// The same energy in LAMMPS `real` units: 8.0 * 23.060549, by hand.
constexpr double kExpectedETotal = 184.484392;

/// Virial the stub Exec returns when asked, in MODEL units, (xx yy zz xy xz yz).
constexpr double kModelVirial[6] = {1.0, -2.0, 4.0, 0.5, -0.25, 8.0};
/// The same, times kEnergyConv, by hand (every factor is a power of two).
constexpr double kExpectedVirial[6] = {23.060549,  -46.121098,  92.242196,
                                       11.5302745, -5.76513725, 184.484392};

using EdgePair = std::pair<int, int>;

/// Every surviving directed pair with n_graph == 4 (the ghost is a node).
const std::vector<EdgePair> kEdgesAll = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}};
/// The same with n_graph == 3: the ghost is not a node, so (2,3) goes.
const std::vector<EdgePair> kEdgesLocal = {{0, 1}, {1, 0}, {1, 2}, {2, 1}};

/// Full neighbour lists as LAMMPS builds them at cutoff + skin: every atom
/// lists every other one, so the model cutoff refilter is doing real work.
std::vector<std::vector<int>> full_lists() {
  return {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {}};
}

/// The storage a synthetic LammpsView borrows: atom arrays plus neighbour spans.
///
/// Non-copyable: the pointer vectors point into this object's own storage.
class SyntheticSystem {
 public:
  /// \param nghost  Ghost count the view reports. Storage is always four atoms;
  ///                nghost == 0 means the fourth is not addressable, which is
  ///                the single-domain shape a supports_pbc=false export needs.
  explicit SyntheticSystem(int nghost)
      : type_(kType), jlists_(full_lists()), ilist_({0, 1, 2}), nghost_(nghost) {
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
                      static_cast<int>(ilist_.size()),
                      kNeighMask};
  }

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
};

// ---- the injected stub Exec ------------------------------------------------

/// An `Exec` that answers with closed-form outputs and records what it was fed.
///
/// Derives from molnex::lammps_bridge::Exec DIRECTLY: no AOTI type, no
/// ModelRunner, no `.pt2`. That is the ac-006 evidence — the interface names no
/// backend, so a test backend attaches exactly where a TorchScript one would.
class StubExec : public Exec {
 public:
  /// \param emit_virial  Whether to return a virial alongside energy/forces.
  explicit StubExec(bool emit_virial)
      : Exec(at::Device(at::kCPU), at::kDouble, emit_virial),
        emit_virial_(emit_virial) {}

  ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
                   const at::Tensor& edge_index) override {
    calls += 1;
    Z_seen = Z.clone();
    pos_seen = pos_f64.clone();
    edges_seen = edge_index.clone();

    const auto opts = at::TensorOptions().dtype(at::kDouble);
    ModelOutputs out;
    out.energy = at::full({1}, kStubEnergy, opts);

    const int64_t n = Z.size(0);
    out.forces = at::zeros({n, 3}, opts);
    auto fa = out.forces.accessor<double, 2>();
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t k = 0; k < 3; ++k) {
        fa[i][k] = kModelForce[i][k];
      }
    }

    if (emit_virial_) {
      out.virial = at::zeros({6}, opts);
      auto va = out.virial.accessor<double, 1>();
      for (int64_t k = 0; k < 6; ++k) va[k] = kModelVirial[k];
    }
    return out;
  }

  /// How many times the impl called through, and with what.
  int calls = 0;
  at::Tensor Z_seen;
  at::Tensor pos_seen;
  at::Tensor edges_seen;

 private:
  bool emit_virial_;
};

/// A CutoffGraphImpl over `fields`, with a stub Exec whose probe is handed back.
///
/// Returns the impl by value — CutoffGraphImpl's PUBLIC constructor is the only
/// construction path any impl_* case uses (ac-005 / ac-010).
struct ImplUnderTest {
  std::unique_ptr<CutoffGraphImpl> impl;
  StubExec* exec;  ///< Borrowed; owned by `impl`.
};

ImplUnderTest impl_over(const MetaFields& fields, bool emit_virial = false) {
  auto stub = std::make_unique<StubExec>(emit_virial);
  StubExec* probe = stub.get();
  return ImplUnderTest{
      std::make_unique<CutoffGraphImpl>(contract_of(fields), kUnitStyle,
                                        std::move(stub)),
      probe};
}

/// Fields for a periodic export: ghosts are graph nodes and n_graph is nall().
MetaFields pbc_fields() {
  MetaFields f;
  f.supports_pbc = "true";
  return f;
}

/// LAMMPS' special-bond arrays, index 0 unused, [1..3] the 1-2 / 1-3 / 1-4
/// scale factors. `1.0 1.0 1.0` is the only setting that leaves the neighbour
/// list intact; LAMMPS' DEFAULT is 0.0 0.0 0.0, which deletes those pairs.
constexpr double kSpecialKept[4] = {1.0, 1.0, 1.0, 1.0};
constexpr double kSpecialDefault[4] = {1.0, 0.0, 0.0, 0.0};

/// A run environment with the specials that keep every bonded neighbour.
LammpsRunEnv env_with(int newton_pair, int nprocs, int molecular,
                      const double* special_lj = kSpecialKept,
                      const double* special_coul = kSpecialKept) {
  return LammpsRunEnv{newton_pair, nprocs, molecular, special_lj, special_coul};
}

/// Require the recorded edge tensor to be exactly `want`.
void expect_edges(const at::Tensor& edges, const std::vector<EdgePair>& want,
                  const std::string& tag) {
  expect_true(edges.defined(), tag + ": the impl must have built an edge_index");
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
  const ImplUnderTest under = impl_over(MetaFields{});
  const NeighborNeeds needs = under.impl->needs();

  // Value for value, the request pair_molnex.cpp made before the refactor:
  // REQ_FULL (cpp:265), comm_reverse / comm_reverse_off = 3 (cpp:56-57) and the
  // cutoff init_one() reported (cpp:256).
  expect_true(needs.full, "a many-body model needs the FULL neighbour list");
  expect_true(!needs.ghost,
              "REQ_GHOST belongs to lammps-iface-05-pbc; this spec requests no "
              "ghost neighbours");
  expect_near(needs.cutoff, kCutoff, 1e-12, "needs.cutoff is the meta cutoff");
  expect_int(needs.comm_reverse, 3, "reverse comm packs fx, fy, fz");
  expect_int(needs.comm_reverse_off, 3, "newton-off variant packs the same 3");

  // The cutoff is READ from the contract, not a constant of the impl: a second
  // export with a different cutoff must report it.
  MetaFields wide;
  wide.cutoff = "5.5";
  expect_near(impl_over(wide).impl->needs().cutoff, 5.5, 1e-12,
              "needs.cutoff follows the contract");

  // needs() carries no predicate: the single-rank constraint is executed by
  // init(), never handed to the facade to interpret (ac-002). That is a
  // compile-time fact about the struct — there is no field to read here.
}

// ---- compute ---------------------------------------------------------------

void test_impl_compute_composes_pipeline() {
  SyntheticSystem sys(/*nghost=*/0);
  ImplUnderTest under = impl_over(MetaFields{});
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);

  const EvalResult result = under.impl->compute(sys.view());

  // (1) stage_nodes ran: Z came through the SpeciesMap the tokens built, and
  // pos is the fp64 coordinates as LAMMPS holds them.
  expect_int(under.exec->calls, 1, "the impl must call the Exec exactly once");
  expect_int(under.exec->Z_seen.size(0), kNlocal, "one Z per graph node");
  const auto za = under.exec->Z_seen.accessor<int64_t, 1>();
  for (int i = 0; i < kNlocal; ++i) {
    expect_int(za[i], kAtomZ[static_cast<std::size_t>(i)],
               "Z[" + std::to_string(i) + "] is the mapped atomic number");
  }
  expect_true(under.exec->pos_seen.scalar_type() == at::kDouble,
              "positions reach the Exec in fp64; the model-dtype cast is the "
              "execution site's own business");
  expect_near(under.exec->pos_seen.accessor<double, 2>()[1][0], kCoord[1][0],
              0.0, "pos[1][0] is LAMMPS' own coordinate");

  // (2) GraphBridge ran, with the cutoff refilter and the (source, target) sort.
  expect_edges(under.exec->edges_seen, kEdgesLocal, "composed edge_index");

  // (3) The energy is the model's, times energy_conv, ONCE: 8.0 eV in a `real`
  // run is 8.0 * 23.060549 = 184.484392 kcal/mol. A missing conversion gives
  // 8.0 and a doubled one 4254.3..., so neither can pass.
  expect_near(result.e_total, kExpectedETotal, 1e-12,
              "e_total is the model energy in LAMMPS units");

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

  // (5) The reduction-ownership seam: a single-domain export never asks the
  // pair style to reverse-communicate.
  expect_true(!result.needs_pair_reverse_comm,
              "supports_pbc=false must not ask the facade for a reverse comm");
}

void test_impl_compute_n_graph_local_only() {
  // supports_pbc = false. n_graph is nlocal, and the ghost guard is what keeps
  // nall() from ever disagreeing with it on this path — the two are the same
  // number precisely because nghost must be 0.
  SyntheticSystem sys(/*nghost=*/0);
  ImplUnderTest under = impl_over(MetaFields{});
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);

  (void)under.impl->compute(sys.view());

  expect_int(under.exec->Z_seen.size(0), kNlocal,
             "a local-only export stages exactly nlocal nodes");
  expect_int(under.exec->pos_seen.size(0), kNlocal, "pos rows == nlocal");
  // The edge set is the local one: the fourth atom's storage exists but is not
  // a node, so the (2,3) pair that survives the cutoff never becomes an edge.
  expect_edges(under.exec->edges_seen, kEdgesLocal, "local-only edge_index");
  expect_force_row(sys, 3, /*scattered=*/false,
                   "storage past nlocal is never written");
}

void test_impl_compute_n_graph_with_ghost() {
  // supports_pbc = true with one ghost: n_graph is nall(), so the ghost is a
  // graph node, gains an edge and receives a force.
  SyntheticSystem sys(/*nghost=*/1);
  ImplUnderTest under = impl_over(pbc_fields());
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);

  const EvalResult result = under.impl->compute(sys.view());

  expect_int(under.exec->Z_seen.size(0), kNall,
             "a periodic export stages local + ghost nodes");
  expect_int(under.exec->Z_seen.accessor<int64_t, 1>()[3], kAtomZ[3],
             "the ghost's Z is mapped like any other atom");
  expect_edges(under.exec->edges_seen, kEdgesAll, "ghost-inclusive edge_index");

  // Every row, ghost included, carries the converted model force.
  for (int i = 0; i < kNall; ++i) {
    expect_force_row(sys, i, /*scattered=*/true, "ghost-inclusive scatter");
  }
  expect_near(result.e_total, kExpectedETotal, 1e-12,
              "energy conversion does not depend on n_graph");
}

void test_impl_compute_rejects_ghost_without_pbc() {
  // A supports_pbc=false export handed ghosts would evaluate a molecule that is
  // silently missing its periodic images. Refused, in the migrated wording.
  SyntheticSystem sys(/*nghost=*/1);
  ImplUnderTest under = impl_over(MetaFields{});
  under.impl->coeff(kCoeffTokens, /*ntypes=*/3);

  const std::string msg = expect_throws(
      [&] { (void)under.impl->compute(sys.view()); }, "ghost without pbc");
  expect_contains(msg, "supports_pbc", "the diagnostic must name the flag");
  expect_contains(msg, "nghost", "the diagnostic must name the ghost count");

  // Nothing ran: the guard precedes the pipeline, so no partial force landed in
  // atom->f — a half-applied step is worse than a refused one.
  expect_int(under.exec->calls, 0, "the guard must precede the Exec call");
  for (int i = 0; i < kNall; ++i) {
    expect_force_row(sys, i, /*scattered=*/false, "refused compute");
  }
}

void test_impl_virial_present_and_absent() {
  SyntheticSystem sys(/*nghost=*/0);

  // (a) The export declares a virial and the model returns one: it arrives
  // (6,) and already multiplied by energy_conv, so the facade's tally is a
  // plain accumulate with no conversion of its own.
  ImplUnderTest with = impl_over(MetaFields{}, /*emit_virial=*/true);
  with.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  const EvalResult present = with.impl->compute(sys.view());
  expect_true(present.virial.defined(), "a returned virial must be carried");
  expect_int(present.virial.numel(), 6, "the virial is (6,)");
  const auto va = present.virial.accessor<double, 1>();
  for (int64_t k = 0; k < 6; ++k) {
    expect_near(va[k], kExpectedVirial[k], 1e-12,
                "virial[" + std::to_string(k) + "] is converted once");
  }

  // (b) The model returns none: `defined()` IS the presence test, so the field
  // stays undefined rather than becoming a zero tensor the facade would tally.
  SyntheticSystem plain(/*nghost=*/0);
  ImplUnderTest without = impl_over(MetaFields{}, /*emit_virial=*/false);
  without.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  const EvalResult absent = without.impl->compute(plain.view());
  expect_true(!absent.virial.defined(),
              "an absent virial must stay undefined, not become an empty tensor");
  expect_near(absent.e_total, kExpectedETotal, 1e-12,
              "energy is unaffected by the virial's absence");

  // (c) The export declares NO virial: even an Exec that offers one must not be
  // tallied, because the contract does not claim the quantity.
  MetaFields no_virial;
  no_virial.supports_virial = "false";
  no_virial.outputs = R"(["energy", "forces"])";
  SyntheticSystem third(/*nghost=*/0);
  ImplUnderTest unsupported = impl_over(no_virial, /*emit_virial=*/true);
  unsupported.impl->coeff(kCoeffTokens, /*ntypes=*/3);
  expect_true(!unsupported.impl->compute(third.view()).virial.defined(),
              "supports_virial=false must leave the virial undefined");
}

void test_impl_reports_pair_reverse_comm_condition() {
  // The reduction-ownership seam, over all four combinations of supports_pbc
  // and a non-zero ghost count. Three are reachable and assert the value; the
  // fourth (no pbc, ghosts present) is refused before the field is ever filled,
  // and that refusal IS its answer — the combination cannot silently produce a
  // reverse-comm decision at all.
  {
    SyntheticSystem sys(/*nghost=*/0);
    ImplUnderTest under = impl_over(MetaFields{});
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    expect_true(!under.impl->compute(sys.view()).needs_pair_reverse_comm,
                "supports_pbc=false, nghost=0 -> no pair-level reverse comm");
  }
  {
    SyntheticSystem sys(/*nghost=*/1);
    ImplUnderTest under = impl_over(MetaFields{});
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    (void)expect_throws([&] { (void)under.impl->compute(sys.view()); },
                        "supports_pbc=false, nghost>0 is refused outright");
  }
  {
    SyntheticSystem sys(/*nghost=*/0);
    ImplUnderTest under = impl_over(pbc_fields());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    expect_true(!under.impl->compute(sys.view()).needs_pair_reverse_comm,
                "supports_pbc=true with no ghost -> nothing to communicate");
  }
  {
    SyntheticSystem sys(/*nghost=*/1);
    ImplUnderTest under = impl_over(pbc_fields());
    under.impl->coeff(kCoeffTokens, /*ntypes=*/3);
    expect_true(under.impl->compute(sys.view()).needs_pair_reverse_comm,
                "supports_pbc=true with ghosts -> the facade must reverse comm");
  }
  // lammps-iface-05-pbc keeps this field and the facade's call site and changes
  // only the last answer above; that is the point of the seam.
}

// ---- coeff -----------------------------------------------------------------

void test_impl_coeff_token_count() {
  ImplUnderTest under = impl_over(MetaFields{});

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
  ImplUnderTest under = impl_over(MetaFields{});

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
  ImplUnderTest under = impl_over(MetaFields{});

  const std::string msg = expect_throws(
      [&] { under.impl->coeff({"1", "", "8"}, /*ntypes=*/3); }, "empty token");
  // The token is empty, so what the diagnostic must show is the empty quotes
  // and which LAMMPS type they came from.
  expect_contains(msg, "''", "the diagnostic must quote the original token");
  expect_contains(msg, "2", "the diagnostic must name the offending type");
}

void test_impl_coeff_rejects_out_of_range() {
  ImplUnderTest under = impl_over(MetaFields{});

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
  ImplUnderTest under = impl_over(MetaFields{});

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

void test_impl_init_single_domain_condition() {
  ImplUnderTest under = impl_over(MetaFields{});

  // Carried over verbatim from pair_molnex.cpp:261-263, newton condition and
  // all: a supports_pbc=false export cannot be decomposed across ranks.
  const std::string msg = expect_throws(
      [&] {
        under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/2,
                                  /*molecular=*/0));
      },
      "multi-rank without pbc");
  expect_contains(msg, "single domain", "the migrated wording is preserved");
  expect_contains(msg, "supports_pbc=true", "the diagnostic names the way out");

  // NOT tightened. newton_pair == 0 with several ranks passes today, and
  // turning that into an error would be a behaviour change owed to
  // lammps-iface-05-pbc, not a tidy-up smuggled into a migration.
  under.impl->init(env_with(/*newton_pair=*/0, /*nprocs=*/2, /*molecular=*/0));

  // A single rank is fine either way.
  under.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/1, /*molecular=*/0));

  // And a periodic export may be decomposed.
  ImplUnderTest periodic = impl_over(pbc_fields());
  periodic.impl->init(env_with(/*newton_pair=*/1, /*nprocs=*/2, /*molecular=*/0));
}

// ---- reverse comm ----------------------------------------------------------

void test_impl_reverse_comm_forwards() {
  // The two comm callbacks are pure forwarding (binding decision 2): the impl
  // adds nothing, so its results must be bit-identical to calling the 03-graph
  // free functions directly. Anything else would mean the reverse-comm defect
  // routed to 05-pbc had quietly changed shape in transit.
  SyntheticSystem through_impl(/*nghost=*/1);
  SyntheticSystem direct(/*nghost=*/1);
  ImplUnderTest under = impl_over(pbc_fields());

  constexpr double kSentinel = -12345.0;
  std::vector<double> via_impl(8, kSentinel);
  std::vector<double> via_free(8, kSentinel);

  const int m_impl =
      under.impl->pack_reverse(through_impl.view(), 2, 1, via_impl.data());
  const int m_free =
      pack_reverse_forces(direct.view(), 2, 1, via_free.data());

  expect_int(m_impl, m_free, "pack_reverse returns what the free function does");
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
                      std::to_string(k) + "] bit-identical to the free function");
    }
  }
  // Accumulation, not assignment, on both sides — asserted from the non-zero
  // starting force so an overwrite cannot pass.
  expect_near(through_impl.force(3, 0), kInitialForce[3][0] + 0.25, 0.0,
              "f[3][0] += buf[0]");
}

// ---- dispatch --------------------------------------------------------------
//
// The only two fixture-backed cases in this file: ContractImpl::for_contract
// takes a ModelRunner&, and a ModelRunner cannot be constructed without a real
// exported package. They therefore carry FIXTURES_REQUIRED and the fixture-dir
// environment variable — the same wiring as contract_reads_generated_v2_fixture.
// The package is only held, never run: the dispatch's job is to CONSTRUCT.

fs::path fixture_dir() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr || *env == '\0') {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  return fs::path(env);
}

void test_registry_returns_cutoff_graph_impl() {
  ModelRunner runner((fixture_dir() / "model_cpu").string());
  expect_string(runner.device(), "cpu", "the fixture is a CPU export");

  const LammpsContract contract = contract_of(MetaFields{});
  std::unique_ptr<ContractImpl> impl = ContractImpl::for_contract(
      contract.contract(), contract, kUnitStyle, runner);

  expect_true(impl != nullptr, "the cutoff-graph case must hand back an impl");
  expect_true(dynamic_cast<CutoffGraphImpl*>(impl.get()) != nullptr,
              "contract 'cutoff-graph' dispatches to CutoffGraphImpl");

  // The dispatched impl is configured from the same contract a directly
  // constructed one would be — the dispatch adds no policy of its own.
  const NeighborNeeds needs = impl->needs();
  expect_near(needs.cutoff, kCutoff, 1e-12, "the dispatched impl carries the "
                                            "contract cutoff");
  expect_true(needs.full, "the dispatched impl requests a full neighbour list");
  expect_int(needs.comm_reverse, 3, "the dispatched impl packs 3 doubles");
}

void test_registry_rejects_fixed_graph() {
  ModelRunner runner((fixture_dir() / "model_cpu").string());

  // A well-formed v2 export that declares the OTHER contract. Nothing about it
  // is malformed — `fixed-graph` parses to its enumerator — so the refusal must
  // come from the dispatch, which has no implementation to hand back.
  MetaFields fixed;
  fixed.contract = "fixed-graph";
  const LammpsContract contract = contract_of(fixed);
  expect_true(contract.contract() == Contract::fixed_graph,
              "the document really does declare the fixed-graph contract");

  const std::string msg = expect_throws(
      [&] {
        std::unique_ptr<ContractImpl> impl = ContractImpl::for_contract(
            contract.contract(), contract, kUnitStyle, runner);
        (void)impl;
      },
      "fixed-graph has no impl");
  expect_contains(msg, "fixed-graph", "the diagnostic must name the contract");
  expect_contains(msg, "cutoff-graph",
                  "the diagnostic must list every contract that does dispatch");
}

// ---- dispatcher ------------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"impl_needs_matches_legacy_request",
       test_impl_needs_matches_legacy_request},
      {"impl_compute_composes_pipeline", test_impl_compute_composes_pipeline},
      {"impl_compute_n_graph_local_only", test_impl_compute_n_graph_local_only},
      {"impl_compute_n_graph_with_ghost", test_impl_compute_n_graph_with_ghost},
      {"impl_compute_rejects_ghost_without_pbc",
       test_impl_compute_rejects_ghost_without_pbc},
      {"impl_virial_present_and_absent", test_impl_virial_present_and_absent},
      {"impl_reports_pair_reverse_comm_condition",
       test_impl_reports_pair_reverse_comm_condition},
      {"impl_coeff_token_count", test_impl_coeff_token_count},
      {"impl_coeff_rejects_trailing_garbage",
       test_impl_coeff_rejects_trailing_garbage},
      {"impl_coeff_rejects_empty_token", test_impl_coeff_rejects_empty_token},
      {"impl_coeff_rejects_out_of_range", test_impl_coeff_rejects_out_of_range},
      {"impl_init_rejects_special_bonds_on_molecular",
       test_impl_init_rejects_special_bonds_on_molecular},
      {"impl_init_single_domain_condition",
       test_impl_init_single_domain_condition},
      {"impl_reverse_comm_forwards", test_impl_reverse_comm_forwards},
      {"registry_returns_cutoff_graph_impl",
       test_registry_returns_cutoff_graph_impl},
      {"registry_rejects_fixed_graph", test_registry_rejects_fixed_graph},
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
