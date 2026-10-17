// C++ tests for molnex::runtime::SiteEnergyExec.
//
// The no-fixture cases run over a stub ModelArtifact; the fixture case runs
// the five-input graph package generate_fixtures.py wrote, opened the way an
// engine opens it (Metadata::read, then ModelArtifact::open).
//
// Invariants:
//
//   1. The package must return exactly four tensors; forces (n, 3),
//      site_energy one entry per staged atom, virial (3, 3).
//   2. pos and shifts reach the package in the metadata dtype; Z, pair_index
//      and local_mask keep theirs.
//   3. Outputs are CPU float64 copies the caller owns, equal to the reference
//      at 1e-12 for an fp64 package — with non-zero shifts, so a runtime that
//      dropped them would miss.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "expect.h"
#include "molnex/runtime/metadata.h"
#include "molnex/runtime/model_artifact.h"
#include "molnex/runtime/site_energy_exec.h"

namespace fs = std::filesystem;
using molnex::runtime::Metadata;
using molnex::runtime::ModelArtifact;
using molnex::runtime::SiteEnergyExec;
using molnex::runtime::SiteEnergyOutputs;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

/// Answers every run with a fixed output list and records its inputs.
class StubArtifact : public ModelArtifact {
 public:
  explicit StubArtifact(std::vector<at::Tensor> outputs)
      : outputs_(std::move(outputs)) {}

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) override {
    seen = inputs;
    return outputs_;
  }

  at::Device device() const override { return at::Device(at::kCPU); }

  std::vector<at::Tensor> seen;

 private:
  std::vector<at::Tensor> outputs_;
};

const auto kF64 = at::TensorOptions().dtype(at::kDouble);

/// Three staged atoms: Z, pos, pair_index, shifts, local_mask.
struct Staged {
  at::Tensor Z = at::tensor({1, 6, 8}, at::TensorOptions().dtype(at::kLong));
  at::Tensor pos = at::zeros({3, 3}, kF64);
  at::Tensor pair_index =
      at::tensor({0, 1, 1, 0}, at::TensorOptions().dtype(at::kLong))
          .view({2, 2});
  at::Tensor shifts = at::zeros({2, 3}, kF64);
  at::Tensor local_mask = at::ones({3}, at::TensorOptions().dtype(at::kBool));
};

SiteEnergyOutputs run_stub(StubArtifact& stub, at::ScalarType dtype) {
  const Staged s;
  return SiteEnergyExec(stub, dtype)
      .run(s.Z, s.pos, s.pair_index, s.shifts, s.local_mask);
}

/// The four outputs of a well-formed three-atom package, in `dtype`.
std::vector<at::Tensor> four_outputs(at::ScalarType dtype = at::kDouble) {
  const auto o = at::TensorOptions().dtype(dtype);
  return {at::ones({1}, o), at::zeros({3, 3}, o), at::zeros({3}, o),
          at::zeros({3, 3}, o)};
}

void test_exec_requires_four_outputs() {
  StubArtifact three(
      {at::ones({1}, kF64), at::zeros({3, 3}, kF64), at::zeros({3}, kF64)});
  const std::string few = expect_throws(
      [&] { (void)run_stub(three, at::kDouble); }, "three outputs");
  expect_contains(few, "3 output", "the diagnostic states the observed count");
  expect_contains(few, "virial", "the diagnostic names the missing one");
  expect_contains(few, "schema 2", "and says a schema-1 package returns 3");

  std::vector<at::Tensor> five_out = four_outputs();
  five_out.push_back(at::zeros({1}, kF64));
  StubArtifact five(five_out);
  expect_contains(
      expect_throws([&] { (void)run_stub(five, at::kDouble); }, "five outputs"),
      "5 output", "the diagnostic states the observed count");
}

void test_exec_requires_one_site_energy_per_atom() {
  std::vector<at::Tensor> outs = four_outputs();
  outs[2] = at::zeros({2}, kF64);
  StubArtifact short_sites(outs);
  const std::string msg = expect_throws(
      [&] { (void)run_stub(short_sites, at::kDouble); }, "two site energies");
  expect_contains(msg, "2 entries", "states the observed length");
  expect_contains(msg, "3 staged", "states the staged count");
}

void test_exec_requires_shaped_forces_and_virial() {
  std::vector<at::Tensor> bad_forces = four_outputs();
  bad_forces[1] = at::zeros({2, 3}, kF64);
  StubArtifact forces_stub(bad_forces);
  expect_contains(
      expect_throws([&] { (void)run_stub(forces_stub, at::kDouble); },
                    "forces for two atoms"),
      "(3, 3)", "names the required forces shape");

  std::vector<at::Tensor> bad_virial = four_outputs();
  bad_virial[3] = at::zeros({6}, kF64);
  StubArtifact virial_stub(bad_virial);
  expect_contains(
      expect_throws([&] { (void)run_stub(virial_stub, at::kDouble); },
                    "a Voigt-vector virial"),
      "virial", "names the virial");
}

void test_exec_casts_pos_to_model_dtype() {
  StubArtifact stub(four_outputs(at::kFloat));
  const SiteEnergyOutputs out = run_stub(stub, at::kFloat);
  expect_int(static_cast<long long>(stub.seen.size()), 5, "five inputs");
  expect_true(stub.seen[0].scalar_type() == at::kLong, "Z stays int64");
  expect_true(stub.seen[1].scalar_type() == at::kFloat, "pos is cast");
  expect_true(stub.seen[2].scalar_type() == at::kLong,
              "pair_index stays int64");
  expect_true(stub.seen[3].scalar_type() == at::kFloat, "shifts are cast");
  expect_true(stub.seen[4].scalar_type() == at::kBool, "local_mask stays bool");
  expect_true(out.forces.scalar_type() == at::kDouble &&
                  out.site_energy.scalar_type() == at::kDouble &&
                  out.energy.scalar_type() == at::kDouble &&
                  out.virial.scalar_type() == at::kDouble,
              "outputs come back float64");
}

// ---- fixture-backed ---------------------------------------------------------

fs::path fixture_dir() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr) {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  return fs::path(env);
}

std::map<std::string, at::Tensor> load_tensor_dict(const fs::path& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + path.string());
  const std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                                std::istreambuf_iterator<char>());
  std::map<std::string, at::Tensor> out;
  for (const auto& kv : torch::pickle_load(bytes).toGenericDict()) {
    out.emplace(kv.key().toStringRef(), kv.value().toTensor());
  }
  return out;
}

void expect_close(const at::Tensor& got, const at::Tensor& want, double atol,
                  const std::string& tag) {
  const at::Tensor a = got.to(at::kCPU, at::kDouble);
  const at::Tensor b = want.to(at::kCPU, at::kDouble);
  expect_true(a.sizes().equals(b.sizes()), tag + ": shape");
  const double diff = (a - b).abs().max().item<double>();
  expect_true(diff <= atol, tag + ": max abs diff " + std::to_string(diff));
}

void test_exec_matches_reference() {
  const fs::path dir = fixture_dir() / "model_graph_cpu";
  const Metadata meta = Metadata::read((dir / "model.json").string());
  const std::unique_ptr<ModelArtifact> artifact =
      ModelArtifact::open(dir.string(), "model", meta);
  expect_true(artifact->device().is_cpu(), "the graph fixture is a CPU export");

  const auto ref = load_tensor_dict(fixture_dir() / "reference_graph_cpu.pt");
  SiteEnergyExec exec(*artifact, meta.dtype);
  const SiteEnergyOutputs out =
      exec.run(ref.at("Z"), ref.at("pos"), ref.at("pair_index"),
               ref.at("shifts"), ref.at("local_mask"));
  expect_true(ref.at("shifts").abs().sum().item<double>() > 0.0,
              "the reference exercises non-zero shifts");
  // A closed-form fp64 package: 1e-12 is a parity claim, not a tolerance.
  expect_close(out.energy, ref.at("energy"), 1e-12, "energy");
  expect_close(out.forces, ref.at("forces"), 1e-12, "forces");
  expect_close(out.site_energy, ref.at("site_energy"), 1e-12, "site_energy");
  expect_close(out.virial, ref.at("virial"), 1e-12, "virial");
  expect_true(out.forces.is_contiguous() && out.forces.device().is_cpu(),
              "forces are contiguous CPU");

  const SiteEnergyOutputs again =
      exec.run(ref.at("Z"), ref.at("pos"), ref.at("pair_index"),
               ref.at("shifts"), ref.at("local_mask"));
  expect_true(again.forces.data_ptr() != out.forces.data_ptr(),
              "a second run must not alias the first run's storage");
}

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"exec_requires_four_outputs", test_exec_requires_four_outputs},
      {"exec_requires_one_site_energy_per_atom",
       test_exec_requires_one_site_energy_per_atom},
      {"exec_requires_shaped_forces_and_virial",
       test_exec_requires_shaped_forces_and_virial},
      {"exec_casts_pos_to_model_dtype", test_exec_casts_pos_to_model_dtype},
      {"exec_matches_reference", test_exec_matches_reference},
  };
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <test_name>\n", argv[0]);
    for (const auto& [k, _] : registry())
      std::fprintf(stderr, "  %s\n", k.c_str());
    return 2;
  }
  const auto it = registry().find(argv[1]);
  if (it == registry().end()) {
    std::fprintf(stderr, "unknown test: %s\n", argv[1]);
    return 2;
  }
  try {
    it->second();
    std::fprintf(stderr, "[pass] %s\n", argv[1]);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[fail] %s: %s\n", argv[1], e.what());
    return 1;
  }
}
