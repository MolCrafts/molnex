// C++ acceptance tests for molnex::interface::ModelRunner.
//
// Each test is a self-contained function selected by name via argv[1];
// the CMakeLists wires one `add_test(...)` per test so ctest reports
// granular pass/fail.
//
// Fixtures (exported .so + reference .pt files) are produced by
// `generate_fixtures.py`, registered as a ctest setup fixture, and
// found at the path passed via the `MOLNEX_INTERFACE_FIXTURE_DIR`
// environment variable.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;
using molnex::interface::ModelRunner;

namespace {

fs::path fixture_dir() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr || std::string(env).empty()) {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  return fs::path(env);
}

// torch.save({"input": t1, "output": t2}) round-tripped via pickle.
std::pair<at::Tensor, at::Tensor> load_reference(const fs::path& pt_path) {
  std::ifstream is(pt_path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + pt_path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                          std::istreambuf_iterator<char>());
  auto iv = torch::pickle_load(bytes);
  auto dict = iv.toGenericDict();
  at::Tensor input, output;
  for (const auto& kv : dict) {
    const auto& k = kv.key().toStringRef();
    if (k == "input") input = kv.value().toTensor();
    else if (k == "output") output = kv.value().toTensor();
  }
  if (!input.defined() || !output.defined()) {
    throw std::runtime_error("reference .pt missing 'input' or 'output'");
  }
  return {input, output};
}

std::unordered_map<std::string, at::Tensor>
load_state_dict(const fs::path& pt_path) {
  std::ifstream is(pt_path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + pt_path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                          std::istreambuf_iterator<char>());
  auto iv = torch::pickle_load(bytes);
  auto dict = iv.toGenericDict();
  std::unordered_map<std::string, at::Tensor> out;
  for (const auto& kv : dict) {
    out.emplace(kv.key().toStringRef(), kv.value().toTensor());
  }
  return out;
}

void expect_allclose(const at::Tensor& got, const at::Tensor& ref,
                     double atol, const std::string& tag) {
  auto a = got.detach().to(at::kCPU).to(at::kFloat).contiguous();
  auto b = ref.detach().to(at::kCPU).to(at::kFloat).contiguous();
  if (!a.sizes().equals(b.sizes())) {
    throw std::runtime_error(tag + ": shape mismatch " +
                             c10::str(a.sizes()) + " vs " + c10::str(b.sizes()));
  }
  auto diff = (a - b).abs().max().item<double>();
  if (diff > atol) {
    throw std::runtime_error(tag + ": max abs diff " + std::to_string(diff) +
                             " exceeds atol " + std::to_string(atol));
  }
}

// ---- tests ----------------------------------------------------------------

void test_cpu_run_matches_reference() {
  auto dir = fixture_dir();
  ModelRunner runner((dir / "model_cpu").string());
  if (runner.device() != "cpu") {
    throw std::runtime_error("expected device=cpu, got " + runner.device());
  }
  auto [input, ref] = load_reference(dir / "reference_cpu.pt");
  auto out = runner.run({input});
  if (out.size() != 1) {
    throw std::runtime_error("expected 1 output, got " + std::to_string(out.size()));
  }
  expect_allclose(out[0], ref, 1e-5, "cpu_run");
}

void test_cuda_run_matches_reference() {
  if (!torch::cuda::is_available()) {
    std::fprintf(stderr, "[skip] CUDA not available\n");
    return;
  }
  auto dir = fixture_dir();
  if (!fs::exists(dir / "model_cuda")) {
    std::fprintf(stderr, "[skip] cuda fixture missing\n");
    return;
  }
  ModelRunner runner((dir / "model_cuda").string());
  if (runner.device() != "cuda") {
    throw std::runtime_error("expected device=cuda, got " + runner.device());
  }
  auto [input, ref] = load_reference(dir / "reference_cuda.pt");
  auto out = runner.run({input.to(at::kCUDA)});
  expect_allclose(out[0], ref, 1e-5, "cuda_run");
}

void test_update_weights_from_path() {
  auto dir = fixture_dir();
  ModelRunner runner((dir / "model_cpu").string());
  auto [input, _orig_ref] = load_reference(dir / "reference_cpu.pt");
  (void)_orig_ref;
  auto [_alt_input, alt_ref] = load_reference(dir / "reference_alt_cpu.pt");
  (void)_alt_input;

  runner.update_weights((dir / "alt_cpu.pt").string());
  auto out = runner.run({input});
  expect_allclose(out[0], alt_ref, 1e-5, "update_path");
}

void test_update_weights_from_map() {
  auto dir = fixture_dir();
  ModelRunner runner((dir / "model_cpu").string());
  auto [input, _] = load_reference(dir / "reference_cpu.pt");
  (void)_;
  auto [_alt_input, alt_ref] = load_reference(dir / "reference_alt_cpu.pt");
  (void)_alt_input;

  auto sd = load_state_dict(dir / "alt_cpu.pt");
  runner.update_weights(sd);
  auto out = runner.run({input});
  expect_allclose(out[0], alt_ref, 1e-5, "update_map");
}

void test_parameter_info() {
  auto dir = fixture_dir();
  ModelRunner runner((dir / "model_cpu").string());
  auto info = runner.parameter_info();
  if (info.empty()) {
    throw std::runtime_error("parameter_info() returned empty list");
  }
  // expect at least the linear.weight and linear.bias from the TinyLinear fixture
  std::set<std::string> names;
  for (const auto& [name, _dtype] : info) names.insert(name);
  bool has_w = false, has_b = false;
  for (const auto& n : names) {
    if (n.find("weight") != std::string::npos) has_w = true;
    if (n.find("bias") != std::string::npos) has_b = true;
  }
  if (!has_w || !has_b) {
    std::string joined;
    for (const auto& n : names) {
      if (!joined.empty()) joined += ", ";
      joined += n;
    }
    throw std::runtime_error("parameter_info missing weight/bias; got: " + joined);
  }
}

void test_concurrent_run_and_update() {
  auto dir = fixture_dir();
  ModelRunner runner((dir / "model_cpu").string());
  auto [input, _] = load_reference(dir / "reference_cpu.pt");
  (void)_;
  const std::string alt = (dir / "alt_cpu.pt").string();

  std::atomic<bool> stop{false};
  std::atomic<int> run_count{0};
  std::atomic<bool> failed{false};

  std::thread reader([&] {
    try {
      while (!stop.load()) {
        auto out = runner.run({input});
        if (!out[0].defined()) failed.store(true);
        run_count.fetch_add(1);
      }
    } catch (...) {
      failed.store(true);
    }
  });

  try {
    for (int i = 0; i < 10; ++i) {
      runner.update_weights(alt);
    }
  } catch (...) {
    failed.store(true);
  }
  // let reader catch up to at least 100 inferences
  while (run_count.load() < 100 && !failed.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  stop.store(true);
  reader.join();

  if (failed.load()) {
    throw std::runtime_error("concurrent run/update saw failure");
  }
  if (run_count.load() < 100) {
    throw std::runtime_error("reader did not reach 100 inferences");
  }
}

void test_load_bad_path_throws() {
  bool threw = false;
  try {
    ModelRunner runner("/this/path/does/not/exist");
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) {
    throw std::runtime_error("expected exception on missing model dir");
  }
}

// ---- dispatcher -----------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"cpu_run_matches_reference", test_cpu_run_matches_reference},
      {"cuda_run_matches_reference", test_cuda_run_matches_reference},
      {"update_weights_from_path", test_update_weights_from_path},
      {"update_weights_from_map", test_update_weights_from_map},
      {"parameter_info", test_parameter_info},
      {"concurrent_run_and_update", test_concurrent_run_and_update},
      {"load_bad_path_throws", test_load_bad_path_throws},
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
