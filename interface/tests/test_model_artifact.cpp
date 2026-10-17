// C++ tests for molnex::runtime::ModelArtifact::open: the format named in the
// metadata selects the loader.
//
// Invariants:
//
//   1. format "aoti" opens `{dir}/{name}.pt2` as an AotiArtifact.
//   2. A format with no loader in this build is refused, naming it.
//   3. Op libraries are loaded before the package: a missing one is refused
//      before any package is touched.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "expect.h"
#include "molnex/runtime/aoti_artifact.h"
#include "molnex/runtime/metadata.h"
#include "molnex/runtime/model_artifact.h"

namespace fs = std::filesystem;
using molnex::runtime::AotiArtifact;
using molnex::runtime::Metadata;
using molnex::runtime::ModelArtifact;
using molnex::testing::expect_contains;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

fs::path graph_dir() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr) {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  return fs::path(env) / "model_graph_cpu";
}

Metadata graph_metadata() {
  return Metadata::read((graph_dir() / "model.json").string());
}

void test_open_aoti_from_metadata() {
  const std::unique_ptr<ModelArtifact> artifact =
      ModelArtifact::open(graph_dir().string(), "model", graph_metadata());
  expect_true(dynamic_cast<AotiArtifact*>(artifact.get()) != nullptr,
              "format aoti opens an AotiArtifact");
}

void test_open_refuses_unknown_format() {
  Metadata meta = graph_metadata();
  meta.format = "onnx";
  const std::string msg = expect_throws(
      [&] { (void)ModelArtifact::open(graph_dir().string(), "model", meta); },
      "an unknown format");
  expect_contains(msg, "onnx", "names the format");
  expect_contains(msg, "aoti", "names the formats it knows");
}

void test_open_loads_op_libraries_first() {
  Metadata meta = graph_metadata();
  meta.op_libraries["molnex_absent_ops"] = "absent==1";
  setenv("MOLNEX_OP_LIBRARY_PATH", graph_dir().c_str(), 1);
  expect_contains(expect_throws(
                      [&] {
                        (void)ModelArtifact::open(graph_dir().string(), "model",
                                                  meta);
                      },
                      "a missing op library"),
                  "molnex_absent_ops", "names the library the package needs");
}

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"open_aoti_from_metadata", test_open_aoti_from_metadata},
      {"open_refuses_unknown_format", test_open_refuses_unknown_format},
      {"open_loads_op_libraries_first", test_open_loads_op_libraries_first},
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
