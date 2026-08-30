// C++ acceptance tests for molnex::interface::MetaSchema (spec
// lammps-iface-01-meta).
//
// House style follows test_model_runner.cpp: no test framework, one function
// per case, a function-local static map dispatcher, and the [pass]/[fail]/
// [skip] stderr protocol. Unlike that file these cases need NO fixtures — every
// document is an in-test raw string literal — so CMake registers them without
// FIXTURES_REQUIRED and a LAMMPS-free, AOTI-free CI job can select them by
// name.
//
// API contract under test. The header arrives with the next task and MUST
// satisfy exactly this (pure std: no torch/, no ATen/, no LAMMPS headers):
//
//   #include "molnex/interface/meta_schema.h"
//
//   namespace molnex::interface {
//   class MetaSchema {
//    public:
//     explicit MetaSchema(std::string json_text);
//     static MetaSchema from_file(const std::string& path);
//     std::string device() const;                // top-level-only field;
//     throws
//                                                // "missing 'device'" when
//                                                called
//                                                // on a scoped sub-document
//     bool has_block(const std::string& name) const;    // top-depth key whose
//                                                       // value is a {...}
//                                                       object
//     MetaSchema block(const std::string& name) const;  // scoped sub-document;
//                                                       // throws if absent
//     std::string get_string(const std::string& key) const;          // top
//     depth std::vector<std::string> get_string_array(const std::string& key)
//     const; double get_number(const std::string& key) const; bool
//     get_bool(const std::string& key, bool fallback) const;
//   };
//   }  // namespace molnex::interface
//
// The invariant these cases exist to lock: every reader resolves at the top
// depth of *this* document only. A key inside a nested sub-object must never
// shadow — nor satisfy — a lookup at block depth, and a block-depth key must
// never be answered from the enclosing document. That scoping is what replaces
// the unenforceable "key names are unique across the whole file" promise of the
// deleted interface/lammps/lammps_meta.h.

#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "expect.h"
#include "molnex/interface/meta_schema.h"

using molnex::interface::MetaSchema;
using molnex::testing::expect_contains;
using molnex::testing::expect_near;
using molnex::testing::expect_string;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- tests ----------------------------------------------------------------

void test_meta_device_from_bare_export() {
  // Exactly what generate_fixtures.py writes through the bare Exporter: a
  // device and a model class, no `lammps` block and no schema_version. These
  // metas must keep loading after ModelRunner moves onto MetaSchema, so the
  // neutral reader may not require any of the contract fields.
  const std::string json = R"json({
  "device": "cpu",
  "model_class": "TinyLinear"
})json";
  MetaSchema doc{json};
  expect_string(doc.device(), "cpu", "bare export device()");
  expect_string(doc.get_string("model_class"), "TinyLinear",
                "bare export model_class");
  expect_true(!doc.has_block("lammps"),
              "a bare Exporter meta must not report a 'lammps' block");
}

void test_meta_scoped_block_shadowing() {
  // "cutoff" occurs at three depths, and the deepest occurrence comes *first*
  // inside the block, so the deleted whole-text scanner would answer 99.0 to
  // all three lookups below.
  const std::string json = R"json({
  "device": "cuda",
  "model_class": "Sonata",
  "cutoff": 1.0,
  "lammps": {
    "energy_conv": {"cutoff": 99.0, "real": 1.0, "metal": 0.0433641043},
    "cutoff": 5.0,
    "units": "real"
  }
})json";
  MetaSchema doc{json};
  expect_true(doc.has_block("lammps"), "has_block('lammps')");

  const MetaSchema lammps = doc.block("lammps");
  expect_near(lammps.get_number("cutoff"), 5.0, 1e-12, "block-depth cutoff");
  expect_string(lammps.get_string("units"), "real", "block-depth units");
  // The enclosing document keeps its own value: scoping cuts both ways.
  expect_near(doc.get_number("cutoff"), 1.0, 1e-12, "top-depth cutoff");
  // A sub-object one level below the block does not shadow the block-depth key,
  // and is itself addressable as a scoped document.
  expect_near(lammps.block("energy_conv").get_number("cutoff"), 99.0, 1e-12,
              "nested-object cutoff");

  // device() is a top-level-only field: on a scoped sub-document it must throw
  // rather than silently resolving to the enclosing document's "cuda".
  const std::string msg = expect_throws([&] { (void)lammps.device(); },
                                        "device() on a sub-document");
  expect_contains(msg, "device", "device() on a sub-document");
}

void test_meta_string_array() {
  // `inputs` / `outputs` are JSON arrays in schema v2. Reading only the first
  // element (the old get_string behaviour) is the defect this case locks out.
  const std::string json = R"json({
  "device": "cuda",
  "lammps": {
    "inputs": ["Z", "pos", "edge_index"],
    "outputs": ["energy", "forces", "virial"]
  }
})json";
  const MetaSchema lammps = MetaSchema{json}.block("lammps");

  const std::vector<std::string> inputs = lammps.get_string_array("inputs");
  expect_true(inputs.size() == 3, "inputs: expected 3 elements, got " +
                                      std::to_string(inputs.size()));
  expect_string(inputs[0], "Z", "inputs[0]");
  expect_string(inputs[1], "pos", "inputs[1]");
  expect_string(inputs[2], "edge_index", "inputs[2]");

  const std::vector<std::string> outputs = lammps.get_string_array("outputs");
  expect_true(outputs.size() == 3, "outputs: expected 3 elements, got " +
                                       std::to_string(outputs.size()));
  expect_string(outputs[0], "energy", "outputs[0]");
  expect_string(outputs[1], "forces", "outputs[1]");
  expect_string(outputs[2], "virial", "outputs[2]");
}

void test_meta_block_missing_returns_absent() {
  // "lammps_notes" shares a prefix with "lammps" and is a string, not an
  // object: neither may be mistaken for the contract block.
  const std::string json = R"json({
  "device": "cpu",
  "model_class": "TinyLinear",
  "lammps_notes": "exported by the bare Exporter, not export_for_lammps"
})json";
  MetaSchema doc{json};
  expect_true(!doc.has_block("lammps"), "missing block must report absent");
  expect_true(!doc.has_block("lammps_notes"), "a string value is not a block");

  const std::string msg = expect_throws([&] { (void)doc.block("lammps"); },
                                        "block() on a missing block");
  expect_contains(msg, "lammps", "block() on a missing block");
}

// ---- dispatcher -----------------------------------------------------------

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"meta_device_from_bare_export", test_meta_device_from_bare_export},
      {"meta_scoped_block_shadowing", test_meta_scoped_block_shadowing},
      {"meta_string_array", test_meta_string_array},
      {"meta_block_missing_returns_absent",
       test_meta_block_missing_returns_absent},
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
