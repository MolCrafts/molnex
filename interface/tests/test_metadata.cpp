// C++ tests for molnex::runtime::Metadata, the reader of `{name}.json`.
//
// Every document but one is an in-test string, so these cases need no fixture,
// no package and no engine. `metadata_reads_python_writer` reads the document
// generate_fixtures.py wrote with molix.deploy.DeployMetadata: the one check
// that the two sides agree on the layout.
//
// Invariants:
//
//   1. Every field of the Python writer is read, species sorted and unique.
//   2. A key this reader does not know is ignored.
//   3. species must be a non-empty array of whole integers > 0.
//   4. receptive_field_layers is required, whole and >= 1.
//   5. dtype is float32 or float64; cutoff is positive.
//   6. A key nested in a sub-object never satisfies a top-level lookup.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "expect.h"
#include "molnex/runtime/metadata.h"

namespace fs = std::filesystem;
using molnex::runtime::Metadata;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_string;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

/// Raw JSON value per key, so a case perturbs or drops exactly one field.
using Fields = std::map<std::string, std::string>;

Fields valid_fields() {
  return {
      {"format", "\"aoti\""},
      {"cutoff", "5.0"},
      {"species", "[8, 1, 6, 1]"},
      {"dtype", "\"float32\""},
      {"receptive_field_layers", "2"},
      {"energy_unit", "\"eV\""},
      {"length_unit", "\"angstrom\""},
      {"op_libraries",
       R"({"molnex_cueq_ops": "cuequivariance_ops_torch==0.12.0"})"},
  };
}

std::string document(const Fields& fields) {
  std::string out = "{";
  for (const auto& [key, value] : fields) {
    if (out.size() > 1) out += ",\n";
    out += "  \"" + key + "\": " + value;
  }
  return out + "\n}";
}

/// The refusal of a document with `key` set to `value` (empty drops it).
std::string refusal(Fields fields, const std::string& key,
                    const std::string& value) {
  if (value.empty()) {
    fields.erase(key);
  } else {
    fields[key] = value;
  }
  return expect_throws([&] { (void)Metadata::parse(document(fields)); },
                       key + " = '" + value + "'");
}

void test_metadata_parses_every_field() {
  const Metadata meta = Metadata::parse(document(valid_fields()));
  expect_string(meta.format, "aoti", "format");
  expect_near(meta.cutoff, 5.0, 0.0, "cutoff");
  const std::vector<int> want = {1, 6, 8};
  expect_true(meta.species == want, "species sorted and unique");
  expect_true(meta.dtype == at::kFloat, "dtype float32");
  expect_int(meta.receptive_field_layers, 2, "receptive_field_layers");
  expect_string(meta.energy_unit, "eV", "energy_unit");
  expect_string(meta.length_unit, "angstrom", "length_unit");
  expect_int(static_cast<long long>(meta.op_libraries.size()), 1,
             "op_libraries size");
  expect_string(meta.op_libraries.at("molnex_cueq_ops"),
                "cuequivariance_ops_torch==0.12.0", "op_libraries entry");

  Fields aten_only = valid_fields();
  aten_only["op_libraries"] = "{}";
  expect_true(Metadata::parse(document(aten_only)).op_libraries.empty(),
              "an ATen-only package needs no library");
}

void test_metadata_ignores_unknown_field() {
  Fields fields = valid_fields();
  fields["comment"] = "\"written by hand\"";
  expect_string(Metadata::parse(document(fields)).format, "aoti",
                "a key the reader does not know is ignored");
}

void test_metadata_refuses_bad_species() {
  for (const std::string bad : {"[]", "[0, 1]", "[1, 6.5]", "[1, \"C\"]"}) {
    expect_contains(refusal(valid_fields(), "species", bad), "species",
                    "species " + bad);
  }
}

void test_metadata_refuses_bad_receptive_field() {
  for (const std::string bad : {"0", "-2", "2.5", ""}) {
    expect_contains(refusal(valid_fields(), "receptive_field_layers", bad),
                    "receptive_field_layers", "receptive_field_layers " + bad);
  }
}

void test_metadata_refuses_bad_dtype_and_cutoff() {
  expect_contains(refusal(valid_fields(), "dtype", "\"float16\""), "float16",
                  "the diagnostic quotes the dtype");
  expect_contains(refusal(valid_fields(), "cutoff", "0.0"), "cutoff",
                  "a non-positive cutoff is refused");
}

void test_metadata_nested_key_is_not_top_level() {
  Fields fields = valid_fields();
  fields.erase("format");
  fields["op_libraries"] = R"({"format": "aoti"})";
  expect_contains(
      expect_throws([&] { (void)Metadata::parse(document(fields)); },
                    "format only inside op_libraries"),
      "missing field 'format'", "a nested key must not answer for the top");
}

void test_metadata_reads_python_writer() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr) {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  const Metadata meta = Metadata::read(
      (fs::path(env) / "model_graph_cpu" / "model.json").string());
  expect_string(meta.format, "aoti", "format");
  expect_near(meta.cutoff, 2.0, 0.0, "cutoff");
  const std::vector<int> want = {1, 6, 8};
  expect_true(meta.species == want, "species");
  expect_true(meta.dtype == at::kDouble, "dtype");
  expect_int(meta.receptive_field_layers, 1, "receptive_field_layers");
  expect_string(meta.energy_unit, "eV", "energy_unit");
  expect_string(meta.length_unit, "angstrom", "length_unit");
  expect_true(meta.op_libraries.empty(), "the toy package is ATen-only");
}

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"metadata_parses_every_field", test_metadata_parses_every_field},
      {"metadata_ignores_unknown_field", test_metadata_ignores_unknown_field},
      {"metadata_refuses_bad_species", test_metadata_refuses_bad_species},
      {"metadata_refuses_bad_receptive_field",
       test_metadata_refuses_bad_receptive_field},
      {"metadata_refuses_bad_dtype_and_cutoff",
       test_metadata_refuses_bad_dtype_and_cutoff},
      {"metadata_nested_key_is_not_top_level",
       test_metadata_nested_key_is_not_top_level},
      {"metadata_reads_python_writer", test_metadata_reads_python_writer},
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
