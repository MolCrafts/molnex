// C++ tests for molnex::runtime::OpLibraries.
//
// The library under load is `libmolnex_stub_ops.so` (op_library_stub.cpp),
// built beside this test; CMake passes its directory as
// MOLNEX_STUB_OPS_DIR. No package, no engine.
//
// Invariants:
//
//   1. A library is found in the search path and its requirement checked.
//   2. A library built against another requirement is refused, naming both.
//   3. A missing library is refused, naming it and where it was looked for.
//   4. The search path is MOLNEX_OP_LIBRARY_PATH, colon-separated.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "expect.h"
#include "molnex/runtime/op_libraries.h"

using molnex::runtime::OpLibraries;
using molnex::testing::expect_contains;
using molnex::testing::expect_throws;

namespace {

std::string stub_dir() {
  const char* env = std::getenv("MOLNEX_STUB_OPS_DIR");
  if (env == nullptr) throw std::runtime_error("MOLNEX_STUB_OPS_DIR not set");
  return env;
}

void test_op_library_loads_matching_requirement() {
  OpLibraries({"/nonexistent", stub_dir()})
      .load("molnex_stub_ops", "molnex_stub_ops==1.0");
}

void test_op_library_refuses_other_requirement() {
  const std::string msg = expect_throws(
      [] {
        OpLibraries({stub_dir()})
            .load("molnex_stub_ops", "molnex_stub_ops==2.0");
      },
      "a library built for another version");
  expect_contains(msg, "molnex_stub_ops==1.0", "names what it was built for");
  expect_contains(msg, "molnex_stub_ops==2.0", "names what the package needs");
}

void test_op_library_missing_is_refused() {
  const std::string msg = expect_throws(
      [] { OpLibraries({stub_dir()}).load("molnex_absent_ops", "absent==1"); },
      "a library that is not there");
  expect_contains(msg, "molnex_absent_ops", "names the library");
  expect_contains(msg, stub_dir(), "names where it looked");
}

void test_op_library_finds_arch_tagged_name() {
  OpLibraries({stub_dir()})
      .load("molnex_tagged_stub_ops", "molnex_stub_ops==1.0");
}

void test_op_library_path_from_environment() {
  setenv("MOLNEX_OP_LIBRARY_PATH", ("/nonexistent:" + stub_dir()).c_str(), 1);
  OpLibraries::from_environment().load("molnex_stub_ops",
                                       "molnex_stub_ops==1.0");
}

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"op_library_loads_matching_requirement",
       test_op_library_loads_matching_requirement},
      {"op_library_refuses_other_requirement",
       test_op_library_refuses_other_requirement},
      {"op_library_missing_is_refused", test_op_library_missing_is_refused},
      {"op_library_path_from_environment",
       test_op_library_path_from_environment},
      {"op_library_finds_arch_tagged_name",
       test_op_library_finds_arch_tagged_name},
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
