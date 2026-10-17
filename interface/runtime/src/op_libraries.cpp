// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See op_libraries.h.

#include "molnex/runtime/op_libraries.h"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace molnex::runtime {

namespace {

// The machine tag of molnex's own per-arch builds (`libmolnex_opLib.<machine>.so`,
// so x86_64 and aarch64 builds coexist in one source tree); matches
// `platform.machine()` / CMAKE_SYSTEM_PROCESSOR.
#if defined(__aarch64__)
constexpr const char* kMachine = "aarch64";
#elif defined(__x86_64__)
constexpr const char* kMachine = "x86_64";
#else
#error "unsupported machine for the arch-tagged op library name"
#endif

}  // namespace

OpLibraries::OpLibraries(std::vector<std::string> search_path)
    : search_path_(std::move(search_path)) {}

OpLibraries OpLibraries::from_environment() {
  std::vector<std::string> dirs;
  const char* env = std::getenv("MOLNEX_OP_LIBRARY_PATH");
  if (env != nullptr) {
    std::string rest(env);
    std::size_t start = 0;
    while (start <= rest.size()) {
      const std::size_t colon = rest.find(':', start);
      const std::size_t stop = colon == std::string::npos ? rest.size() : colon;
      if (stop > start) dirs.push_back(rest.substr(start, stop - start));
      if (colon == std::string::npos) break;
      start = colon + 1;
    }
  }
  return OpLibraries(std::move(dirs));
}

void OpLibraries::load(const std::string& library,
                       const std::string& requirement) const {
  // `lib<name>.so`, or the arch-tagged `lib<name>.<machine>.so`.
  const std::vector<std::string> files = {
      "lib" + library + ".so", "lib" + library + "." + kMachine + ".so"};
  std::string file = files.front();
  // RTLD_GLOBAL: the library's operators reference torch symbols the process
  // already holds, and a later library may reference this one's.
  constexpr int kFlags = RTLD_NOW | RTLD_GLOBAL;
  void* handle = nullptr;
  std::string looked;
  if (search_path_.empty()) {
    for (const std::string& name : files) {
      handle = dlopen(name.c_str(), kFlags);
      if (handle != nullptr) {
        file = name;
        break;
      }
    }
    looked = "the dynamic loader's search path (MOLNEX_OP_LIBRARY_PATH unset)";
  } else {
    for (const std::string& dir : search_path_) {
      for (const std::string& name : files) {
        const std::string candidate =
            (std::filesystem::path(dir) / name).string();
        if (!looked.empty()) looked += ", ";
        looked += candidate;
        if (!std::filesystem::exists(candidate)) continue;
        handle = dlopen(candidate.c_str(), kFlags);
        if (handle == nullptr) {
          throw std::runtime_error("op library " + candidate +
                                   " failed to load: " + dlerror());
        }
        file = name;
        break;
      }
      if (handle != nullptr) break;
    }
  }
  if (handle == nullptr) {
    throw std::runtime_error("the package needs op library '" + library +
                             "' (" + requirement + "); " + files[0] + " / " +
                             files[1] + " was not found in " + looked);
  }

  using RequirementFn = const char* (*)();
  const auto built_for = reinterpret_cast<RequirementFn>(
      dlsym(handle, "molnex_op_library_requirement"));
  if (built_for == nullptr) {
    throw std::runtime_error(file +
                             " does not export molnex_op_library_requirement; "
                             "it is not a molnex op library");
  }
  if (requirement != built_for()) {
    throw std::runtime_error("op library " + file + " was built for " +
                             built_for() +
                             ", but the package was exported "
                             "against " +
                             requirement);
  }
}

}  // namespace molnex::runtime
