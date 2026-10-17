// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See aoti_artifact.h. Every call delegates to
// torch::inductor::AOTIModelPackageLoader.

#include "molnex/runtime/aoti_artifact.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/api/include/torch/serialize.h>

namespace molnex::runtime {
namespace {

/// The package metadata spells the device `"cpu"`, `"cuda"` or `"cuda:N"`.
at::Device device_named(const std::string& raw) {
  if (raw == "cpu") return at::Device(at::kCPU);
  if (raw == "cuda" || raw.rfind("cuda:", 0) == 0) return at::Device(at::kCUDA);
  throw std::runtime_error("unsupported AOTI device '" + raw + "'");
}

/// Load the package, refusing the argument combinations the loader rejects
/// with a less specific message.
std::unique_ptr<torch::inductor::AOTIModelPackageLoader> load_package(
    const std::string& pt2_path, int num_models, bool run_single_threaded) {
  if (!std::filesystem::is_regular_file(pt2_path)) {
    throw std::runtime_error("missing AOTI package: " + pt2_path);
  }
  if (num_models < 1) throw std::runtime_error("num_models must be >= 1");
  if (num_models > 1 && run_single_threaded) {
    throw std::runtime_error(
        "AotiArtifact: num_models > 1 requires run_single_threaded=false "
        "(the AOTI loader rejects that combination)");
  }
  // The archive model name is always "model"; the file stem is the caller's.
  return std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      pt2_path, /*model_name=*/"model", run_single_threaded,
      /*num_runners=*/static_cast<size_t>(num_models),
      /*device_index=*/-1);
}

}  // namespace

AotiArtifact::AotiArtifact(const std::string& pt2_path, int num_models,
                           bool run_single_threaded)
    : device_(at::kCPU),
      loader_(load_package(pt2_path, num_models, run_single_threaded)) {
  const std::unordered_map<std::string, std::string> meta =
      loader_->get_metadata();
  const auto device_it = meta.find("AOTI_DEVICE_KEY");
  if (device_it == meta.end()) {
    throw std::runtime_error("AOTI package metadata has no AOTI_DEVICE_KEY");
  }
  device_ = device_named(device_it->second);
#ifndef MOLNEX_RUNTIME_CUDA
  if (device_.is_cuda()) {
    throw std::runtime_error(
        "AOTI package device=cuda, but molnex_runtime was built CPU-only (no "
        "CUDA toolkit at build time); rebuild where CUDA is available");
  }
#endif
}

AotiArtifact::~AotiArtifact() = default;

std::vector<at::Tensor> AotiArtifact::run(
    const std::vector<at::Tensor>& inputs) {
  std::shared_lock<std::shared_mutex> guard(reload_mutex_);
  return loader_->run(inputs);
}

void AotiArtifact::update_weights(const std::string& weight_path) {
  std::ifstream is(weight_path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + weight_path);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                                std::istreambuf_iterator<char>());
  const c10::IValue iv = torch::pickle_load(bytes);
  if (!iv.isGenericDict()) {
    throw std::runtime_error("weight file is not a state_dict-like dict: " +
                             weight_path);
  }
  std::unordered_map<std::string, at::Tensor> params;
  for (const auto& kv : iv.toGenericDict()) {
    if (!kv.key().isString() || !kv.value().isTensor()) {
      throw std::runtime_error("weight file entries must be (str -> Tensor): " +
                               weight_path);
    }
    params.emplace(kv.key().toStringRef(), kv.value().toTensor());
  }
  update_weights(params);
}

void AotiArtifact::update_weights(
    const std::unordered_map<std::string, at::Tensor>& params) {
  // Exclusive: drains in-flight run() holders before the active buffer is
  // rewritten. load_constants takes a non-const map and translates state-dict
  // FQNs itself; allow_h2d_copy lets a CPU state_dict land on a CUDA package.
  std::unique_lock<std::shared_mutex> guard(reload_mutex_);
  std::unordered_map<std::string, at::Tensor> constants = params;
  loader_->load_constants(constants, /*use_inactive=*/false,
                          /*check_full_update=*/false, /*user_managed=*/false,
                          /*allow_h2d_copy=*/true);
}

std::vector<std::string> AotiArtifact::constant_names() const {
  return loader_->get_constant_fqns();
}

}  // namespace molnex::runtime
