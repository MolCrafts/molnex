// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `model_runner.h` for the design notes. This .cpp keeps the
// implementation in one file deliberately: the public surface is
// small, all heavy lifting delegates to torch::inductor, and there is
// no reason to scatter helpers across translation units.

#include "molnex/interface/model_runner.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace fs = std::filesystem;

namespace molnex::interface {
namespace {

// Tiny "find one string field" JSON reader. Sufficient for meta.json
// — we only need the `device` field at load time, and the rest of the
// schema is informational. Avoids pulling in a real JSON dependency.
std::string parse_device_field(const std::string& json_text) {
  const std::string key = "\"device\"";
  auto k = json_text.find(key);
  if (k == std::string::npos) {
    throw std::runtime_error("meta.json: missing 'device' field");
  }
  auto colon = json_text.find(':', k + key.size());
  if (colon == std::string::npos) {
    throw std::runtime_error("meta.json: malformed near 'device'");
  }
  auto open_q = json_text.find('"', colon + 1);
  if (open_q == std::string::npos) {
    throw std::runtime_error("meta.json: 'device' value not a string");
  }
  auto close_q = json_text.find('"', open_q + 1);
  if (close_q == std::string::npos) {
    throw std::runtime_error("meta.json: 'device' value not terminated");
  }
  return json_text.substr(open_q + 1, close_q - open_q - 1);
}

std::string read_file(const fs::path& path) {
  std::ifstream is(path);
  if (!is) throw std::runtime_error("cannot open " + path.string());
  std::stringstream ss;
  ss << is.rdbuf();
  return ss.str();
}

std::vector<char> read_binary(const fs::path& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + path.string());
  return std::vector<char>((std::istreambuf_iterator<char>(is)),
                           std::istreambuf_iterator<char>());
}

const char* scalar_type_name(c10::ScalarType st) {
  return c10::toString(st);
}

}  // namespace

ModelRunner::ModelRunner(const std::string& model_dir,
                         int num_models,
                         const std::string& name) {
  fs::path dir(model_dir);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model_dir does not exist: " + model_dir);
  }
  fs::path so_path = dir / (name + ".so");
  fs::path meta_path = dir / (name + ".meta.json");
  if (!fs::exists(so_path)) {
    throw std::runtime_error("missing artifact: " + so_path.string());
  }
  if (!fs::exists(meta_path)) {
    throw std::runtime_error("missing artifact: " + meta_path.string());
  }
  if (num_models < 1) {
    throw std::runtime_error("num_models must be >= 1");
  }

  device_ = parse_device_field(read_file(meta_path));
  if (device_ != "cpu" && device_ != "cuda") {
    throw std::runtime_error("unsupported device in meta.json: " + device_);
  }

  if (device_ == "cuda") {
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        so_path.string(),
        static_cast<size_t>(num_models),
        /*device_str=*/"cuda",
        /*cubin_dir=*/"",
        /*run_single_threaded=*/false);
  } else {
    runner_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        so_path.string(),
        static_cast<size_t>(num_models),
        /*run_single_threaded=*/false);
  }
}

ModelRunner::~ModelRunner() = default;

std::vector<at::Tensor> ModelRunner::run(const std::vector<at::Tensor>& inputs) {
  return runner_->run(inputs);
}

std::vector<at::Tensor> ModelRunner::run_async(
    const std::vector<at::Tensor>& inputs,
    at::cuda::CUDAStream stream) {
  if (device_ != "cuda") {
    throw std::runtime_error("run_async only supported on CUDA runners");
  }
  auto* cuda_runner =
      dynamic_cast<torch::inductor::AOTIModelContainerRunnerCuda*>(runner_.get());
  if (cuda_runner == nullptr) {
    // Should be unreachable given device_ == "cuda", but keep the
    // invariant honest.
    throw std::runtime_error("internal: CUDA device but non-CUDA runner");
  }
  return cuda_runner->run_with_cuda_stream(inputs, stream);
}

void ModelRunner::update_weights(const std::string& weight_path) {
  auto bytes = read_binary(weight_path);
  auto iv = torch::pickle_load(bytes);
  if (!iv.isGenericDict()) {
    throw std::runtime_error("weight file is not a state_dict-like dict: " +
                             weight_path);
  }
  std::unordered_map<std::string, at::Tensor> params;
  for (const auto& kv : iv.toGenericDict()) {
    if (!kv.key().isString() || !kv.value().isTensor()) {
      throw std::runtime_error(
          "weight file entries must be (str -> Tensor): " + weight_path);
    }
    params.emplace(kv.key().toStringRef(), kv.value().toTensor());
  }
  update_weights(params);
}

void ModelRunner::update_weights(
    const std::unordered_map<std::string, at::Tensor>& params) {
  std::lock_guard<std::mutex> guard(reload_mutex_);

  // Translate FQN-keyed user map into the internal-name-keyed map the
  // AOTI runner expects. getConstantNamesToOriginalFQNs() returns
  // {internal_name: original_FQN} — invert it for lookup.
  auto fqn_map = runner_->getConstantNamesToOriginalFQNs();
  std::unordered_map<std::string, std::string> fqn_to_internal;
  fqn_to_internal.reserve(fqn_map.size());
  for (const auto& [internal, fqn] : fqn_map) {
    fqn_to_internal.emplace(fqn, internal);
  }

  // Hold ownership of any moved/.to()-converted tensors so the
  // pointers we hand the runner stay valid for the call.
  std::vector<at::Tensor> owned;
  owned.reserve(params.size());
  torch::inductor::TensorConstantMap const_map;
  const_map.reserve(params.size());

  const auto target_device =
      device_ == "cuda" ? at::Device(at::kCUDA) : at::Device(at::kCPU);

  for (const auto& [user_key, tensor] : params) {
    auto it = fqn_to_internal.find(user_key);
    if (it == fqn_to_internal.end()) {
      // Fall back to assuming the user already passed the internal
      // name — covers callers that read parameter_info() directly.
      if (fqn_map.find(user_key) == fqn_map.end()) {
        throw std::runtime_error(
            "update_weights: unknown parameter '" + user_key +
            "' (not in model's constants)");
      }
      // user_key is already an internal name
      auto& placed = owned.emplace_back(tensor.to(target_device));
      const_map.emplace(user_key, &placed);
    } else {
      auto& placed = owned.emplace_back(tensor.to(target_device));
      const_map.emplace(it->second, &placed);
    }
  }

  // Double-buffered swap: rewrite the inactive buffer, then flip.
  // The active buffer keeps serving run() calls until swap returns.
  runner_->update_inactive_constant_buffer(const_map);
  runner_->swap_constant_buffer();
}

std::vector<std::pair<std::string, std::string>>
ModelRunner::parameter_info() const {
  auto dtypes = runner_->getConstantNamesToDtypes();
  auto fqn_map = runner_->getConstantNamesToOriginalFQNs();
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(dtypes.size());
  for (const auto& [internal, dtype_int] : dtypes) {
    std::string name = internal;
    auto it = fqn_map.find(internal);
    if (it != fqn_map.end() && !it->second.empty()) {
      name = it->second;
    }
    out.emplace_back(
        std::move(name),
        scalar_type_name(static_cast<c10::ScalarType>(dtype_int)));
  }
  return out;
}

}  // namespace molnex::interface
