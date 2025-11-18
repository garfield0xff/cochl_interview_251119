// Implementation of CochlApi using RuntimeManager for multi-runtime support.
// Automatically selects runtime based on model file extension:
// - .tflite -> TensorFlow Lite
// - .pt, .pth -> LibTorch

#include "cochl_api.h"

#include <iostream>
#include <memory>
#include <string>

#include "runtime/runtime_manager.h"

namespace external_api {
std::unique_ptr<CochlApi> CochlApi::Create(const std::string& model_path) {
  std::cout << "[CochlApi] Loading model from: " << model_path << std::endl;

  if (model_path.empty()) {
    std::cerr << "[CochlApi] Error: Empty model path" << std::endl;
    return nullptr;
  }

  auto api = std::unique_ptr<CochlApi>(new CochlApi());

  // Create runtime manager (automatically selects runtime based on file extension)
  api->runtime_manager_ = cochl_api::runtime::RuntimeManager::Create(model_path);

  if (!api->runtime_manager_) {
    std::cerr << "[CochlApi] Failed to create runtime manager" << std::endl;
    return nullptr;
  }

  std::cout << "[CochlApi] Model loaded with "
            << api->runtime_manager_->GetRuntimeTypeName() << std::endl;

  return api;
}

CochlApi::CochlApi() = default;

bool CochlApi::RunInference(const float* input, size_t input_size, float* output,
                            size_t output_size) const {
  if (!runtime_manager_) {
    std::cerr << "[CochlApi] Runtime manager not initialized" << std::endl;
    return false;
  }

  if (!input || !output) {
    std::cerr << "[CochlApi] Invalid input or output pointer" << std::endl;
    return false;
  }

  // Validate input/output sizes
  if (input_size == 0 || input_size > GetInputSize()) {
    std::cerr << "[CochlApi] Invalid input size: " << input_size << std::endl;
    return false;
  }

  if (output_size != GetOutputSize()) {
    std::cerr << "[CochlApi] Invalid output size: " << output_size << std::endl;
    return false;
  }

  // Run inference using the selected runtime
  return runtime_manager_->RunInference(input, input_size, output, output_size);
}
}  // namespace external_api
