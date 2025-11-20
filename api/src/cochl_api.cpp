// Implementation of CochlApi using RuntimeManager for multi-runtime support.
// Automatically selects runtime based on model file extension:
// - .tflite -> TensorFlow Lite
// - .pt, .pth -> LibTorch

#include "cochl_api.h"

#include <iostream>
#include <memory>
#include <string>

#include "runtime/runtime_manager.h"
#include "error/api_error.h"

namespace external_api {
std::unique_ptr<CochlApi> CochlApi::create(const std::string& model_path) {
  std::cout << "[CochlApi] Loading model from: " << model_path << std::endl;

  if (model_path.empty()) {
    cochl_api::error::printError(cochl_api::error::ApiError::EMPTY_PATH);
    return nullptr;
  }

  auto api = std::unique_ptr<CochlApi>(new CochlApi());

  api->runtime_manager_ = cochl_api::runtime::RuntimeManager::create(model_path);

  if (!api->runtime_manager_) {
    cochl_api::error::printError(cochl_api::error::ApiError::RUNTIME_CREATION_FAILED);
    return nullptr;
  }

  return api;
}

CochlApi::CochlApi() = default;
CochlApi::~CochlApi() = default;

bool CochlApi::runInference(const float* input, size_t input_size, float* output,
                            size_t output_size) const {
  if (!runtime_manager_) {
    cochl_api::error::printError(cochl_api::error::ApiError::RUNTIME_NOT_INITIALIZED);
    return false;
  }

  if (!input) {
    cochl_api::error::printError(cochl_api::error::ApiError::INVALID_INPUT_DATA);
    return false;
  }

  if (!output) {
    cochl_api::error::printError(cochl_api::error::ApiError::INVALID_OUTPUT_DATA);
    return false;
  }

  if (input_size == 0 || input_size > getInputSize()) {
    cochl_api::error::printError(cochl_api::error::ApiError::INVALID_INPUT_SIZE,
                                   std::to_string(input_size));
    return false;
  }

  if (output_size != getOutputSize()) {
    cochl_api::error::printError(cochl_api::error::ApiError::INVALID_OUTPUT_SIZE,
                                   std::to_string(output_size));
    return false;
  }

  return runtime_manager_->runInference(input, input_size, output, output_size);
}

size_t CochlApi::getInputSize() const {
  if (!runtime_manager_) {
    cochl_api::error::printError(cochl_api::error::ApiError::RUNTIME_NOT_INITIALIZED);
    return 0;
  }
  return runtime_manager_->getInputSize();
}

size_t CochlApi::getOutputSize() const {
  if (!runtime_manager_) {
    cochl_api::error::printError(cochl_api::error::ApiError::RUNTIME_NOT_INITIALIZED);
    return 0;
  }
  return runtime_manager_->getOutputSize();
}

}  // namespace external_api
