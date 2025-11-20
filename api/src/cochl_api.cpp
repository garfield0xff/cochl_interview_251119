// Implementation of CochlApi using RuntimeManager for multi-runtime support.
// Automatically selects runtime based on model file extension:
// - .tflite -> TensorFlow Lite
// - .pt, .pth -> LibTorch

#include "cochl_api.h"

#include <glog/logging.h>
#include <memory>
#include <string>

#include "runtime/runtime_manager.h"
#include "error/api_error.h"

namespace external_api {
std::unique_ptr<CochlApi> CochlApi::create(const std::string& model_path) {
  LOG(INFO) << "[CochlApi] Loading model from: " << model_path;

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

bool CochlApi::runInference(const float* input, const std::vector<int64_t>& input_shape,
                            float* output) const {
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

  if (input_shape.empty()) {
    cochl_api::error::printError(cochl_api::error::ApiError::INVALID_INPUT_SIZE, "Empty input shape");
    return false;
  }

  return runtime_manager_->runInference(input, input_shape, output);
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
