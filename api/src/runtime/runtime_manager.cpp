#include "runtime/runtime_manager.h"

#include <algorithm>
#include <iostream>

#include "error/api_error.h"

#ifdef USE_TFLITE
#include "runtime/tf_runtime.h"
#endif

#ifdef USE_LIBTORCH
#include "runtime/torch_runtime.h"
#endif

#ifdef USE_CUSTOM
#include "runtime/custom_runtime.h"
#endif

namespace cochl_api {
namespace runtime {

RuntimeManager::RuntimeManager() : runtime_type_(InferenceEngine::UNKNOWN), initialized_(false) {}

RuntimeManager::~RuntimeManager() = default;

std::unique_ptr<RuntimeManager> RuntimeManager::create(const std::string& model_path) {
  if (model_path.empty()) {
    error::printError(error::ApiError::EMPTY_PATH);
    return nullptr;
  }

  auto manager = std::unique_ptr<RuntimeManager>(new RuntimeManager());

  // Detect runtime type from file extension
  InferenceEngine type = detectInferenceEngine(model_path);

  if (type == InferenceEngine::UNKNOWN) {
    error::printError(error::ApiError::MODEL_INVALID_FORMAT, model_path);
    return nullptr;
  }

  // Load model with detected runtime
  if (!manager->loadModel(model_path, type)) {
    error::printError(error::ApiError::MODEL_LOAD_FAILED, model_path);
    return nullptr;
  }

  const char* runtime_name = "Unknown";
  switch (manager->runtime_type_) {
    case InferenceEngine::TFLITE:
      runtime_name = "TensorFlow Lite";
      break;
    case InferenceEngine::LIBTORCH:
      runtime_name = "LibTorch";
      break;
    case InferenceEngine::CUSTOM:
      runtime_name = "Custom Backend (Thread Pool)";
      break;
    default:
      break;
  }
  std::cout << "[RuntimeManager] Successfully loaded model with "
            << runtime_name << " runtime" << std::endl;

  return manager;
}

RuntimeManager::InferenceEngine RuntimeManager::detectInferenceEngine(
    const std::string& model_path) {

  size_t dot_pos = model_path.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return InferenceEngine::UNKNOWN;
  }

  std::string extension = model_path.substr(dot_pos + 1);

  // Convert to lowercase for case-insensitive comparison
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  // Match extension to runtime
  if (extension == "tflite") {
#ifdef USE_TFLITE
    return InferenceEngine::TFLITE;
#else
    error::printError(error::ApiError::RUNTIME_NOT_SUPPORTED, "TFLite");
    return InferenceEngine::UNKNOWN;
#endif
  } else if (extension == "pt" || extension == "pth") {
#ifdef USE_LIBTORCH
    return InferenceEngine::LIBTORCH;
#else
    error::printError(error::ApiError::RUNTIME_NOT_SUPPORTED, "LibTorch");
    return InferenceEngine::UNKNOWN;
#endif
  } else if (extension == "bin") {
#ifdef USE_CUSTOM
    return InferenceEngine::CUSTOM;
#else
    error::printError(error::ApiError::RUNTIME_NOT_SUPPORTED, "Custom");
    return InferenceEngine::UNKNOWN;
#endif
  }

  return InferenceEngine::UNKNOWN;
}

bool RuntimeManager::loadModel(const std::string& model_path, InferenceEngine type) {
  if (initialized_) {
    error::printError(error::ApiError::MODEL_ALREADY_LOADED);
    return false;
  }

  runtime_type_ = type;

  switch (type) {
#ifdef USE_TFLITE
    case InferenceEngine::TFLITE: {
      auto tf_runtime = std::make_unique<TFRuntime>();
      if (!tf_runtime->loadModel(model_path.c_str())) {
        error::printError(error::ApiError::MODEL_LOAD_FAILED, "TFLite runtime");
        return false;
      }
      runtime_ = std::move(tf_runtime);
      initialized_ = true;
      return true;
    }
#endif

#ifdef USE_LIBTORCH
    case InferenceEngine::LIBTORCH: {
      auto torch_runtime = std::make_unique<TorchRuntime>();
      if (!torch_runtime->loadModel(model_path.c_str())) {
        error::printError(error::ApiError::MODEL_LOAD_FAILED, "LibTorch runtime");
        return false;
      }
      runtime_ = std::move(torch_runtime);
      initialized_ = true;
      return true;
    }
#endif

#ifdef USE_CUSTOM
    case InferenceEngine::CUSTOM: {
      auto custom_runtime = std::make_unique<CustomRuntime>();
      if (!custom_runtime->loadModel(model_path.c_str())) {
        error::printError(error::ApiError::MODEL_LOAD_FAILED, "Custom runtime");
        return false;
      }
      runtime_ = std::move(custom_runtime);
      initialized_ = true;
      return true;
    }
#endif

    default:
      error::printError(error::ApiError::RUNTIME_NOT_SUPPORTED);
      return false;
  }
}

bool RuntimeManager::runInference(const float* input, size_t input_size, float* output,
                                   size_t output_size) const {
  if (!runtime_) {
    error::printError(error::ApiError::RUNTIME_NOT_INITIALIZED);
    return false;
  }

  return runtime_->runInference(input, input_size, output, output_size);
}

size_t RuntimeManager::getInputSize() const {
  if (!runtime_) {
    error::printError(error::ApiError::RUNTIME_NOT_INITIALIZED);
    return 0;
  }

  return runtime_->getInputSize();
}

size_t RuntimeManager::getOutputSize() const {
  if (!runtime_) {
    error::printError(error::ApiError::RUNTIME_NOT_INITIALIZED);
    return 0;
  }

  return runtime_->getOutputSize();
}

}  // namespace runtime
}  // namespace cochl_api
