#include "runtime/runtime_manager.h"

#include <algorithm>
#include <iostream>

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

std::unique_ptr<RuntimeManager> RuntimeManager::Create(const std::string& model_path) {
  if (model_path.empty()) {
    std::cerr << "[RuntimeManager] Error: Empty model path" << std::endl;
    return nullptr;
  }

  auto manager = std::unique_ptr<RuntimeManager>(new RuntimeManager());

  // Detect runtime type from file extension
  InferenceEngine type = DetectInferenceEngine(model_path);

  if (type == InferenceEngine::UNKNOWN) {
    std::cerr << "[RuntimeManager] Error: Unsupported model format: " << model_path
              << std::endl;
    return nullptr;
  }

  // Load model with detected runtime
  if (!manager->LoadModel(model_path, type)) {
    std::cerr << "[RuntimeManager] Error: Failed to load model: " << model_path
              << std::endl;
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

RuntimeManager::InferenceEngine RuntimeManager::DetectInferenceEngine(
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
    std::cerr << "[RuntimeManager] TFLite runtime not compiled in" << std::endl;
    return InferenceEngine::UNKNOWN;
#endif
  } else if (extension == "pt" || extension == "pth") {
#ifdef USE_LIBTORCH
    return InferenceEngine::LIBTORCH;
#else
    std::cerr << "[RuntimeManager] LibTorch runtime not compiled in" << std::endl;
    return InferenceEngine::UNKNOWN;
#endif
  } else if (extension == "bin") {
#ifdef USE_CUSTOM
    return InferenceEngine::CUSTOM;
#else
    std::cerr << "[RuntimeManager] Custom runtime not compiled in" << std::endl;
    return InferenceEngine::UNKNOWN;
#endif
  }

  return InferenceEngine::UNKNOWN;
}

bool RuntimeManager::LoadModel(const std::string& model_path, InferenceEngine type) {
  if (initialized_) {
    std::cerr << "[RuntimeManager] Error: Runtime already initialized" << std::endl;
    return false;
  }

  runtime_type_ = type;

  switch (type) {
#ifdef USE_TFLITE
    case InferenceEngine::TFLITE: {
      auto tf_runtime = std::make_unique<TFRuntime>();
      if (!tf_runtime->LoadModel(model_path.c_str())) {
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
      if (!torch_runtime->LoadModel(model_path.c_str())) {
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
      if (!custom_runtime->LoadModel(model_path.c_str())) {
        return false;
      }
      runtime_ = std::move(custom_runtime);
      initialized_ = true;
      return true;
    }
#endif

    default:
      std::cerr << "[RuntimeManager] Unsupported runtime type" << std::endl;
      return false;
  }
}

bool RuntimeManager::RunInference(const float* input, size_t input_size, float* output,
                                   size_t output_size) const {
  if (!runtime_) {
    std::cerr << "[RuntimeManager] No runtime loaded" << std::endl;
    return false;
  }

  return runtime_->RunInference(input, input_size, output, output_size);
}

size_t RuntimeManager::GetInputSize() const {
  if (!runtime_) {
    std::cerr << "[RuntimeManager] No runtime loaded" << std::endl;
    return 0;
  }

  return runtime_->GetInputSize();
}

size_t RuntimeManager::GetOutputSize() const {
  if (!runtime_) {
    std::cerr << "[RuntimeManager] No runtime loaded" << std::endl;
    return 0;
  }

  return runtime_->GetOutputSize();
}

}  // namespace runtime
}  // namespace cochl_api
