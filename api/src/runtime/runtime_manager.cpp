#include "runtime/runtime_manager.h"

#include <algorithm>
#include <iostream>

#ifdef USE_TFLITE
#include "runtime/tf_runtime.h"
#endif

#ifdef USE_LIBTORCH
#include "runtime/torch_runtime.h"
#endif

namespace cochl_api {
namespace runtime {

RuntimeManager::RuntimeManager() : runtime_type_(RuntimeType::UNKNOWN) {}

RuntimeManager::~RuntimeManager() = default;

std::unique_ptr<RuntimeManager> RuntimeManager::Create(const std::string& model_path) {
  if (model_path.empty()) {
    std::cerr << "[RuntimeManager] Error: Empty model path" << std::endl;
    return nullptr;
  }

  auto manager = std::unique_ptr<RuntimeManager>(new RuntimeManager());

  // Detect runtime type from file extension
  RuntimeType type = DetectRuntimeType(model_path);

  if (type == RuntimeType::UNKNOWN) {
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

  std::cout << "[RuntimeManager] Successfully loaded model with "
            << manager->GetRuntimeTypeName() << " runtime" << std::endl;

  return manager;
}

RuntimeManager::RuntimeType RuntimeManager::DetectRuntimeType(
    const std::string& model_path) {
  // Find last dot for extension
  size_t dot_pos = model_path.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return RuntimeType::UNKNOWN;
  }

  std::string extension = model_path.substr(dot_pos + 1);

  // Convert to lowercase for case-insensitive comparison
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  // Match extension to runtime
  if (extension == "tflite") {
#ifdef USE_TFLITE
    return RuntimeType::TFLITE;
#else
    std::cerr << "[RuntimeManager] TFLite runtime not compiled in" << std::endl;
    return RuntimeType::UNKNOWN;
#endif
  } else if (extension == "pt" || extension == "pth") {
#ifdef USE_LIBTORCH
    return RuntimeType::LIBTORCH;
#else
    std::cerr << "[RuntimeManager] LibTorch runtime not compiled in" << std::endl;
    return RuntimeType::UNKNOWN;
#endif
  }

  return RuntimeType::UNKNOWN;
}

bool RuntimeManager::LoadModel(const std::string& model_path, RuntimeType type) {
  runtime_type_ = type;

  switch (type) {
#ifdef USE_TFLITE
    case RuntimeType::TFLITE: {
      auto tf_runtime = std::make_unique<TFRuntime>();
      if (!tf_runtime->LoadModel(model_path.c_str())) {
        return false;
      }
      runtime_ = std::move(tf_runtime);
      return true;
    }
#endif

#ifdef USE_LIBTORCH
    case RuntimeType::LIBTORCH: {
      auto torch_runtime = std::make_unique<TorchRuntime>();
      if (!torch_runtime->LoadModel(model_path.c_str())) {
        return false;
      }
      runtime_ = std::move(torch_runtime);
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

const char* RuntimeManager::GetRuntimeTypeName() const {
  if (runtime_) {
    return runtime_->GetRuntimeType();
  }

  switch (runtime_type_) {
    case RuntimeType::TFLITE:
      return "TensorFlow Lite (not loaded)";
    case RuntimeType::LIBTORCH:
      return "LibTorch (not loaded)";
    default:
      return "Unknown";
  }
}

}  // namespace runtime
}  // namespace cochl_api
