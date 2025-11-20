#include "inference_engine.h"

#include <dlfcn.h>
#include <iostream>

namespace cochl {

InferenceEngine::InferenceEngine()
    : lib_handle_(nullptr),
      api_instance_(nullptr),
      class_map_(nullptr),
      CochlApi_Create_(nullptr),
      CochlApi_RunInference_(nullptr),
      CochlApi_GetInputSize_(nullptr),
      CochlApi_GetOutputSize_(nullptr),
      CochlApi_Destroy_(nullptr),
      CochlApi_LoadImage_(nullptr),
      CochlApi_LoadClassNames_(nullptr),
      CochlApi_GetClassName_(nullptr),
      CochlApi_DestroyClassMap_(nullptr) {}

InferenceEngine::~InferenceEngine() {
  // Destroy class map
  if (class_map_ && CochlApi_DestroyClassMap_) {
    CochlApi_DestroyClassMap_(class_map_);
    class_map_ = nullptr;
  }

  // Destroy API instance
  if (api_instance_ && CochlApi_Destroy_) {
    CochlApi_Destroy_(api_instance_);
    api_instance_ = nullptr;
  }

  // Close library
  if (lib_handle_) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
}

bool InferenceEngine::loadLibrary(const std::string& library_path) {
  if (lib_handle_) {
    std::cerr << "[InferenceEngine] Library already loaded" << std::endl;
    return false;
  }

  // Load library with RTLD_LAZY | RTLD_LOCAL
  lib_handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!lib_handle_) {
    std::cerr << "[InferenceEngine] Failed to load library: " << library_path << std::endl;
    std::cerr << "[InferenceEngine] dlerror: " << dlerror() << std::endl;
    return false;
  }

  // Clear any existing error
  dlerror();

  // Load function pointers
  CochlApi_Create_ = reinterpret_cast<void* (*)(const char*)>(
      dlsym(lib_handle_, "CochlApi_Create"));
  if (!CochlApi_Create_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_Create: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_RunInference_ = reinterpret_cast<int (*)(void*, const float*, size_t, float*, size_t)>(
      dlsym(lib_handle_, "CochlApi_RunInference"));
  if (!CochlApi_RunInference_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_RunInference: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_GetInputSize_ = reinterpret_cast<size_t (*)(void*)>(
      dlsym(lib_handle_, "CochlApi_GetInputSize"));
  if (!CochlApi_GetInputSize_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_GetInputSize: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_GetOutputSize_ = reinterpret_cast<size_t (*)(void*)>(
      dlsym(lib_handle_, "CochlApi_GetOutputSize"));
  if (!CochlApi_GetOutputSize_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_GetOutputSize: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_Destroy_ = reinterpret_cast<void (*)(void*)>(
      dlsym(lib_handle_, "CochlApi_Destroy"));
  if (!CochlApi_Destroy_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_Destroy: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  // Load image utility functions
  CochlApi_LoadImage_ = reinterpret_cast<int (*)(const char*, float*, size_t)>(
      dlsym(lib_handle_, "CochlApi_LoadImage"));
  if (!CochlApi_LoadImage_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_LoadImage: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_LoadClassNames_ = reinterpret_cast<void* (*)(const char*)>(
      dlsym(lib_handle_, "CochlApi_LoadClassNames"));
  if (!CochlApi_LoadClassNames_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_LoadClassNames: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_GetClassName_ = reinterpret_cast<const char* (*)(void*, int)>(
      dlsym(lib_handle_, "CochlApi_GetClassName"));
  if (!CochlApi_GetClassName_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_GetClassName: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  CochlApi_DestroyClassMap_ = reinterpret_cast<void (*)(void*)>(
      dlsym(lib_handle_, "CochlApi_DestroyClassMap"));
  if (!CochlApi_DestroyClassMap_) {
    std::cerr << "[InferenceEngine] Failed to load CochlApi_DestroyClassMap: " << dlerror() << std::endl;
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  std::cout << "[InferenceEngine] Library loaded successfully: " << library_path << std::endl;
  return true;
}

bool InferenceEngine::loadModel(const std::string& model_path) {
  if (!lib_handle_) {
    std::cerr << "[InferenceEngine] Error: Library not loaded. Call LoadLibrary() first" << std::endl;
    return false;
  }

  if (api_instance_) {
    std::cerr << "[InferenceEngine] Error: Model already loaded" << std::endl;
    return false;
  }

  if (model_path.empty()) {
    std::cerr << "[InferenceEngine] Error: Empty model path" << std::endl;
    return false;
  }

  // Create API instance
  api_instance_ = CochlApi_Create_(model_path.c_str());
  if (!api_instance_) {
    std::cerr << "[InferenceEngine] Failed to create API instance for: " << model_path << std::endl;
    return false;
  }

  std::cout << "[InferenceEngine] Model loaded successfully: " << model_path << std::endl;
  std::cout << "[InferenceEngine] Input size: " << GetInputSize() << std::endl;
  std::cout << "[InferenceEngine] Output size: " << GetOutputSize() << std::endl;
  return true;
}

InferenceStatus InferenceEngine::runInference(const float* input, size_t input_size,
                                               float* output, size_t output_size) {
  if (!api_instance_) {
    std::cerr << "[InferenceEngine] Error: Model not loaded" << std::endl;
    return InferenceStatus::ERROR_NOT_INITIALIZED;
  }

  if (!input || input_size == 0) {
    std::cerr << "[InferenceEngine] Error: Invalid input" << std::endl;
    return InferenceStatus::ERROR_INVALID_INPUT;
  }

  if (!output || output_size == 0) {
    std::cerr << "[InferenceEngine] Error: Invalid output buffer" << std::endl;
    return InferenceStatus::ERROR_INVALID_INPUT;
  }

  // Run inference via C API
  int result = CochlApi_RunInference_(api_instance_, input, input_size, output, output_size);

  if (result == 0) {
    std::cerr << "[InferenceEngine] Inference failed" << std::endl;
    return InferenceStatus::ERROR_INFERENCE_FAILED;
  }

  return InferenceStatus::OK;
}

size_t InferenceEngine::getInputSize() const {
  if (!api_instance_) {
    return 0;
  }
  return CochlApi_GetInputSize_(api_instance_);
}

size_t InferenceEngine::getOutputSize() const {
  if (!api_instance_) {
    return 0;
  }
  return CochlApi_GetOutputSize_(api_instance_);
}

bool InferenceEngine::loadImage(const std::string& image_path, float* output_data, size_t output_size) {
  if (!lib_handle_) {
    std::cerr << "[InferenceEngine] Error: Library not loaded" << std::endl;
    return false;
  }

  int result = CochlApi_LoadImage_(image_path.c_str(), output_data, output_size);
  return result == 1;
}

bool InferenceEngine::loadClassNames(const std::string& json_path) {
  if (!lib_handle_) {
    std::cerr << "[InferenceEngine] Error: Library not loaded" << std::endl;
    return false;
  }

  if (class_map_) {
    std::cerr << "[InferenceEngine] Warning: Class names already loaded" << std::endl;
    return true;
  }

  class_map_ = CochlApi_LoadClassNames_(json_path.c_str());
  if (!class_map_) {
    std::cerr << "[InferenceEngine] Failed to load class names from: " << json_path << std::endl;
    return false;
  }

  std::cout << "[InferenceEngine] Class names loaded successfully" << std::endl;
  return true;
}

std::string InferenceEngine::getClassName(int class_idx) const {
  if (!class_map_) {
    return "Unknown (class map not loaded)";
  }

  const char* name = CochlApi_GetClassName_(class_map_, class_idx);
  if (name) {
    return std::string(name);
  }

  return "Unknown";
}

}  // namespace cochl
