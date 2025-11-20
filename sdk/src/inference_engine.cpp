#include "inference_engine.h"
#include "error/sdk_error.h"

#include <glog/logging.h>

namespace cochl {

InferenceEngine::InferenceEngine()
    : api_instance_(nullptr),
      class_map_(nullptr) {}

InferenceEngine::~InferenceEngine() {
  // Destroy class map
  if (class_map_ && api_loader_.destroyClassMap) {
    api_loader_.destroyClassMap(class_map_);
    class_map_ = nullptr;
  }

  // Destroy API instance
  if (api_instance_ && api_loader_.destroy) {
    api_loader_.destroy(api_instance_);
    api_instance_ = nullptr;
  }

  // Library is automatically closed by CochlApiLoader destructor
}

bool InferenceEngine::loadLibrary(const std::string& library_path) {
  return api_loader_.load(library_path);
}

bool InferenceEngine::create(const std::string& model_path) {
  if (!api_loader_.isLoaded()) {
    error::printError(error::SdkError::API_NOT_INITIALIZED, "Library not loaded. Call loadLibrary() first");
    return false;
  }

  if (api_instance_) {
    error::printError(error::SdkError::API_ALREADY_CREATED);
    return false;
  }

  if (model_path.empty()) {
    error::printError(error::SdkError::EMPTY_PATH, "model_path");
    return false;
  }

  // Create API instance
  api_instance_ = api_loader_.create(model_path.c_str());
  if (!api_instance_) {
    error::printError(error::SdkError::API_CREATE_FAILED, model_path);
    return false;
  }

  LOG(INFO) << "[InferenceEngine] API instance created successfully with model: " << model_path;
  LOG(INFO) << "[InferenceEngine] Input size: " << getInputSize();
  LOG(INFO) << "[InferenceEngine] Output size: " << getOutputSize();
  return true;
}

bool InferenceEngine::runInference(const float* input, const std::vector<int64_t>& input_shape,
                                    float* output, TensorLayout layout) {
  if (!api_instance_) {
    error::printError(error::SdkError::API_NOT_INITIALIZED, "Model not loaded");
    return false;
  }

  if (!input) {
    error::printError(error::SdkError::INVALID_INPUT_DATA);
    return false;
  }

  if (!output) {
    error::printError(error::SdkError::INVALID_OUTPUT_DATA);
    return false;
  }

  if (input_shape.empty()) {
    error::printError(error::SdkError::INVALID_INPUT_DATA, "Input shape is empty");
    return false;
  }

  // Convert TensorLayout enum to int for C API
  int layout_int = static_cast<int>(layout);

  // Run inference via C API
  // Cast int64_t* to long long* for C API compatibility
  int result = api_loader_.runInference(api_instance_, input,
                                        reinterpret_cast<const long long*>(input_shape.data()),
                                        input_shape.size(),
                                        output, layout_int);

  if (result == 0) {
    error::printError(error::SdkError::INFERENCE_FAILED);
    return false;
  }

  return true;
}

size_t InferenceEngine::getInputSize() const {
  if (!api_instance_) {
    return 0;
  }
  return api_loader_.getInputSize(api_instance_);
}

size_t InferenceEngine::getOutputSize() const {
  if (!api_instance_) {
    return 0;
  }
  return api_loader_.getOutputSize(api_instance_);
}

bool InferenceEngine::loadImage(const std::string& image_path, float* output_data, size_t output_size) {
  if (!api_loader_.isLoaded()) {
    error::printError(error::SdkError::API_NOT_INITIALIZED, "Library not loaded");
    return false;
  }

  int result = api_loader_.loadImage(image_path.c_str(), output_data, output_size);
  if (result != 1) {
    error::printError(error::SdkError::IMAGE_LOAD_FAILED, image_path);
    return false;
  }
  return true;
}

bool InferenceEngine::loadClassNames(const std::string& json_path) {
  if (!api_loader_.isLoaded()) {
    error::printError(error::SdkError::API_NOT_INITIALIZED, "Library not loaded");
    return false;
  }

  if (class_map_) {
    LOG(WARNING) << "Class names already loaded";
    return true;
  }

  class_map_ = api_loader_.loadClassNames(json_path.c_str());
  if (!class_map_) {
    error::printError(error::SdkError::CLASS_NAMES_LOAD_FAILED, json_path);
    return false;
  }

  LOG(INFO) << "[InferenceEngine] Class names loaded successfully";
  return true;
}

std::string InferenceEngine::getClassName(int class_idx) const {
  if (!class_map_) {
    return "Unknown (class map not loaded)";
  }

  const char* name = api_loader_.getClassName(class_map_, class_idx);
  if (name) {
    return std::string(name);
  }

  return "Unknown";
}

}  // namespace cochl
