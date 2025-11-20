#include "error/api_error.h"

#include <iostream>

namespace cochl_api {
namespace error {

const char* toString(ApiError error) {
  switch (error) {
    case ApiError::SUCCESS:
      return "Success";
    case ApiError::UNKNOWN_ERROR:
      return "Unknown error";

    // Model loading errors
    case ApiError::MODEL_NOT_FOUND:
      return "Model file not found";
    case ApiError::MODEL_LOAD_FAILED:
      return "Failed to load model";
    case ApiError::MODEL_INVALID_FORMAT:
      return "Invalid or unsupported model format";
    case ApiError::MODEL_ALREADY_LOADED:
      return "Model is already loaded";

    // Runtime errors
    case ApiError::RUNTIME_NOT_INITIALIZED:
      return "Runtime not initialized";
    case ApiError::RUNTIME_NOT_SUPPORTED:
      return "Runtime not supported or not compiled";
    case ApiError::RUNTIME_CREATION_FAILED:
      return "Failed to create runtime instance";

    // Inference errors
    case ApiError::INFERENCE_FAILED:
      return "Inference execution failed";
    case ApiError::INVALID_INPUT_SIZE:
      return "Invalid input size";
    case ApiError::INVALID_OUTPUT_SIZE:
      return "Invalid output size";
    case ApiError::INVALID_INPUT_DATA:
      return "Invalid input data pointer";
    case ApiError::INVALID_OUTPUT_DATA:
      return "Invalid output data pointer";

    // Parameter errors
    case ApiError::INVALID_PARAMETER:
      return "Invalid parameter";
    case ApiError::NULL_POINTER:
      return "Null pointer provided";
    case ApiError::EMPTY_PATH:
      return "Empty file path";

    // Resource errors
    case ApiError::OUT_OF_MEMORY:
      return "Out of memory";
    case ApiError::RESOURCE_EXHAUSTED:
      return "System resources exhausted";

    default:
      return "Unknown error";
  }
}

void printError(ApiError error) {
  std::cerr << "[API ERROR] " << toString(error) << std::endl;
}

void printError(ApiError error, const std::string& context) {
  std::cerr << "[API ERROR] " << toString(error) << ": " << context << std::endl;
}

}  // namespace error
}  // namespace cochl_api
