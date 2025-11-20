#include "error/sdk_error.h"
#include <glog/logging.h>

namespace cochl {
namespace error {

const char* toString(SdkError error) {
  switch (error) {
    case SdkError::SUCCESS:
      return "Success";
    case SdkError::UNKNOWN_ERROR:
      return "Unknown error";

    // Library errors
    case SdkError::LIBRARY_NOT_FOUND:
      return "Library file not found";
    case SdkError::LIBRARY_LOAD_FAILED:
      return "Failed to load library";
    case SdkError::LIBRARY_SYMBOL_NOT_FOUND:
      return "Symbol not found in library";
    case SdkError::LIBRARY_ALREADY_LOADED:
      return "Library already loaded";

    // API instance errors
    case SdkError::API_NOT_INITIALIZED:
      return "API instance not initialized";
    case SdkError::API_CREATE_FAILED:
      return "Failed to create API instance";
    case SdkError::API_ALREADY_CREATED:
      return "API instance already created";

    // Inference errors
    case SdkError::INFERENCE_FAILED:
      return "Inference execution failed";
    case SdkError::INVALID_INPUT_SIZE:
      return "Invalid input size";
    case SdkError::INVALID_OUTPUT_SIZE:
      return "Invalid output size";

    // Parameter errors
    case SdkError::INVALID_INPUT_DATA:
      return "Invalid input data pointer";
    case SdkError::INVALID_OUTPUT_DATA:
      return "Invalid output data pointer";
    case SdkError::INVALID_PARAMETER:
      return "Invalid parameter";
    case SdkError::NULL_POINTER:
      return "Null pointer provided";
    case SdkError::EMPTY_PATH:
      return "Empty path provided";

    // Image processing errors
    case SdkError::IMAGE_LOAD_FAILED:
      return "Failed to load image";
    case SdkError::IMAGE_INVALID_FORMAT:
      return "Invalid image format";
    case SdkError::IMAGE_PROCESSING_FAILED:
      return "Image processing failed";

    // Class name errors
    case SdkError::CLASS_NAMES_LOAD_FAILED:
      return "Failed to load class names";
    case SdkError::CLASS_NAMES_NOT_LOADED:
      return "Class names not loaded";
    case SdkError::INVALID_CLASS_INDEX:
      return "Invalid class index";

    // Resource errors
    case SdkError::OUT_OF_MEMORY:
      return "Out of memory";
    case SdkError::RESOURCE_EXHAUSTED:
      return "Resource exhausted";

    default:
      return "Unknown error code";
  }
}

void printError(SdkError error) {
  LOG(ERROR) << "[SDK ERROR] " << toString(error);
}

void printError(SdkError error, const std::string& context) {
  LOG(ERROR) << "[SDK ERROR] " << toString(error) << ": " << context;
}

}  // namespace error
}  // namespace cochl
