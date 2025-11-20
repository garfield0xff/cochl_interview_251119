// SDK error handling system
// Provides error codes and error printing functionality for SDK layer

#pragma once

#include <string>

namespace cochl {
namespace error {

enum class SdkError {
  SUCCESS = 0,
  UNKNOWN_ERROR,

  // Library errors
  LIBRARY_NOT_FOUND,
  LIBRARY_LOAD_FAILED,
  LIBRARY_SYMBOL_NOT_FOUND,
  LIBRARY_ALREADY_LOADED,

  // API instance errors
  API_NOT_INITIALIZED,
  API_CREATE_FAILED,
  API_ALREADY_CREATED,

  // Inference errors
  INFERENCE_FAILED,
  INVALID_INPUT_SIZE,
  INVALID_OUTPUT_SIZE,

  // Parameter errors
  INVALID_INPUT_DATA,
  INVALID_OUTPUT_DATA,
  INVALID_PARAMETER,
  NULL_POINTER,
  EMPTY_PATH,

  // Image processing errors
  IMAGE_LOAD_FAILED,
  IMAGE_INVALID_FORMAT,
  IMAGE_PROCESSING_FAILED,

  // Class name errors
  CLASS_NAMES_LOAD_FAILED,
  CLASS_NAMES_NOT_LOADED,
  INVALID_CLASS_INDEX,

  // Resource errors
  OUT_OF_MEMORY,
  RESOURCE_EXHAUSTED
};

// Convert error code to string
const char* toString(SdkError error);

// Print error message with [SDK ERROR] prefix
void printError(SdkError error);
void printError(SdkError error, const std::string& context);

}  // namespace error
}  // namespace cochl
