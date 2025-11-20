/**
 * @file api_error.h
 * @brief Cochl API error code definitions and exception handling
 */

#ifndef API_ERROR_H
#define API_ERROR_H

#include <exception>
#include <string>

namespace cochl_api {
namespace error {

/**
 * @brief Error codes used throughout the Cochl API
 *
 * Defines various error conditions that can occur during API operations.
 */
enum class ApiError {
  SUCCESS = 0,              /**< Operation completed successfully */
  UNKNOWN_ERROR,            /**< Unknown error occurred */

  // Model loading errors
  MODEL_NOT_FOUND,          /**< Model file does not exist */
  MODEL_LOAD_FAILED,        /**< Failed to load model */
  MODEL_INVALID_FORMAT,     /**< Unsupported model format */
  MODEL_ALREADY_LOADED,     /**< Model already loaded */

  // Runtime errors
  RUNTIME_NOT_INITIALIZED,  /**< Runtime not properly initialized */
  RUNTIME_NOT_SUPPORTED,    /**< Runtime not compiled or supported */
  RUNTIME_CREATION_FAILED,  /**< Failed to create runtime instance */

  // Inference errors
  INFERENCE_FAILED,         /**< Inference execution failed */
  INVALID_INPUT_SIZE,       /**< Input size mismatch */
  INVALID_OUTPUT_SIZE,      /**< Output size mismatch */
  INVALID_INPUT_DATA,       /**< Invalid input data pointer */
  INVALID_OUTPUT_DATA,      /**< Invalid output data pointer */

  // Parameter errors
  INVALID_PARAMETER,        /**< Invalid function parameter */
  NULL_POINTER,             /**< Null pointer provided */
  EMPTY_PATH,               /**< Empty file path */

  // Resource errors
  OUT_OF_MEMORY,            /**< Memory allocation failed */
  RESOURCE_EXHAUSTED        /**< System resources exhausted */
};

/**
 * @brief Type alias for backward compatibility
 */
using result_t = ApiError;

/**
 * @brief Convert ApiError enum value to error message string
 * @param error Error code to convert
 * @return Error message string corresponding to the error code
 */
const char* toString(ApiError error);

/**
 * @brief Print error to stderr with standardized format
 * @param error Error code to print
 */
void printError(ApiError error);

/**
 * @brief Print error to stderr with additional context information
 * @param error Error code to print
 * @param context Additional context string
 */
void printError(ApiError error, const std::string& context);

/**
 * @brief Convert ApiError enum value to integer
 * @param error Error code to convert
 * @return Integer representation of error code
 */
inline int toInt(ApiError error) {
  return static_cast<int>(error);
}

/**
 * @brief Check if error code indicates success
 * @param error Error code to check
 * @return true if error is SUCCESS, false otherwise
 */
inline bool isSuccess(ApiError error) {
  return error == ApiError::SUCCESS;
}

/**
 * @brief Check if error code indicates failure
 * @param error Error code to check
 * @return true if error is not SUCCESS, false otherwise
 */
inline bool isError(ApiError error) {
  return error != ApiError::SUCCESS;
}

/**
 * @brief Exception class for Cochl API errors
 *
 * Provides exception handling based on ApiError codes with
 * detailed error messages and stack context.
 */
class ApiException : public std::exception {
 private:
  ApiError error_code_;
  mutable std::string error_message_;
  std::string context_;

 public:
  /**
   * @brief Construct exception with error code only
   * @param code Error code
   */
  explicit ApiException(ApiError code)
    : error_code_(code), context_("") {}

  /**
   * @brief Construct exception with error code and context
   * @param code Error code
   * @param context Additional context information
   */
  ApiException(ApiError code, const std::string& context)
    : error_code_(code), context_(context) {}

  virtual ~ApiException() noexcept {}

  /**
   * @brief Get the error code
   * @return ApiError code associated with this exception
   */
  ApiError getErrorCode() const { return error_code_; }

  /**
   * @brief Get the context information
   * @return Context string
   */
  const std::string& getContext() const { return context_; }

  /**
   * @brief Get the error message
   * @return Error message string with context if available
   */
  virtual const char* what() const noexcept override {
    if (context_.empty()) {
      error_message_ = toString(error_code_);
    } else {
      error_message_ = std::string(toString(error_code_)) + ": " + context_;
    }
    return error_message_.c_str();
  }
};

}  // namespace error
}  // namespace cochl_api

#endif  // API_ERROR_H
