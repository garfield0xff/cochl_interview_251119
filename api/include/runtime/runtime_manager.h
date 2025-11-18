#ifndef RUNTIME_MANAGER_H
#define RUNTIME_MANAGER_H

#include "i_runtime.h"

#include <memory>
#include <string>

namespace cochl_api {
namespace runtime {

/**
 * @brief Manages runtime selection based on model file extension
 *
 * Supported formats:
 * - .tflite -> TensorFlow Lite runtime
 * - .pt, .pth -> LibTorch runtime
 * - .so, .dll -> TVM runtime (future)
 */
class RuntimeManager {
public:
  /**
   * @brief Runtime type enumeration
   */
  enum class RuntimeType {
    UNKNOWN,
    TFLITE,   // .tflite
    LIBTORCH  // .pt, .pth
  };

  /**
   * @brief Create runtime manager and load model
   * @param model_path Path to model file
   * @return Unique pointer to RuntimeManager, nullptr on failure
   */
  static std::unique_ptr<RuntimeManager> Create(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~RuntimeManager();

  /**
   * @brief Run inference using the loaded runtime
   * @param input Input data array
   * @param input_size Size of input array
   * @param output Output data array
   * @param output_size Size of output array
   * @return true if successful, false otherwise
   */
  bool RunInference(const float* input, size_t input_size, float* output,
                    size_t output_size) const;

  /**
   * @brief Get current runtime type
   * @return RuntimeType enum value
   */
  RuntimeType GetRuntimeType() const { return runtime_type_; }

  /**
   * @brief Get runtime type name string
   * @return Runtime type name
   */
  const char* GetRuntimeTypeName() const;

private:
  RuntimeManager();

  /**
   * @brief Detect runtime type from file extension
   * @param model_path Model file path
   * @return Detected RuntimeType
   */
  static RuntimeType DetectRuntimeType(const std::string& model_path);

  /**
   * @brief Load model using appropriate runtime
   * @param model_path Path to model file
   * @param type Runtime type to use
   * @return true if successful, false otherwise
   */
  bool LoadModel(const std::string& model_path, RuntimeType type);

  std::unique_ptr<IRuntime> runtime_;  // Holds the actual runtime instance
  RuntimeType runtime_type_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // RUNTIME_MANAGER_H
