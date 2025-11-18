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
 * - .bin -> Custom thread pool runtime (mock)
 */
class RuntimeManager {
public:

  enum class InferenceEngine {
    UNKNOWN,
    TFLITE,
    LIBTORCH,
    CUSTOM
  };

  /**
   * @brief Create runtime manager and load model
   * @param model_path Path to model file
   * @return Unique pointer to RuntimeManager, nullptr on failure
   */
  static std::unique_ptr<RuntimeManager> Create(const std::string& model_path);

  ~RuntimeManager();

  /**
   * @brief Run inference using the loaded runtime
   */
  bool RunInference(const float* input, size_t input_size, float* output,
                    size_t output_size) const;

  /**
   * @brief Get current inference engine type
   */
  InferenceEngine GetInferenceEngineType() const { return runtime_type_; }

  /**
   * @brief Get input size
   */
  size_t GetInputSize() const;

  /**
   * @brief Get output size
   */
  size_t GetOutputSize() const;

private:
  RuntimeManager();

  /**
   * @brief Detect inference engine from file extension
   * @param model_path Model file path
   * @return Detected InferenceEngine
   */
  static InferenceEngine DetectInferenceEngine(const std::string& model_path);

  /**
   * @brief Load model using appropriate runtime
   * @param model_path Path to model file
   * @param type Runtime type to use
   * @return true if successful, false otherwise
   */
  bool LoadModel(const std::string& model_path, InferenceEngine type);

  std::unique_ptr<IRuntime> runtime_;  // Holds the actual runtime instance
  InferenceEngine runtime_type_;
  bool initialized_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // RUNTIME_MANAGER_H
