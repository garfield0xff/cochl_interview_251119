#ifndef RUNTIME_MANAGER_H
#define RUNTIME_MANAGER_H

#include "i_runtime.h"

#include <memory>
#include <string>

namespace cochl_api {
namespace runtime {

/**
 * @brief Runtime AutoCastingㅆ
 */
class RuntimeManager {
public:

  /**
   * @brief Support Runtime
   */
  enum class InferenceEngine {
    UNKNOWN,
    TFLITE,
    LIBTORCH,
    TVM,
    CUSTOM
  };

  /**
   * @brief Create runtime manager and load model
   * @param model_path Path to model file
   * @return Unique pointer to RuntimeManager, nullptr on failure
   */
  static std::unique_ptr<RuntimeManager> create(const std::string& model_path);

  ~RuntimeManager();

  /**
   * @brief Run inference using the loaded runtime
   * @param input Input data array (must be in NCHW format)
   * @param input_shape Shape of input tensor (e.g., {1, 3, 224, 224} for NCHW)
   * @param output Output data array (must be pre-allocated with getOutputSize())
   */
  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output) const;

  /**
   * @brief Get inference engine type
   */
  InferenceEngine getInferenceEngineType() const { return runtime_type_; }

  /**
   * @brief Get input size
   */
  size_t getInputSize() const;

  /**
   * @brief Get output size
   */
  size_t getOutputSize() const;

private:
  RuntimeManager();

  /**
   * @brief Detect inference engine from file extension
   * @param model_path Model file path
   * @return Detected InferenceEngine
   */
  static InferenceEngine detectInferenceEngine(const std::string& model_path);

  /**
   * @brief Load model using appropriate runtime
   * @param model_path Path to model file
   * @param type Runtime type to use
   * @return true if successful, false otherwise
   */
  bool loadModel(const std::string& model_path, InferenceEngine type);

  std::unique_ptr<IRuntime> runtime_;  //runtime instance
  InferenceEngine runtime_type_;
  bool initialized_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // RUNTIME_MANAGER_H
