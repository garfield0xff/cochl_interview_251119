#ifndef I_RUNTIME_H
#define I_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cochl_api {
namespace runtime {

// Forward declaration
enum class TensorLayout;

/**
 * @brief Base interface for all runtime backends
 */
class IRuntime {
public:
  virtual ~IRuntime() = default;

  /**
   * @brief Load model from file
   * @param model_path Path to model file
   * @return true if successful, false otherwise
   */
  virtual bool loadModel(const char* model_path) = 0;

  /**
   * @brief Run inference
   * @param input Input data array
   * @param input_shape Shape of input tensor (e.g., {1, 3, 224, 224})
   * @param output Output data array (must be pre-allocated with getOutputSize())
   * @param layout Tensor layout (NCHW or NHWC)
   * @return true if successful, false otherwise
   */
  virtual bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                            float* output, TensorLayout layout) = 0;

  /**
   * @brief Get runtime type name
   * @note  use later
   */
  virtual const char* getRuntimeType() const = 0;

  /**
   * @brief Get input size
   * @return Size of input array
   */
  virtual size_t getInputSize() const = 0;

  /**
   * @brief Get output size
   * @return Size of output array
   */
  virtual size_t getOutputSize() const = 0;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // I_RUNTIME_H
