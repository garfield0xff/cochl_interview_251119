#ifndef I_RUNTIME_H
#define I_RUNTIME_H

#include <cstddef>

namespace cochl_api {
namespace runtime {

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
  virtual bool LoadModel(const char* model_path) = 0;

  /**
   * @brief Run inference
   * @param input Input data array
   * @param input_size Size of input array
   * @param output Output data array
   * @param output_size Size of output array
   * @return true if successful, false otherwise
   */
  virtual bool RunInference(const float* input, size_t input_size, float* output,
                            size_t output_size) = 0;

  /**
   * @brief Get runtime type name
   * @note  use later  
   */
  virtual const char* GetRuntimeType() const = 0;

  /**
   * @brief Get input size
   * @return Size of input array
   */
  virtual size_t GetInputSize() const = 0;

  /**
   * @brief Get output size
   * @return Size of output array
   */
  virtual size_t GetOutputSize() const = 0;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // I_RUNTIME_H
