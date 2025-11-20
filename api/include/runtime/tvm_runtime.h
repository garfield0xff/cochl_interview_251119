#ifndef TVM_RUNTIME_H
#define TVM_RUNTIME_H

#include "i_runtime.h"

#ifdef USE_TVM
#include <memory>
#include <string>
#include <vector>

// TVM runtime includes
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/tensor.h>

namespace cochl_api {
namespace runtime {

/**
 * @brief TVM runtime implementation
 *
 * This runtime loads and executes models compiled with TVM.
 * TVM models should be compiled as shared libraries (.so files).
 */
class TVMRuntime : public IRuntime {
public:
  TVMRuntime();
  ~TVMRuntime() override;

  bool loadModel(const char* model_path) override;
  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output) override;
  const char* getRuntimeType() const override { return "TVM"; }
  size_t getInputSize() const override;
  size_t getOutputSize() const override;

private:
  // TVM module loaded from compiled model
  tvm::runtime::Module module_;

  // Main inference function
  tvm::runtime::PackedFunc inference_func_;

  // Input and output tensor metadata
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;

  // Cached sizes
  size_t input_size_;
  size_t output_size_;

  // Device context (CPU by default)
  DLDevice device_;

  // Initialization flag
  bool initialized_;

  /**
   * @brief Calculate total size from shape vector
   */
  size_t calculateSize(const std::vector<int64_t>& shape) const;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TVM
#endif  // TVM_RUNTIME_H
