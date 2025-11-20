#ifndef TORCH_RUNTIME_H
#define TORCH_RUNTIME_H

#include "i_runtime.h"

#ifdef USE_LIBTORCH
#include <memory>
#include <vector>

// LibTorch includes
#include <torch/script.h>

namespace cochl_api {
namespace runtime {

/**
 * @brief LibTorch runtime implementation
 */
class TorchRuntime : public IRuntime {
public:
  TorchRuntime();
  ~TorchRuntime() override;

  bool loadModel(const char* model_path) override;
  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output, TensorLayout layout) override;
  const char* getRuntimeType() const override { return "LibTorch"; }
  size_t getInputSize() const override;
  size_t getOutputSize() const override;

private:
  std::unique_ptr<torch::jit::Module> module_;
  bool initialized_;

  // Cached shape information
  std::vector<int64_t> input_shape_;
  size_t input_size_;
  size_t output_size_;

  // Helper to infer shapes from model
  bool inferShapes();
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_LIBTORCH
#endif  // TORCH_RUNTIME_H
