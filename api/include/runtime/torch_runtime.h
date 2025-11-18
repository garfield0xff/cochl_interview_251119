#ifndef TORCH_RUNTIME_H
#define TORCH_RUNTIME_H

#include "i_runtime.h"

#ifdef USE_LIBTORCH
#include <memory>
#include <vector>

// Forward declarations for LibTorch
namespace torch {
namespace jit {
class Module;
}
namespace autograd {
class Variable;
}
}  // namespace torch

namespace cochl_api {
namespace runtime {

/**
 * @brief LibTorch runtime implementation
 */
class TorchRuntime : public IRuntime {
public:
  TorchRuntime();
  ~TorchRuntime() override;

  bool LoadModel(const char* model_path) override;
  bool RunInference(const float* input, size_t input_size, float* output,
                    size_t output_size) override;
  const char* GetRuntimeType() const override { return "LibTorch"; }

private:
  std::unique_ptr<torch::jit::Module> module_;
  bool initialized_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_LIBTORCH
#endif  // TORCH_RUNTIME_H
