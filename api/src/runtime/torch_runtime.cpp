#include "runtime/torch_runtime.h"

#ifdef USE_LIBTORCH

#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

namespace cochl_api {
namespace runtime {

TorchRuntime::TorchRuntime() : initialized_(false) {}

TorchRuntime::~TorchRuntime() = default;

bool TorchRuntime::LoadModel(const char* model_path) {
  std::cout << "[TorchRuntime] Loading model from: " << model_path << std::endl;

  try {
    // Load the model
    module_ = std::make_unique<torch::jit::script::Module>(
        torch::jit::load(model_path));

    // Set to eval mode
    module_->eval();

    initialized_ = true;
    std::cout << "[TorchRuntime] Model loaded successfully" << std::endl;
    return true;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Error loading model: " << e.what() << std::endl;
    return false;
  }
}

bool TorchRuntime::RunInference(const float* input, size_t input_size, float* output,
                                 size_t output_size) {
  if (!initialized_) {
    std::cerr << "[TorchRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  try {
    // Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_size)};
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input_tensor =
        torch::from_blob(const_cast<float*>(input), input_shape, options).clone();

    // Prepare inputs for the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute the model
    torch::Tensor output_tensor = module_->forward(inputs).toTensor();

    // Verify output size
    auto output_shape = output_tensor.sizes();
    size_t total_elements = 1;
    for (auto dim : output_shape) {
      total_elements *= dim;
    }

    if (total_elements != output_size) {
      std::cerr << "[TorchRuntime] Output size mismatch. Expected: " << output_size
                << ", Got: " << total_elements << std::endl;
      return false;
    }

    // Copy output data
    auto output_accessor = output_tensor.accessor<float, 2>();
    size_t idx = 0;
    for (int i = 0; i < output_accessor.size(0); ++i) {
      for (int j = 0; j < output_accessor.size(1); ++j) {
        output[idx++] = output_accessor[i][j];
      }
    }

    return true;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Inference error: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_LIBTORCH
