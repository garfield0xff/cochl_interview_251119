#include "runtime/torch_runtime.h"

#ifdef USE_LIBTORCH

#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

namespace cochl_api {
namespace runtime {

TorchRuntime::TorchRuntime() : initialized_(false) {}

TorchRuntime::~TorchRuntime() = default;

bool TorchRuntime::loadModel(const char* model_path) {
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

bool TorchRuntime::runInference(const float* input, size_t input_size, float* output,
                                 size_t output_size) {
  if (!initialized_) {
    std::cerr << "[TorchRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  try {
    // Create input tensor (for ResNet50: [1, 3, 224, 224])
    // Assuming input_size = 3 * 224 * 224 = 150528
    std::vector<int64_t> input_shape;
    if (input_size == 150528) {  // ResNet50 input
      input_shape = {1, 3, 224, 224};
    } else {
      // Fallback to 1D tensor
      input_shape = {1, static_cast<int64_t>(input_size)};
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input_tensor =
        torch::from_blob(const_cast<float*>(input), input_shape, options).clone();

    // Prepare inputs for the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute the model
    torch::Tensor output_tensor = module_->forward(inputs).toTensor();

    // Get output shape and flatten
    output_tensor = output_tensor.flatten();

    // Verify output size
    size_t total_elements = output_tensor.numel();

    if (total_elements != output_size) {
      std::cerr << "[TorchRuntime] Output size mismatch. Expected: " << output_size
                << ", Got: " << total_elements << std::endl;
      return false;
    }

    // Copy output data
    std::memcpy(output, output_tensor.data_ptr<float>(), output_size * sizeof(float));

    return true;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Inference error: " << e.what() << std::endl;
    return false;
  }
}

size_t TorchRuntime::getInputSize() const {
  // For ResNet50: 1 * 3 * 224 * 224 = 150528
  return 150528;
}

size_t TorchRuntime::getOutputSize() const {
  // For ResNet50: 1000 classes
  return 1000;
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_LIBTORCH
