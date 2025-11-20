#include "runtime/torch_runtime.h"
#include "runtime/runtime_manager.h"

#ifdef USE_LIBTORCH

#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

namespace cochl_api {
namespace runtime {

TorchRuntime::TorchRuntime()
    : initialized_(false),
      input_size_(0),
      output_size_(0) {}

TorchRuntime::~TorchRuntime() = default;

bool TorchRuntime::inferShapes() {
  try {
    // Try common input shapes for image models
    std::vector<std::vector<int64_t>> common_shapes = {
      {1, 3, 224, 224},  // ResNet, VGG, etc.
      {1, 3, 299, 299},  // Inception
      {1, 3, 512, 512},  // Larger models
    };

    for (const auto& shape : common_shapes) {
      try {
        // Create dummy input
        auto dummy_input = torch::zeros(shape, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(dummy_input);

        // Try forward pass
        auto output = module_->forward(inputs).toTensor();

        // Success! Cache the shapes
        input_shape_ = shape;
        input_size_ = 1;
        for (auto dim : shape) {
          input_size_ *= dim;
        }

        output_size_ = output.numel();

        std::cout << "[TorchRuntime] Inferred input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
          std::cout << input_shape_[i];
          if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], size: " << input_size_ << std::endl;
        std::cout << "[TorchRuntime] Inferred output size: " << output_size_ << std::endl;

        return true;
      } catch (...) {
        // Try next shape
        continue;
      }
    }

    std::cerr << "[TorchRuntime] Failed to infer shapes with common input sizes" << std::endl;
    return false;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Error inferring shapes: " << e.what() << std::endl;
    return false;
  }
}

bool TorchRuntime::loadModel(const char* model_path) {
  std::cout << "[TorchRuntime] Loading model from: " << model_path << std::endl;

  try {
    // Load the model
    module_ = std::make_unique<torch::jit::script::Module>(
        torch::jit::load(model_path));

    // Set to eval mode
    module_->eval();

    // Infer input/output shapes
    if (!inferShapes()) {
      std::cerr << "[TorchRuntime] Failed to infer model shapes" << std::endl;
      return false;
    }

    initialized_ = true;
    std::cout << "[TorchRuntime] Model loaded successfully" << std::endl;
    return true;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Error loading model: " << e.what() << std::endl;
    return false;
  }
}

bool TorchRuntime::runInference(const float* input, const std::vector<int64_t>& input_shape,
                                 float* output) {
  if (!initialized_) {
    std::cerr << "[TorchRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  if (!input || !output) {
    std::cerr << "[TorchRuntime] Invalid input or output pointer" << std::endl;
    return false;
  }

  if (input_shape.empty()) {
    std::cerr << "[TorchRuntime] Empty input shape" << std::endl;
    return false;
  }

  try {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    // Input is always in NCHW format (preprocessing handles conversion)
    torch::Tensor input_tensor = torch::from_blob(const_cast<float*>(input), input_shape, options).clone();

    // Prepare inputs for the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute the model
    torch::Tensor output_tensor = module_->forward(inputs).toTensor();

    // Get output shape and flatten
    output_tensor = output_tensor.flatten();

    // Verify output size matches cached value
    size_t total_elements = output_tensor.numel();
    if (total_elements != output_size_) {
      std::cerr << "[TorchRuntime] Output size mismatch. Expected: " << output_size_
                << ", Got: " << total_elements << std::endl;
      return false;
    }

    // Copy output data
    std::memcpy(output, output_tensor.data_ptr<float>(), output_size_ * sizeof(float));

    return true;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchRuntime] Inference error: " << e.what() << std::endl;
    return false;
  }
}

size_t TorchRuntime::getInputSize() const {
  return input_size_;
}

size_t TorchRuntime::getOutputSize() const {
  return output_size_;
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_LIBTORCH
