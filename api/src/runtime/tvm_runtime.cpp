#include "runtime/tvm_runtime.h"

#ifdef USE_TVM

#include <iostream>
#include <numeric>

namespace cochl_api {
namespace runtime {

TVMRuntime::TVMRuntime()
    : input_size_(0), output_size_(0), initialized_(false) {
  // Initialize device to CPU
  device_.device_type = kDLCPU;
  device_.device_id = 0;
}

TVMRuntime::~TVMRuntime() = default;

bool TVMRuntime::loadModel(const char* model_path) {
  std::cout << "[TVMRuntime] Loading model from: " << model_path << std::endl;

  try {
    // Load the compiled TVM module (.so file)
    module_ = tvm::runtime::Module::LoadFromFile(model_path);

    if (module_.operator->() == nullptr) {
      std::cerr << "[TVMRuntime] Failed to load module" << std::endl;
      return false;
    }

    // Get the main inference function
    // Common function names: "main", "default", or the entry point specified during compilation
    auto func_opt = module_->GetFunction("main");
    if (!func_opt.has_value()) {
      // Try alternative function names
      func_opt = module_->GetFunction("default");
      if (!func_opt.has_value()) {
        std::cerr << "[TVMRuntime] Could not find main/default function in module" << std::endl;
        return false;
      }
    }

    inference_func_ = func_opt.value();

    // For TVM runtime, we need to determine input/output shapes
    // This can be done by:
    // 1. Running the function once with dummy data
    // 2. Using metadata if available
    // 3. Setting default shapes (e.g., for ResNet50: input [1, 3, 224, 224], output [1, 1000])

    // Set default shapes for ResNet50-like models
    // These should be configured based on your actual model
    input_shape_ = {1, 3, 224, 224};  // NCHW format
    output_shape_ = {1, 1000};         // ImageNet classes

    input_size_ = calculateSize(input_shape_);
    output_size_ = calculateSize(output_shape_);

    initialized_ = true;
    std::cout << "[TVMRuntime] Model loaded successfully" << std::endl;
    std::cout << "[TVMRuntime] Input size: " << input_size_ << std::endl;
    std::cout << "[TVMRuntime] Output size: " << output_size_ << std::endl;

    return true;
  } catch (const std::exception& e) {
    std::cerr << "[TVMRuntime] Exception during model loading: " << e.what() << std::endl;
    return false;
  }
}

bool TVMRuntime::runInference(const float* input, const std::vector<int64_t>& input_shape,
                               float* output) {
  if (!initialized_) {
    std::cerr << "[TVMRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  if (!input || !output) {
    std::cerr << "[TVMRuntime] Invalid input or output pointer" << std::endl;
    return false;
  }

  if (input_shape.empty()) {
    std::cerr << "[TVMRuntime] Empty input shape" << std::endl;
    return false;
  }

  // Calculate input size from shape
  size_t input_size = 1;
  for (auto dim : input_shape) {
    input_size *= dim;
  }

  try {
    // Create input tensor
    DLDataType dtype;
    dtype.code = kDLFloat;
    dtype.bits = 32;
    dtype.lanes = 1;

    // Create TVM tensor for input
    tvm::runtime::Tensor input_tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape(input_shape), dtype, device_);

    // Copy input data to tensor
    // Input is already in NCHW format (TVM's native format)
    float* input_data = static_cast<float*>(input_tensor->data);
    std::copy(input, input + input_size, input_data);

    // Create output tensor
    tvm::runtime::Tensor output_tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape(output_shape_), dtype, device_);

    // Run inference
    // The function signature is typically: func(input_tensor, output_tensor)
    inference_func_(input_tensor, output_tensor);

    // Copy output data from tensor
    float* output_data = static_cast<float*>(output_tensor->data);
    std::copy(output_data, output_data + output_size_, output);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "[TVMRuntime] Exception during inference: " << e.what() << std::endl;
    return false;
  }
}

size_t TVMRuntime::getInputSize() const {
  return input_size_;
}

size_t TVMRuntime::getOutputSize() const {
  return output_size_;
}

size_t TVMRuntime::calculateSize(const std::vector<int64_t>& shape) const {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1LL,
                         std::multiplies<int64_t>());
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TVM
