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
    module_ = tvm::ffi::Module::LoadFromFile(model_path);

    if (!module_.defined()) {
      std::cerr << "[TVMRuntime] Failed to load module" << std::endl;
      return false;
    }

    const tvm::ffi::Module& mod = module_.value();

    // Check if this is a Relax VM module by looking for vm_load_executable
    auto vm_load_func = mod->GetFunction("vm_load_executable");
    if (vm_load_func.defined()) {
      std::cout << "[TVMRuntime] Detected Relax VM module, creating VirtualMachine..." << std::endl;

      // Step 1: Call vm_load_executable to get the VM module
      // This is equivalent to Python's: self.module = rt_mod["vm_load_executable"]()
      tvm::ffi::Any vm_result = vm_load_func.value()();
      vm_module_ = vm_result.cast<tvm::ffi::Module>();

      if (!vm_module_.defined()) {
        std::cerr << "[TVMRuntime] Failed to create VM module from vm_load_executable" << std::endl;
        return false;
      }
      std::cout << "[TVMRuntime] VM module created from vm_load_executable" << std::endl;

      // Step 2: Initialize the VM with device info
      // Python equivalent: self.module["vm_initialization"](device_type, device_id, alloc_type)
      auto vm_init_func = vm_module_.value()->GetFunction("vm_initialization");
      if (vm_init_func.defined()) {
        // Args: device_type, device_id, allocator_type (POOLED_ALLOCATOR = 2)
        int device_type = static_cast<int>(device_.device_type);
        int device_id = device_.device_id;
        int alloc_type = 2;  // POOLED_ALLOCATOR

        // Also add CPU device for shape functions (same as Python)
        vm_init_func.value()(device_type, device_id, alloc_type, kDLCPU, 0, alloc_type);
        std::cout << "[TVMRuntime] VM initialized with device" << std::endl;
      } else {
        std::cerr << "[TVMRuntime] vm_initialization function not found" << std::endl;
        return false;
      }

      // Step 3: Get the "main" function from VM module
      auto main_func = vm_module_.value()->GetFunction("main");
      if (!main_func.defined()) {
        std::cerr << "[TVMRuntime] Could not find 'main' function in VirtualMachine" << std::endl;
        return false;
      }

      inference_func_ = main_func;
      std::cout << "[TVMRuntime] 'main' function found in VirtualMachine" << std::endl;
    } else {
      // Non-VM module: Try to get the main function directly
      auto func_opt = mod->GetFunction("main");
      if (!func_opt.defined()) {
        func_opt = mod->GetFunction("__tvm_main__");
        if (!func_opt.defined()) {
          func_opt = mod->GetFunction("default");
          if (!func_opt.defined()) {
            std::cerr << "[TVMRuntime] Could not find main/default function in module" << std::endl;
            return false;
          }
        }
      }
      inference_func_ = func_opt;
    }

    // Set default shapes for ResNet50-like models (NHWC for TFLite-converted models)
    input_shape_ = {1, 224, 224, 3};  // NHWC format (from TFLite)
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

    // Create TVM tensor for input using Tensor::Empty
    tvm::runtime::Tensor input_tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape(input_shape), dtype, device_);

    // Copy input data to tensor
    float* input_data = static_cast<float*>(input_tensor->data);
    std::copy(input, input + input_size, input_data);

    // Run inference and get output
    tvm::ffi::Any result = inference_func_.value()(input_tensor);

    // Extract output tensor from result
    tvm::runtime::Tensor output_tensor = result.cast<tvm::runtime::Tensor>();

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
