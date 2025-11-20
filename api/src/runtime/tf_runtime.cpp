#include "runtime/tf_runtime.h"
#include "runtime/runtime_manager.h"

#ifdef USE_TFLITE

#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace cochl_api {
namespace runtime {

TFRuntime::TFRuntime()
    : initialized_(false),
      input_size_(0),
      output_size_(0) {}

TFRuntime::~TFRuntime() = default;

bool TFRuntime::loadModel(const char* model_path) {
  std::cout << "[TFRuntime] Loading model from: " << model_path << std::endl;

  // Load model
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model_) {
    std::cerr << "[TFRuntime] Failed to load model" << std::endl;
    return false;
  }

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);

  interpreter_ = std::make_unique<tflite::Interpreter>();
  if (builder(&interpreter_) != kTfLiteOk) {
    std::cerr << "[TFRuntime] Failed to build interpreter" << std::endl;
    return false;
  }

  // Allocate tensors
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    std::cerr << "[TFRuntime] Failed to allocate tensors" << std::endl;
    return false;
  }

  // Cache input/output sizes
  int input_idx = interpreter_->inputs()[0];
  TfLiteTensor* input_tensor = interpreter_->tensor(input_idx);
  input_size_ = 1;
  for (int i = 0; i < input_tensor->dims->size; ++i) {
    input_size_ *= input_tensor->dims->data[i];
  }

  int output_idx = interpreter_->outputs()[0];
  const TfLiteTensor* output_tensor = interpreter_->tensor(output_idx);
  output_size_ = 1;
  for (int i = 0; i < output_tensor->dims->size; ++i) {
    output_size_ *= output_tensor->dims->data[i];
  }

  std::cout << "[TFRuntime] Input size: " << input_size_ << std::endl;
  std::cout << "[TFRuntime] Output size: " << output_size_ << std::endl;

  initialized_ = true;
  std::cout << "[TFRuntime] Model loaded successfully" << std::endl;

  return true;
}

bool TFRuntime::runInference(const float* input, const std::vector<int64_t>& input_shape,
                              float* output) {
  if (!initialized_) {
    std::cerr << "[TFRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  if (!input || !output) {
    std::cerr << "[TFRuntime] Invalid input or output pointer" << std::endl;
    return false;
  }

  if (input_shape.empty()) {
    std::cerr << "[TFRuntime] Empty input shape" << std::endl;
    return false;
  }

  float* input_tensor = interpreter_->typed_input_tensor<float>(0);

  // Input is in NCHW format, TFLite expects NHWC, so convert NCHW -> NHWC
  if (input_shape.size() == 4) {
    // input_shape is NCHW: [N, C, H, W]
    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    // Convert NCHW -> NHWC
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int nchw_idx = n * C * H * W + c * H * W + h * W + w;
            int nhwc_idx = n * H * W * C + h * W * C + w * C + c;
            input_tensor[nhwc_idx] = input[nchw_idx];
          }
        }
      }
    }
  } else {
    std::cerr << "[TFRuntime] TFLite runtime requires 4D tensor (NCHW)" << std::endl;
    return false;
  }

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    std::cerr << "[TFRuntime] Inference failed" << std::endl;
    return false;
  }

  // Copy output data
  std::copy(interpreter_->typed_output_tensor<float>(0),
            interpreter_->typed_output_tensor<float>(0) + output_size_, output);

  return true;
}

size_t TFRuntime::getInputSize() const {
  return input_size_;
}

size_t TFRuntime::getOutputSize() const {
  return output_size_;
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TFLITE
