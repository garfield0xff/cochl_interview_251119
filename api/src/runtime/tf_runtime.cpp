#include "runtime/tf_runtime.h"

#ifdef USE_TFLITE

#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace cochl_api {
namespace runtime {

TFRuntime::TFRuntime() : initialized_(false) {}

TFRuntime::~TFRuntime() = default;

bool TFRuntime::LoadModel(const char* model_path) {
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

  initialized_ = true;
  std::cout << "[TFRuntime] Model loaded successfully" << std::endl;

  return true;
}

bool TFRuntime::RunInference(const float* input, size_t input_size, float* output,
                              size_t output_size) {
  if (!initialized_) {
    std::cerr << "[TFRuntime] Runtime not initialized" << std::endl;
    return false;
  }

  // Get input tensor
  int input_idx           = interpreter_->inputs()[0];
  TfLiteTensor* input_tensor = interpreter_->tensor(input_idx);

  // Verify input size
  size_t expected_input_size = 1;
  for (int i = 0; i < input_tensor->dims->size; ++i) {
    expected_input_size *= input_tensor->dims->data[i];
  }

  if (input_size != expected_input_size) {
    std::cerr << "[TFRuntime] Input size mismatch. Expected: " << expected_input_size
              << ", Got: " << input_size << std::endl;
    return false;
  }

  // Copy input data
  std::copy(input, input + input_size, interpreter_->typed_input_tensor<float>(0));

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    std::cerr << "[TFRuntime] Inference failed" << std::endl;
    return false;
  }

  // Get output tensor
  int output_idx             = interpreter_->outputs()[0];
  const TfLiteTensor* output_tensor = interpreter_->tensor(output_idx);

  // Verify output size
  size_t expected_output_size = 1;
  for (int i = 0; i < output_tensor->dims->size; ++i) {
    expected_output_size *= output_tensor->dims->data[i];
  }

  if (output_size != expected_output_size) {
    std::cerr << "[TFRuntime] Output size mismatch. Expected: " << expected_output_size
              << ", Got: " << output_size << std::endl;
    return false;
  }

  // Copy output data
  std::copy(interpreter_->typed_output_tensor<float>(0),
            interpreter_->typed_output_tensor<float>(0) + output_size, output);

  return true;
}

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TFLITE
