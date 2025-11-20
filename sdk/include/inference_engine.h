// Main inference engine that manages backend selection and execution.
// Uses dlopen to dynamically load libcochl_api.so at runtime.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api/cochl_api.h"

namespace cochl {

class InferenceEngine {
 public:
  InferenceEngine();
  ~InferenceEngine();

  // Load libcochl_api.so dynamically
  bool loadLib(const std::string& library_path);

  // Create API instance and load model from file path
  // Model format is auto-detected: .tflite -> TFLite, .pt/.pth -> LibTorch
  bool create(const std::string& model_path);

  // Run inference
  // input: float array of input data (must be in NCHW format)
  // input_shape: shape of input tensor in NCHW format (e.g., {1, 3, 224, 224})
  // output: float array to store output (must be pre-allocated with getOutputSize())
  // Returns true on success, false on error
  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output);

  // Get input tensor size
  size_t getInputSize() const;

  // Get output tensor size
  size_t getOutputSize() const;

  // Load and preprocess image (returns preprocessed data in NCHW format)
  bool loadImage(const std::string& image_path, float* output_data, size_t output_size);

  // Load ImageNet class names
  bool loadClassNames(const std::string& json_path);

  // Get class name from index
  std::string getClassName(int class_idx) const;

 private:
  api::CochlApi api_loader_;  // Dynamic library loader
  void* api_instance_;        // CochlApi instance
  void* class_map_;           // ImageNet class map
};

}  // namespace cochl
