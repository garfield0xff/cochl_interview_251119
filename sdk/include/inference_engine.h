// Main inference engine that manages backend selection and execution.
// Uses dlopen to dynamically load libcochl_api.so at runtime.

#pragma once

#include <memory>
#include <string>

namespace cochl {

enum class InferenceStatus {
  OK,
  ERROR_NOT_INITIALIZED,
  ERROR_INVALID_INPUT,
  ERROR_INFERENCE_FAILED,
  ERROR_LIBRARY_LOAD_FAILED
};

class InferenceEngine {
 public:
  InferenceEngine();
  ~InferenceEngine();

  // Load libcochl_api.so dynamically
  bool loadLibrary(const std::string& library_path);

  // Load model from file path
  // Model format is auto-detected: .tflite -> TFLite, .pt/.pth -> LibTorch
  bool loadModel(const std::string& model_path);

  // Run inference
  // input: float array of input data
  // input_size: size of input array
  // output: float array to store output (must be pre-allocated)
  // output_size: size of output array
  InferenceStatus runInference(const float* input, size_t input_size,
                                float* output, size_t output_size);

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
  void* lib_handle_;        // dlopen handle
  void* api_instance_;      // CochlApi instance
  void* class_map_;         // ImageNet class map

  // Function pointers from C API
  void* (*CochlApi_Create_)(const char*);
  int (*CochlApi_RunInference_)(void*, const float*, size_t, float*, size_t);
  size_t (*CochlApi_GetInputSize_)(void*);
  size_t (*CochlApi_GetOutputSize_)(void*);
  void (*CochlApi_Destroy_)(void*);
  int (*CochlApi_LoadImage_)(const char*, float*, size_t);
  void* (*CochlApi_LoadClassNames_)(const char*);
  const char* (*CochlApi_GetClassName_)(void*, int);
  void (*CochlApi_DestroyClassMap_)(void*);
};

}  // namespace cochl
