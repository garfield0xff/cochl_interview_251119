// Abstract interface for inference backend implementations.
// Defines the contract for synchronous inference operations.
// Supports various backends (CppApi, TFLite, QNpu) with consistent API.

#pragma once

#include <string>
#include <vector>

namespace cochl {
enum class InferenceStatus {
  kSuccess = 0,
  kErrorInvalidInput = 1,
  kErrorInternal = 2,
};

struct Tensor {
  float* data = nullptr;
  std::vector<int> dims;
};

class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  virtual bool LoadModel(const std::string& model_path) = 0;
  virtual InferenceStatus RunInference(const Tensor& input, Tensor* output) = 0;
  virtual std::vector<int> GetInputDims() const = 0;
  virtual std::vector<int> GetOutputDims() const = 0;

 protected:
  IInferenceBackend() = default;
  IInferenceBackend(const IInferenceBackend&) = delete;
  IInferenceBackend& operator=(const IInferenceBackend&) = delete;
};
}  // namespace cochl
