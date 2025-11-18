#pragma once

#include <memory>
#include <string>

namespace external_api {
class CochlApi {
 public:
  // load_model
  static std::unique_ptr<CochlApi> Create(const std::string& model_path);

  bool RunInference(const float* input, size_t input_size, float* output, size_t output_size) const;

  size_t GetInputSize() const { return 22050; }
  size_t GetOutputSize() const { return 10; }

 private:
  CochlApi();
  bool is_initialized = false;
};
}  // namespace external_api