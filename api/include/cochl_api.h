#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cochl_api {
namespace runtime {
class RuntimeManager;
enum class TensorLayout;
}
}  // namespace cochl_api

namespace external_api {
class CochlApi {
 public:
  // load_model
  static std::unique_ptr<CochlApi> create(const std::string& model_path);

  // Destructor must be declared here and defined in .cpp (for unique_ptr with forward declaration)
  ~CochlApi();

  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output) const;

  size_t getInputSize() const;
  size_t getOutputSize() const;

 private:
  CochlApi();
  std::unique_ptr<cochl_api::runtime::RuntimeManager> runtime_manager_;
};
}  // namespace external_api