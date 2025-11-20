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

/**
 * @brief Support Multi Runtime Backend
 * 1. API 는 Core로서 Multi Runtime에 대한 조건부 컴파일을 지원해야함
 * 2. 조건부는 Runtime, Architecture, Accelerator 를 지원
 * 3. EdgeSDK에서 사용하기위해 정적링크를 지원해야함. <--- 이거 먼저 ( libtorch, tensorflow 정적 컴파일 해놔야할듯 ) 
 */
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