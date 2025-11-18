#ifndef TF_RUNTIME_H
#define TF_RUNTIME_H

#include "i_runtime.h"

#ifdef USE_TFLITE
#include <memory>
#include <vector>

// Forward declarations for TensorFlow Lite
namespace tflite {
class FlatBufferModel;
class Interpreter;
}  // namespace tflite

namespace cochl_api {
namespace runtime {

/**
 * @brief TensorFlow Lite runtime implementation
 */
class TFRuntime : public IRuntime {
public:
  TFRuntime();
  ~TFRuntime() override;

  bool LoadModel(const char* model_path) override;
  bool RunInference(const float* input, size_t input_size, float* output,
                    size_t output_size) override;
  const char* GetRuntimeType() const override { return "TensorFlow Lite"; }

private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  bool initialized_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TFLITE
#endif  // TF_RUNTIME_H
