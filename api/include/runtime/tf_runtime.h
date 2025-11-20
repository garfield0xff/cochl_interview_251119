#ifndef TF_RUNTIME_H
#define TF_RUNTIME_H

#include "i_runtime.h"

#ifdef USE_TFLITE
#include <memory>
#include <vector>

// TensorFlow Lite includes
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

namespace cochl_api {
namespace runtime {

/**
 * @brief TensorFlow Lite runtime implementation
 */
class TFRuntime : public IRuntime {
public:
  TFRuntime();
  ~TFRuntime() override;

  bool loadModel(const char* model_path) override;
  bool runInference(const float* input, const std::vector<int64_t>& input_shape,
                    float* output, TensorLayout layout) override;
  const char* getRuntimeType() const override { return "TensorFlow Lite"; }
  size_t getInputSize() const override;
  size_t getOutputSize() const override;

private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  bool initialized_;

  // Cached shape information
  size_t input_size_;
  size_t output_size_;
};

}  // namespace runtime
}  // namespace cochl_api

#endif  // USE_TFLITE
#endif  // TF_RUNTIME_H
