#ifndef SDK_INFERENCE_ENGINE_H
#define SDK_INFERENCE_ENGINE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cochl {
class ThreadPool;
}

namespace cochl_sdk {

// Forward declaration
class Profiler;

/**
 * @brief Main inference engine for edge devices
 *
 * Uses API shared library (libcochl_api.so/dll) for inference.
 * Supports Linux, Windows, and Mac platforms.
 */
class InferenceEngine {
public:
  /**
   * @brief Create inference engine
   * @param model_path Path to model file (.tflite, .pt, .pth)
   * @param num_threads Number of threads for thread pool (0 = auto)
   * @param enable_profiler Enable performance profiling
   * @return Unique pointer to InferenceEngine, nullptr on failure
   */
  static std::unique_ptr<InferenceEngine> Create(const std::string& model_path,
                                                   int num_threads = 0,
                                                   bool enable_profiler = false);

  ~InferenceEngine();

  /**
   * @brief Run inference
   * @param input Input data array
   * @param input_size Size of input array
   * @param output Output data array
   * @param output_size Size of output array
   * @return true if successful, false otherwise
   */
  bool RunInference(const float* input, size_t input_size, float* output,
                    size_t output_size);

  /**
   * @brief Run inference asynchronously using thread pool
   * @param input Input data array
   * @param input_size Size of input array
   * @param output Output data array
   * @param output_size Size of output array
   * @param callback Callback function called when inference completes
   * @return true if task submitted successfully, false otherwise
   */
  bool RunInferenceAsync(const float* input, size_t input_size, float* output,
                         size_t output_size,
                         std::function<void(bool success)> callback);

  /**
   * @brief Get profiling statistics
   * @return Profiling stats as formatted string
   */
  std::string GetProfilingStats() const;

  /**
   * @brief Get system resource usage
   * @return Resource usage info (memory, temperature, etc.)
   */
  std::string GetResourceUsage() const;

  /**
   * @brief Get input size required by model
   * @return Input size
   */
  size_t GetInputSize() const { return input_size_; }

  /**
   * @brief Get output size produced by model
   * @return Output size
   */
  size_t GetOutputSize() const { return output_size_; }

private:
  InferenceEngine();

  // API library handle (dynamically loaded)
  void* api_handle_;
  void* cochl_api_instance_;

  // Function pointers from API library
  using CreateFn = void* (*)(const char*);
  using RunInferenceFn = bool (*)(void*, const float*, size_t, float*, size_t);
  using GetInputSizeFn = size_t (*)(void*);
  using GetOutputSizeFn = size_t (*)(void*);
  using DestroyFn = void (*)(void*);

  CreateFn create_fn_;
  RunInferenceFn run_inference_fn_;
  GetInputSizeFn get_input_size_fn_;
  GetOutputSizeFn get_output_size_fn_;
  DestroyFn destroy_fn_;

  // SDK components
  std::unique_ptr<Profiler> profiler_;
  std::unique_ptr<cochl::ThreadPool> thread_pool_;

  // Model info
  size_t input_size_;
  size_t output_size_;
  bool initialized_;

  /**
   * @brief Load API shared library
   * @param model_path Model path (used to locate library)
   * @return true if successful, false otherwise
   */
  bool LoadApiLibrary(const std::string& model_path);

  /**
   * @brief Unload API shared library
   */
  void UnloadApiLibrary();
};

}  // namespace cochl_sdk

#endif  // SDK_INFERENCE_ENGINE_H
