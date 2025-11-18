#include "inference_engine.h"

#include <dlfcn.h>  // For dynamic library loading on Linux/Mac
#include <iostream>
#include <sstream>

#include "kernel/system_monitor.h"
#include "profiler.h"
#include "thread_pool.h"

namespace cochl_sdk {

InferenceEngine::InferenceEngine()
    : api_handle_(nullptr),
      cochl_api_instance_(nullptr),
      create_fn_(nullptr),
      run_inference_fn_(nullptr),
      get_input_size_fn_(nullptr),
      get_output_size_fn_(nullptr),
      destroy_fn_(nullptr),
      input_size_(0),
      output_size_(0),
      initialized_(false) {}

InferenceEngine::~InferenceEngine() {
  if (cochl_api_instance_ && destroy_fn_) {
    destroy_fn_(cochl_api_instance_);
  }
  UnloadApiLibrary();
}

std::unique_ptr<InferenceEngine> InferenceEngine::Create(const std::string& model_path,
                                                           int num_threads,
                                                           bool enable_profiler) {
  std::cout << "[InferenceEngine] Creating engine for model: " << model_path
            << std::endl;

  auto engine = std::unique_ptr<InferenceEngine>(new InferenceEngine());

  // Load API shared library
  if (!engine->LoadApiLibrary(model_path)) {
    std::cerr << "[InferenceEngine] Failed to load API library" << std::endl;
    return nullptr;
  }

  // Create API instance
  engine->cochl_api_instance_ = engine->create_fn_(model_path.c_str());
  if (!engine->cochl_api_instance_) {
    std::cerr << "[InferenceEngine] Failed to create API instance" << std::endl;
    return nullptr;
  }

  // Get model input/output sizes
  engine->input_size_  = engine->get_input_size_fn_(engine->cochl_api_instance_);
  engine->output_size_ = engine->get_output_size_fn_(engine->cochl_api_instance_);

  std::cout << "[InferenceEngine] Model loaded - Input: " << engine->input_size_
            << ", Output: " << engine->output_size_ << std::endl;

  // Create profiler if enabled
  if (enable_profiler) {
    engine->profiler_ = std::make_unique<Profiler>();
    std::cout << "[InferenceEngine] Profiler enabled" << std::endl;
  }

  // Create thread pool
  size_t threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
  engine->thread_pool_ = std::make_unique<cochl::ThreadPool>(threads);
  std::cout << "[InferenceEngine] Thread pool created with " << threads << " threads"
            << std::endl;

  engine->initialized_ = true;
  return engine;
}

bool InferenceEngine::LoadApiLibrary(const std::string& model_path) {
#ifdef _WIN32
  const char* lib_name = "cochl_api.dll";
#elif __APPLE__
  const char* lib_name = "libcochl_api.dylib";
#else
  const char* lib_name = "libcochl_api.so";
#endif

  std::cout << "[InferenceEngine] Loading API library: " << lib_name << std::endl;

  api_handle_ = dlopen(lib_name, RTLD_LAZY);
  if (!api_handle_) {
    std::cerr << "[InferenceEngine] Failed to load library: " << dlerror() << std::endl;
    return false;
  }

  // Load function pointers
  create_fn_ = reinterpret_cast<CreateFn>(dlsym(api_handle_, "CochlApi_Create"));
  run_inference_fn_ =
      reinterpret_cast<RunInferenceFn>(dlsym(api_handle_, "CochlApi_RunInference"));
  get_input_size_fn_ =
      reinterpret_cast<GetInputSizeFn>(dlsym(api_handle_, "CochlApi_GetInputSize"));
  get_output_size_fn_ =
      reinterpret_cast<GetOutputSizeFn>(dlsym(api_handle_, "CochlApi_GetOutputSize"));
  destroy_fn_ = reinterpret_cast<DestroyFn>(dlsym(api_handle_, "CochlApi_Destroy"));

  if (!create_fn_ || !run_inference_fn_ || !get_input_size_fn_ || !get_output_size_fn_ ||
      !destroy_fn_) {
    std::cerr << "[InferenceEngine] Failed to load API functions" << std::endl;
    UnloadApiLibrary();
    return false;
  }

  std::cout << "[InferenceEngine] API library loaded successfully" << std::endl;
  return true;
}

void InferenceEngine::UnloadApiLibrary() {
  if (api_handle_) {
    dlclose(api_handle_);
    api_handle_ = nullptr;
  }
}

bool InferenceEngine::RunInference(const float* input, size_t input_size, float* output,
                                    size_t output_size) {
  if (!initialized_) {
    std::cerr << "[InferenceEngine] Engine not initialized" << std::endl;
    return false;
  }

  if (profiler_) {
    profiler_->StartTiming("inference");
  }

  // Run inference through API library
  bool success = run_inference_fn_(cochl_api_instance_, input, input_size, output,
                                    output_size);

  if (profiler_) {
    profiler_->StopTiming("inference");
    // Record system metrics
    cochl::kernel::SystemMonitor::RecordLatency(profiler_->GetAverageLatency());
  }

  return success;
}

bool InferenceEngine::RunInferenceAsync(const float* input, size_t input_size,
                                        float* output, size_t output_size,
                                        std::function<void(bool)> callback) {
  if (!initialized_) {
    std::cerr << "[InferenceEngine] Engine not initialized" << std::endl;
    return false;
  }

  // Submit task to thread pool
  thread_pool_->Submit([this, input, input_size, output, output_size, callback]() {
    bool success = RunInference(input, input_size, output, output_size);
    if (callback) {
      callback(success);
    }
  });

  return true;
}

std::string InferenceEngine::GetProfilingStats() const {
  if (!profiler_) {
    return "Profiler not enabled";
  }
  return profiler_->GetStats();
}

std::string InferenceEngine::GetResourceUsage() const {
  return cochl::kernel::SystemMonitor::GetSystemStatus();
}

}  // namespace cochl_sdk
