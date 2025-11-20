#include "runtime/custom_runtime.h"

#include <iostream>

namespace cochl_api {
namespace runtime {

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this]() {
      while (true) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);

          // Wait for new task or stop signal
          this->condition_.wait(lock, [this]() {
            return this->stop_ || !this->tasks_.empty();
          });

          // Exit if stopped and no tasks remaining
          if (this->stop_ && this->tasks_.empty()) {
            return;
          }

          // Get task from queue
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }

        // Execute task
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  // Notify all workers
  condition_.notify_all();

  // Wait for all workers to finish
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

// CustomRuntime implementation
CustomRuntime::CustomRuntime()
    : thread_pool_(nullptr),
      input_size_(0),
      output_size_(0),
      num_threads_(4) {
}

CustomRuntime::~CustomRuntime() = default;

bool CustomRuntime::loadModel(const char* model_path) {
  if (!model_path) {
    std::cerr << "[CustomRuntime] NULL model path" << std::endl;
    return false;
  }

  model_path_ = std::string(model_path);

  std::cout << "[CustomRuntime] Loading model from: " << model_path_ << std::endl;

  // Mock: Set fixed input/output sizes (ResNet50 compatible)
  input_size_ = 224 * 224 * 3;  // 150528
  output_size_ = 1000;

  // Initialize thread pool
  thread_pool_ = std::make_unique<ThreadPool>(num_threads_);

  std::cout << "[CustomRuntime] Model loaded successfully (Mock)" << std::endl;
  std::cout << "[CustomRuntime] Thread pool initialized with " << num_threads_ << " threads" << std::endl;

  return true;
}

bool CustomRuntime::runInference(const float* input, const std::vector<int64_t>& input_shape,
                                  float* output) {
  if (!thread_pool_) {
    std::cerr << "[CustomRuntime] Thread pool not initialized" << std::endl;
    return false;
  }

  if (!input || !output) {
    std::cerr << "[CustomRuntime] Invalid input or output pointer" << std::endl;
    return false;
  }

  if (input_shape.empty()) {
    std::cerr << "[CustomRuntime] Empty input shape" << std::endl;
    return false;
  }

  // Calculate input size from shape
  size_t input_size = 1;
  for (auto dim : input_shape) {
    input_size *= dim;
  }

  std::cout << "[CustomRuntime] Running inference with thread pool..." << std::endl;

  // Mock inference: Parallel computation using thread pool
  // Split output computation across threads
  thread_pool_->ParallelFor(0, output_size_, [this, input, output, input_size](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      // Mock computation: Simple weighted sum with some fake processing
      float sum = 0.0f;

      // Sample a few input values to create output
      for (size_t j = 0; j < 10; ++j) {
        size_t idx = (i * 17 + j * 13) % input_size;
        sum += input[idx] * 0.01f;
      }

      // Add some variation based on output index
      output[i] = sum + static_cast<float>(i % 100) * 0.001f;
    }
  });

  std::cout << "[CustomRuntime] Inference completed" << std::endl;
  return true;
}

size_t CustomRuntime::getInputSize() const {
  return input_size_;
}

size_t CustomRuntime::getOutputSize() const {
  return output_size_;
}

const char* CustomRuntime::getRuntimeType() const {
  return "Custom Backend (Thread Pool)";
}

void CustomRuntime::setNumThreads(size_t num_threads) {
  num_threads_ = num_threads;
  if (thread_pool_) {
    // Recreate thread pool with new size
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    std::cout << "[CustomRuntime] Thread pool recreated with " << num_threads_ << " threads" << std::endl;
  }
}

}  // namespace runtime
}  // namespace cochl_api
